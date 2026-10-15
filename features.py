"""The single definition of AeroSurrogate's model inputs and targets.

Imported by `data.py`, `train.py`, `evaluate.py` and `api.py`. It exists as its
own module for one reason: if training and serving ever compute features
differently, the surrogate becomes quietly wrong -- not crashed, just wrong by
a few percent, with nothing to report it. `tests/test_features.py` pins the two
paths together.

Nine inputs, chosen so that one column means the same physical thing for every
row. The raw NACA digits cannot do that: a 4-digit's first parameter is maximum
camber in percent of chord, a 5-digit's is a design-lift-coefficient code, and
the dataset is a 489/511 mix of the two. Feeding those to one model as a shared
column would force it to learn a gate from the series flag, and would leave the
linear baseline structurally unable to express that gate -- which would make
"the MLP beat the baseline" an artefact of the encoding rather than a result.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import naca

#: Bump when the meaning of any feature changes. Stored in the model artifact
#: and checked at load time, so a stale model can never be served against new
#: feature code.
FEATURE_VERSION = "1.0.0"

FEATURE_NAMES: tuple[str, ...] = (
    "max_camber",       # signed maximum camber / chord
    "x_max_camber",     # chordwise position of maximum camber / chord
    "max_thickness",    # maximum thickness / chord
    "alpha_L0_deg",     # thin-airfoil-theory zero-lift angle, degrees
    "te_camber_slope",  # dy_c/dx at x/c = 0.95
    "is_5digit",        # 0 for the 4-digit series, 1 for 5-digit
    "aoa_deg",          # angle of attack, degrees
    "aoa_eff_sq",       # (aoa_deg - alpha_L0_deg)^2
    "log10_reynolds",
)

TARGET_NAMES: tuple[str, ...] = ("cl", "cd")

#: Cl stays raw because it crosses zero. Cd is modelled as log(Cd): it spans
#: roughly 0.005-0.05 and is strictly positive, so squared error in log space
#: is approximately *relative* error -- which is one of the metrics FR-7 asks
#: for -- and exp() cannot emit a negative drag coefficient. Both the MLP and
#: the linear baseline use this identical transform; comparing a model trained
#: on log(Cd) against one trained on raw Cd would rig the headline result.
TARGET_TRANSFORM: dict[str, str] = {"cl": "identity", "cd": "log"}


# ---------------------------------------------------------------------------
# Notes on features deliberately NOT included
# ---------------------------------------------------------------------------
#
# x_max_thickness: the NACA thickness distribution is a fixed shape scaled by
#   t, so its argmax sits at x/c = 0.2993 for every airfoil in the series. It is
#   a constant column, which carries no information and has zero variance to
#   standardise by.
#
# is_reflex: the 5-digit reflex digit Q provably changes nothing. In the
#   generator airfrans used, k is identically zero, so Q=0 and Q=1 produce
#   bit-identical camber lines (see naca.camber_line), and the reconstructed
#   sections match the real AirfRANS meshes to ~4e-4 chord for Q=1 cases -- so
#   the meshes were built with the same code. Carried in the parquet as
#   evidence, but excluded here rather than spending a dimension on noise.
#
# aoa_eff (the linear term): recoverable from aoa_deg and alpha_L0_deg, so
#   including it would add exact collinearity for no information. The squared
#   term is not recoverable, and drag is roughly quadratic in effective
#   incidence, so that one stays.
#
# sin/cos of the angle of attack: over a 20-degree range with no wraparound,
#   sin(a) ~ a to 0.6% and cos(a) ~ 0.99 is nearly constant. The trig encoding
#   would add collinearity and no information.


def _assemble(desc: dict, aoa_deg: float, reynolds: float) -> dict[str, float]:
    aoa_eff = float(aoa_deg) - desc["alpha_L0_deg"]
    return {
        "max_camber": desc["max_camber"],
        "x_max_camber": desc["x_max_camber"],
        "max_thickness": desc["max_thickness"],
        "alpha_L0_deg": desc["alpha_L0_deg"],
        "te_camber_slope": desc["te_camber_slope"],
        "is_5digit": desc["is_5digit"],
        "aoa_deg": float(aoa_deg),
        "aoa_eff_sq": aoa_eff**2,
        "log10_reynolds": float(np.log10(reynolds)),
    }


def build_features(
    naca_params: Sequence[float], aoa_deg: float, reynolds: float
) -> np.ndarray:
    """Feature vector from exactly what the API receives.

    `naca_params` is 3 values (4-digit: M, P, XX) or 4 (5-digit: L, P, Q, XX).
    Pure numpy -- no pyvista, no VTK, no dataset.
    """
    if reynolds <= 0:
        raise ValueError(f"reynolds must be positive, got {reynolds!r}")
    if not np.isfinite(aoa_deg):
        raise ValueError(f"aoa_deg must be finite, got {aoa_deg!r}")

    desc = naca.geometry_descriptors(naca_params)
    row = _assemble(desc, aoa_deg, reynolds)
    return np.array([row[k] for k in FEATURE_NAMES], dtype=np.float64)


def build_feature_matrix(df) -> np.ndarray:
    """Feature matrix from a `coeffs.parquet` frame, shape (N, 9).

    Goes through the *same* `build_features` the API calls, rebuilding each row
    from its raw NACA digits rather than reading the descriptor columns the
    data pipeline stored. That costs about 80 ms for 400 rows -- the geometry
    cache absorbs the repeats -- and in exchange train/serve parity is true by
    construction instead of by convention. The stored descriptor columns are
    then purely informational, and a stale one can no longer silently feed the
    model. (That is not hypothetical: the zero-lift-angle definition changed
    once already, after the CFD showed the original was wrong for 12% of rows.)
    """
    return np.vstack([
        build_features(naca.params_from_row(row), row["aoa_deg"], row["reynolds"])
        for _, row in df.iterrows()
    ])


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def transform_targets(cl: np.ndarray, cd: np.ndarray) -> np.ndarray:
    """Physical (Cl, Cd) -> model space (Cl, log Cd), shape (N, 2)."""
    cd = np.asarray(cd, dtype=np.float64)
    if np.any(cd <= 0):
        raise ValueError("Cd must be strictly positive to take its logarithm")
    return np.column_stack([np.asarray(cl, dtype=np.float64), np.log(cd)])


def inverse_targets(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Model space (Cl, log Cd) -> physical (Cl, Cd).

    Note `exp(E[log Cd])` is median- rather than mean-unbiased. With the
    residual spread seen here the Jensen bias is around 0.1%, so no smearing
    correction is applied -- but the README says so rather than leaving it
    implicit.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    return y[:, 0], np.exp(y[:, 1])


def feature_ranges(x: np.ndarray) -> dict[str, list[float]]:
    """Per-feature [min, max] over a matrix, for the API's extrapolation warnings."""
    return {
        name: [float(x[:, i].min()), float(x[:, i].max())]
        for i, name in enumerate(FEATURE_NAMES)
    }
