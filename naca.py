"""NACA 4- and 5-digit geometry, and AirfRANS simulation-name parsing.

Pure numpy on purpose. `camber_line` / `naca_generator` are vendored from the
airfrans library (MIT, https://github.com/Extrality/airfrans_lib, file
`src/airfrans/naca_generator.py`) rather than imported, because
`import airfrans` pulls in pyvista and therefore VTK -- about 100 MB that the
serving environment has no other reason to carry.

Two deliberate differences from upstream, both guards rather than behaviour
changes on real data (`tests/test_geometry.py` pins the two implementations
together to 1e-10 on every parameter set in the dataset):

  * the 5-digit Newton solve for `m` gets an iteration cap;
  * the 4-digit `p == 0` / `p == 1` special cases use a tolerance instead of an
    exact float comparison, since `p = 1e-9` would otherwise divide by `p**2`.

Note the upstream docstring for `camber_line` is wrong: it says `params` has
shape (3) or (4), but the code branches on `len(params)` being 2 or 3 -- the
camber digits only, with thickness dropped. `naca_generator` takes the full 3
or 4. We follow the code.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from config import CHORD, NU_GEN

# numpy 2 renamed trapz -> trapezoid; support both.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

#: Chordwise grid used for every geometric descriptor. Frozen as a module
#: constant because `max_camber` is a max over a sampled grid -- if the data
#: pipeline used 2001 points and the API used 400, the two would silently
#: disagree in the fourth decimal and nothing would ever report it.
N_GRID = 2001
THETA_GRID = np.linspace(0.0, np.pi, N_GRID)
X_GRID = 0.5 * (1.0 - np.cos(THETA_GRID))

#: Chordwise station at which the trailing-edge camber slope is sampled.
TE_SLOPE_X = 0.95

_TOL = 1e-9


class NacaError(ValueError):
    """Raised for NACA parameters this module cannot interpret."""


# ---------------------------------------------------------------------------
# Vendored geometry
# ---------------------------------------------------------------------------


def thickness_dist(t: float, x: np.ndarray, CTE: bool = True) -> np.ndarray:
    """Half-thickness distribution, `t` as a fraction of chord (0.12, not 12)."""
    a = -0.1036 if CTE else -0.1015
    return 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        + a * x**4
    )


def camber_line(params: Sequence[float], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camber line and its slope for the NACA 4- and 5-digit series.

    Args:
        params: camber digits only -- 2 values (M, P) for 4-digit, 3 values
            (L, P, Q) for 5-digit. Thickness is NOT included.
        x: abscissas in chord units.

    Returns:
        ``(y_c, dy_c)``, both shaped like `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    y_c = np.zeros_like(x)
    dy_c = np.zeros_like(x)

    if len(params) == 2:
        m = float(params[0]) / 100.0
        p = float(params[1]) / 10.0

        # Upstream compares p to 0 and 1 exactly; a tolerance avoids dividing
        # by p**2 for a p that is small but not exactly zero.
        if abs(p) < _TOL:
            return y_c, -2 * m * x
        if abs(p - 1.0) < _TOL:
            return y_c, 2 * m * (1 - x)

        fore = x < p
        aft = ~fore
        y_c[fore] = (m / p**2) * (2 * p * x[fore] - x[fore] ** 2)
        dy_c[fore] = (2 * m / p**2) * (p - x[fore])
        y_c[aft] = (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x[aft] - x[aft] ** 2)
        dy_c[aft] = (2 * m / (1 - p) ** 2) * (p - x[aft])
        return y_c, dy_c

    if len(params) == 3:
        ell, p, q = (float(v) for v in params)
        c_l, x_f = 3.0 / 20.0 * ell, p / 20.0

        m = _solve_five_digit_m(x_f)
        r = (3 * m - 7 * m**2 + 8 * m**3 - 4 * m**4) / np.sqrt(m * (1 - m)) - 1.5 * (
            1 - 2 * m
        ) * (np.pi / 2 - np.arcsin(1 - 2 * m))
        k_1 = c_l / r

        fore = x <= m
        aft = ~fore

        q_int = int(round(q))
        if abs(q - q_int) > 1e-6 or q_int not in (0, 1):
            raise NacaError(
                f"5-digit Q must be 0 (normal camber) or 1 (reflex camber), got {q!r}"
            )

        # NOTE: upstream's reflex branch is degenerate, and we reproduce it
        # faithfully rather than "fixing" it. `m` solves m - m**1.5/sqrt(3) = x_f,
        # i.e. m - x_f = m**1.5/sqrt(3), so 3*(m - x_f)**2 == m**3 identically and
        # therefore k == 0 for every valid x_f. At k = 0 the q == 1 expressions
        # reduce algebraically to the q == 1 ones, so Q makes no difference to the
        # generated camber line. Verified numerically: NACA 23012 and 23112 produce
        # bit-identical y_c and dy_c.
        #
        # Consequence for modelling: no geometric descriptor can detect reflex, so
        # `is_reflex` is carried as a raw digit flag instead. Whether the digit means
        # anything at all depends on whether AirfRANS' meshes were generated with this
        # same code -- `data.py check-geometry` settles that against the real meshes.
        if q_int == 0:
            y_c[fore] = k_1 * (
                x[fore] ** 3 - 3 * m * x[fore] ** 2 + m**2 * (3 - m) * x[fore]
            )
            dy_c[fore] = k_1 * (3 * x[fore] ** 2 - 6 * m * x[fore] + m**2 * (3 - m))
            y_c[aft] = k_1 * m**3 * (1 - x[aft])
            dy_c[aft] = -k_1 * m**3 * np.ones_like(dy_c[aft])
        else:
            k = (3 * (m - x_f) ** 2 - m**3) / (1 - m) ** 3
            y_c[fore] = k_1 * (
                (x[fore] - m) ** 3 - k * (1 - m) ** 3 * x[fore] - m**3 * x[fore] + m**3
            )
            dy_c[fore] = k_1 * (3 * (x[fore] - m) ** 2 - k * (1 - m) ** 3 - m**3)
            y_c[aft] = k_1 * (
                k * (x[aft] - m) ** 3 - k * (1 - m) ** 3 * x[aft] - m**3 * x[aft] + m**3
            )
            dy_c[aft] = k_1 * (3 * k * (x[aft] - m) ** 2 - k * (1 - m) ** 3 - m**3)
        return y_c, dy_c

    raise NacaError(
        "camber_line expects 2 digits (4-digit series) or 3 digits (5-digit series), "
        f"got {len(params)}"
    )


def _solve_five_digit_m(x_f: float, max_iter: int = 100) -> float:
    """Newton solve of ``x(1 - sqrt(x/3)) = x_f`` for the 5-digit camber.

    Upstream loops until convergence with no iteration cap; we bound it.
    """
    old_m = 0.5
    for _ in range(max_iter):
        f = old_m * (1 - np.sqrt(old_m / 3)) - x_f
        df = 1 - 1.5 * np.sqrt(old_m / 3)
        new_m = max(old_m - f / df, 0.0)
        if abs(old_m - new_m) <= 1e-15:
            return new_m
        old_m = new_m
    raise NacaError(f"5-digit camber solve did not converge for x_f={x_f!r}")


def naca_section(
    params: Sequence[float],
    nb_samples: int = 400,
    scale: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    cosine_spacing: bool = True,
    CTE: bool = True,
) -> np.ndarray:
    """Closed airfoil outline, upper surface then lower surface reversed.

    Mirrors ``airfrans.naca_generator.naca_generator`` (minus its `verbose`
    printing). Returns ``(2 * nb_samples + 1, 2)``.
    """
    camber_params, t = split_params(params)

    if cosine_spacing:
        beta = np.pi * np.linspace(1, 0, nb_samples + 1)
        x = (1 - np.cos(beta)) / 2
    else:
        x = np.linspace(1, 0, nb_samples + 1)

    y_c, dy_c = camber_line(camber_params, x)
    y_t = thickness_dist(t, x, CTE)
    theta = np.arctan(dy_c)

    x_u, x_l = x - y_t * np.sin(theta), x + y_t * np.sin(theta)
    y_u, y_l = y_c + y_t * np.cos(theta), y_c - y_t * np.cos(theta)

    xs = np.concatenate([x_u, x_l[:-1][::-1]], axis=0)
    ys = np.concatenate([y_u, y_l[:-1][::-1]], axis=0)
    pos = np.stack([xs * scale + origin[0], ys * scale + origin[1]], axis=-1)
    pos[0] = pos[-1] = np.array([1.0, 0.0]) * scale + np.array(origin)
    return pos


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def split_params(naca_params: Sequence[float]) -> tuple[list[float], float]:
    """Split full NACA parameters into (camber digits, thickness fraction).

    3 values are the 4-digit series (M, P, XX); 4 values are the 5-digit
    series (L, P, Q, XX). In both cases the last value is thickness in percent
    of chord, which is why 4-digit has 3 parameters and not 4.
    """
    vals = [float(v) for v in naca_params]
    if len(vals) not in (3, 4):
        raise NacaError(
            "NACA parameters must be 3 values (4-digit: M, P, XX) or 4 values "
            f"(5-digit: L, P, Q, XX), got {len(vals)}"
        )
    if not all(np.isfinite(v) for v in vals):
        raise NacaError(f"NACA parameters must all be finite, got {vals!r}")
    return vals[:-1], vals[-1] / 100.0


def is_five_digit(naca_params: Sequence[float]) -> bool:
    return len(naca_params) == 4


def params_from_row(row) -> list[float]:
    """Rebuild the NACA parameter list from a `coeffs.parquet` row.

    Works with a pandas Series or a plain mapping. The 5-digit Q column is
    stored as 0.0 for 4-digit rows, so the series flag -- not the value -- is
    what decides the length.
    """
    if int(row["series"]) == 5:
        return [float(row["camber_code_0"]), float(row["camber_code_1"]),
                float(row["camber_code_2"]), float(row["thickness_pct"])]
    return [float(row["camber_code_0"]), float(row["camber_code_1"]),
            float(row["thickness_pct"])]


def parse_simulation_name(name: str) -> dict:
    """Decode an AirfRANS simulation name.

    Format: ``airFoil2D_<turbulence>_<U_inf>_<aoa_deg>_<naca params...>``.
    Count the trailing tokens to tell the series apart -- never index from the
    right, since 4-digit has 3 parameters and 5-digit has 4.

    >>> r = parse_simulation_name("airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32")
    >>> r["series"], r["aoa_deg"]
    (4, -4.156)
    """
    tokens = name.split("_")
    if len(tokens) < 7 or tokens[0] != "airFoil2D":
        raise NacaError(f"not an AirfRANS simulation name: {name!r}")

    try:
        u_inf = float(tokens[2])
        aoa_deg = float(tokens[3])
        naca_params = [float(t) for t in tokens[4:]]
    except ValueError as exc:
        raise NacaError(f"could not parse numbers out of {name!r}: {exc}") from exc

    if len(naca_params) not in (3, 4):
        raise NacaError(
            f"{name!r} has {len(naca_params)} NACA parameters; expected 3 or 4"
        )

    return {
        "name": name,
        "turbulence_model": tokens[1],
        "u_inf": u_inf,
        "aoa_deg": aoa_deg,
        "naca_params": naca_params,
        "series": 5 if len(naca_params) == 4 else 4,
        "reynolds": u_inf * CHORD / NU_GEN,
    }


# ---------------------------------------------------------------------------
# Physical descriptors
# ---------------------------------------------------------------------------


def _alpha_L0_from_slope(dy_c: np.ndarray) -> float:
    """Zero-lift angle in degrees, given dy_c/dx already sampled on X_GRID."""
    integral = _trapezoid(dy_c * (np.cos(THETA_GRID) - 1.0), THETA_GRID)
    return float(np.degrees(-integral / np.pi))


def zero_lift_angle_deg(camber_params: Sequence[float]) -> float:
    """Thin-airfoil-theory zero-lift angle, in degrees.

        alpha_L0 = -(1/pi) * integral_0^pi (dy_c/dx)(cos(theta) - 1) d(theta)

    with ``x = (1 - cos theta)/2``. Thin-airfoil theory then gives
    ``Cl ~ 2*pi*(alpha - alpha_L0)``, which is why this single number is the
    most informative shape feature available: it collapses all of the camber
    information into the quantity that determines lift to first order, and it
    is directly comparable between the 4- and 5-digit series, whose raw digits
    mean different things.

    Returns 0.0 for any symmetric section. Uses the numerically differentiated
    camber line, for the reason documented in `_geometry_descriptors`.
    """
    y_c, _ = camber_line(camber_params, X_GRID)
    return _alpha_L0_from_slope(np.gradient(y_c, X_GRID))


@lru_cache(maxsize=8192)
def _descriptors_cached(params: tuple[float, ...]) -> tuple[tuple[str, float], ...]:
    return tuple(_geometry_descriptors(list(params)).items())


def geometry_descriptors(naca_params: Sequence[float]) -> dict:
    """Cached wrapper around :func:`_geometry_descriptors`.

    The demo redraws a whole angle-of-attack sweep on every slider move -- same
    geometry, 201 different angles -- so without memoisation the section would
    be rebuilt 201 times identically. A fresh dict is returned each call so
    callers cannot mutate the cache.
    """
    return dict(_descriptors_cached(tuple(float(v) for v in naca_params)))


def _geometry_descriptors(naca_params: Sequence[float]) -> dict:
    """Series-agnostic physical descriptors of a NACA section.

    These exist because the raw digits are not comparable across series: a
    4-digit's first parameter is max camber in percent of chord, while a
    5-digit's is a design-lift-coefficient code. Feeding those to one model as
    though they were the same column would force it to learn a gate from the
    series flag -- and would leave the linear baseline structurally unable to
    express it, which would make "the MLP beat the baseline" an artefact of
    the encoding rather than a result.
    """
    camber_params, t = split_params(naca_params)

    # One pass over the grid for the camber line, reused for every camber-derived
    # descriptor including the zero-lift angle. Calling zero_lift_angle_deg()
    # here instead would evaluate camber_line a second time for nothing.
    y_c, _ = camber_line(camber_params, X_GRID)
    y_t = thickness_dist(t, X_GRID)

    # Differentiate y_c rather than using the analytic slope camber_line also
    # returns, because in the 4-digit p == 0 branch upstream returns y_c == 0
    # alongside dy_c = -2*m*x -- a slope that contradicts its own camber line.
    # 48 of the 200 four-digit simulations here have exactly p == 0, so this is
    # 12% of the dataset, not a corner case.
    #
    # The CFD settles which is right. Inverting Cl ~ 2*pi*(aoa - alpha_L0) over
    # those 48 simulations gives an implied alpha_L0 of +0.24 +/- 0.46 degrees,
    # i.e. zero, against an analytic value averaging -3.1 and reaching -6.0; the
    # correlation between analytic and implied is 0.13. Differentiating y_c
    # gives exactly 0.0 for them, and reproduces the analytic value to four
    # decimals wherever the two are consistent (NACA 2412 -> -2.0772 either
    # way). Across the whole dataset it lifts the correlation with the implied
    # zero-lift angle from 0.78 to 0.94.
    #
    # naca_section() deliberately still uses the analytic slope: AirfRANS' own
    # meshes were generated with it, and the reconstruction matches them to
    # ~4e-4 chord. The geometry is upstream's; only its aerodynamic description
    # is corrected here.
    dy_c = np.gradient(y_c, X_GRID)

    # argmax over a frozen grid. For an exactly symmetric section y_c is all
    # zeros and this reports x_max_camber = 0.0, but max_camber is then 0.0 too,
    # so the pair stays self-consistent. There is no discontinuity for merely
    # near-symmetric sections: argmax is unchanged for any non-zero camber.
    i_camber = int(np.argmax(np.abs(y_c)))
    i_thick = int(np.argmax(y_t))

    is_reflex = (
        1.0 if len(camber_params) == 3 and int(round(float(camber_params[2]))) == 1
        else 0.0
    )

    return {
        "max_camber": float(y_c[i_camber]),          # signed, /chord
        "x_max_camber": float(X_GRID[i_camber]),     # /chord
        "max_thickness": float(2.0 * y_t[i_thick]),  # /chord; equals t by construction
        "x_max_thickness": float(X_GRID[i_thick]),   # /chord
        "te_camber_slope": float(np.interp(TE_SLOPE_X, X_GRID, dy_c)),
        "alpha_L0_deg": _alpha_L0_from_slope(dy_c),
        "is_5digit": 1.0 if len(camber_params) == 3 else 0.0,
        "is_reflex": is_reflex,
    }
