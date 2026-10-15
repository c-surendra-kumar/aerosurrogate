"""Name parsing, Reynolds number and NACA geometry."""

import numpy as np
import pytest

import naca
from config import NU_GEN

# Real names from the dataset, covering both series and a negative incidence.
FOUR = "airFoil2D_SST_31.283_-4.156_0.919_6.98_14.32"
FIVE = "airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252"


def test_parses_four_digit():
    r = naca.parse_simulation_name(FOUR)
    assert r["series"] == 4
    assert r["u_inf"] == pytest.approx(31.283)
    assert r["aoa_deg"] == pytest.approx(-4.156)
    assert r["naca_params"] == [0.919, 6.98, 14.32]


def test_parses_five_digit():
    r = naca.parse_simulation_name(FIVE)
    assert r["series"] == 5
    assert r["naca_params"] == [3.551, 3.1, 1.0, 18.252]


def test_series_is_decided_by_token_count_not_position():
    # 4-digit has 3 parameters and 5-digit has 4, because the last two digits
    # are a single thickness value. Indexing from the right would confuse them.
    assert len(naca.parse_simulation_name(FOUR)["naca_params"]) == 3
    assert len(naca.parse_simulation_name(FIVE)["naca_params"]) == 4


@pytest.mark.parametrize("bad", ["", "nonsense", "airFoil2D_SST_1_2", "foo_SST_1_2_3_4_5"])
def test_rejects_malformed_names(bad):
    with pytest.raises(naca.NacaError):
        naca.parse_simulation_name(bad)


def test_reynolds_uses_the_generation_viscosity():
    # Re = U * c / nu with c = 1 and nu = 1.56e-5, the value the AirfRANS paper
    # reports for dataset generation -- NOT Simulation.NU (1.5498e-5), which
    # would put the dataset's Reynolds range outside its advertised 2-6 million.
    assert NU_GEN == 1.56e-5
    assert naca.parse_simulation_name(FOUR)["reynolds"] == pytest.approx(2.0053e6, rel=1e-4)
    assert 93.592 / NU_GEN == pytest.approx(5.9995e6, rel=1e-4)


@pytest.mark.parametrize(
    "params, camber, x_camber, thickness, alpha_l0",
    [
        ([0.0, 0.0, 12.0], 0.0, None, 0.12, 0.0),      # symmetric
        ([2.0, 4.0, 12.0], 0.02, 0.40, 0.12, -2.08),   # the textbook NACA 2412
        ([4.0, 4.0, 12.0], 0.04, 0.40, 0.12, -4.15),
    ],
)
def test_textbook_sections(params, camber, x_camber, thickness, alpha_l0):
    d = naca.geometry_descriptors(params)
    assert d["max_camber"] == pytest.approx(camber, abs=2e-4)
    assert d["max_thickness"] == pytest.approx(thickness, abs=1e-3)
    assert d["alpha_L0_deg"] == pytest.approx(alpha_l0, abs=0.02)
    if x_camber is not None:
        assert d["x_max_camber"] == pytest.approx(x_camber, abs=2e-3)


def test_max_thickness_equals_the_last_digit():
    for t in (6.0, 12.0, 19.5):
        assert naca.geometry_descriptors([2.0, 4.0, t])["max_thickness"] == \
            pytest.approx(t / 100, abs=1e-3)


def test_degenerate_four_digit_sections_are_uncambered():
    """P == 0 is 24% of the four-digit simulations in this dataset.

    Upstream's camber_line returns y_c == 0 there but a non-zero slope
    dy_c = -2*m*x, which contradicts it. Inverting the CFD lift coefficients
    for those simulations gives an implied zero-lift angle of +0.24 +/- 0.46
    degrees, so they really are uncambered; our descriptors must say so.
    """
    for m in (0.919, 3.0, 6.811):
        d = naca.geometry_descriptors([m, 0.0, 14.0])
        assert d["max_camber"] == pytest.approx(0.0, abs=1e-12)
        assert d["alpha_L0_deg"] == pytest.approx(0.0, abs=1e-9)
        assert d["te_camber_slope"] == pytest.approx(0.0, abs=1e-9)


def test_five_digit_reflex_digit_changes_nothing():
    """Documents an upstream degeneracy rather than asserting it is correct.

    m solves m - m**1.5/sqrt(3) = x_f, which forces 3*(m - x_f)**2 == m**3, so
    upstream's k is identically zero and the Q=1 branch collapses onto Q=0.
    The real AirfRANS meshes match this, so the digit carries no information --
    which is why `is_reflex` is not a model feature.
    """
    a = naca.geometry_descriptors([2.0, 3.0, 0.0, 12.0])
    b = naca.geometry_descriptors([2.0, 3.0, 1.0, 12.0])
    for key in ("max_camber", "x_max_camber", "alpha_L0_deg", "te_camber_slope"):
        assert a[key] == pytest.approx(b[key], abs=1e-12)


def test_rejects_bad_parameter_counts_and_q():
    with pytest.raises(naca.NacaError):
        naca.geometry_descriptors([1.0, 2.0])
    with pytest.raises(naca.NacaError):
        naca.geometry_descriptors([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(naca.NacaError):
        naca.geometry_descriptors([2.0, 3.0, 7.0, 12.0])   # Q must be 0 or 1


def test_descriptors_are_finite_over_the_parameter_space():
    rng = np.random.default_rng(0)
    for _ in range(300):
        p = ([rng.uniform(0, 7), rng.uniform(0, 7), rng.uniform(5, 20)]
             if rng.integers(2) else
             [rng.uniform(0, 6), rng.uniform(3, 8), float(rng.integers(2)),
              rng.uniform(5, 20)])
        assert all(np.isfinite(v) for v in naca.geometry_descriptors(p).values())


def test_section_is_closed_and_unit_chord():
    pts = naca.naca_section([2.0, 4.0, 12.0], nb_samples=200)
    assert pts.shape == (401, 2)
    assert np.allclose(pts[0], pts[-1])
    # The surface wraps slightly around the leading edge, so x dips a hair
    # below zero there; the chord is still essentially [0, 1].
    assert -1e-3 < pts[:, 0].min() < 1e-3
    assert pts[:, 0].max() == pytest.approx(1.0, abs=1e-6)
    assert abs(pts[:, 1]).max() < 0.5


def test_vendored_geometry_matches_upstream():
    """Our vendored copy must stay numerically identical to airfrans'.

    Skipped when airfrans is absent, which is the normal state of the serving
    environment -- vendoring exists precisely so serving needs no pyvista/VTK.
    """
    up = pytest.importorskip("airfrans.naca_generator")
    rng = np.random.default_rng(1)
    for _ in range(60):
        if rng.integers(2):
            p = [rng.uniform(0.05, 6), rng.uniform(1.0, 9.0), rng.uniform(5, 20)]
        else:
            p = [rng.uniform(0.05, 6), rng.uniform(3.0, 7.9), float(rng.integers(2)),
                 rng.uniform(5, 20)]
        camber = p[:-1]
        ours = naca.camber_line(camber, naca.X_GRID)
        theirs = up.camber_line(np.array(camber), naca.X_GRID.copy())
        assert np.allclose(ours[0], theirs[0], atol=1e-10)
        assert np.allclose(ours[1], theirs[1], atol=1e-10)
        assert np.allclose(
            naca.naca_section(p, nb_samples=200),
            up.naca_generator(np.array(p), nb_samples=200, verbose=False),
            atol=1e-10,
        )
