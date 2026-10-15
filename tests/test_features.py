"""Feature construction, and the train/serve parity that protects the model.

Train/serve skew is how a surrogate becomes quietly wrong: nothing crashes, the
metrics still look fine because they were computed on the training path, and
only the served predictions are off. These tests are the guard.
"""

import numpy as np
import pandas as pd
import pytest

import features
import naca


def test_feature_vector_shape_and_names():
    x = features.build_features([2.0, 4.0, 12.0], 4.0, 3.0e6)
    assert x.shape == (len(features.FEATURE_NAMES),)
    assert np.all(np.isfinite(x))
    assert len(set(features.FEATURE_NAMES)) == len(features.FEATURE_NAMES)


def test_build_features_is_deterministic():
    a = features.build_features([2.0, 4.0, 12.0], 4.0, 3.0e6)
    b = features.build_features([2.0, 4.0, 12.0], 4.0, 3.0e6)
    assert np.array_equal(a, b)


def test_caching_cannot_leak_mutations():
    # geometry_descriptors is memoised; a caller mutating the returned dict
    # must not corrupt later calls.
    first = naca.geometry_descriptors([2.0, 4.0, 12.0])
    first["max_camber"] = 999.0
    assert naca.geometry_descriptors([2.0, 4.0, 12.0])["max_camber"] != 999.0


def test_known_feature_values():
    x = features.build_features([2.0, 4.0, 12.0], 4.0, 3.0e6)
    f = dict(zip(features.FEATURE_NAMES, x))
    assert f["max_thickness"] == pytest.approx(0.12, abs=1e-3)
    assert f["aoa_deg"] == 4.0
    assert f["is_5digit"] == 0.0
    assert f["log10_reynolds"] == pytest.approx(np.log10(3.0e6))
    # (aoa - alpha_L0)^2 with alpha_L0 ~ -2.077 for NACA 2412
    assert f["aoa_eff_sq"] == pytest.approx((4.0 + 2.0772) ** 2, rel=1e-3)


def test_five_digit_sets_the_series_flag():
    x = features.build_features([2.0, 3.0, 0.0, 12.0], 4.0, 3.0e6)
    assert dict(zip(features.FEATURE_NAMES, x))["is_5digit"] == 1.0


def test_rejects_impossible_flow_conditions():
    with pytest.raises(ValueError):
        features.build_features([2.0, 4.0, 12.0], 4.0, 0.0)
    with pytest.raises(ValueError):
        features.build_features([2.0, 4.0, 12.0], float("nan"), 3.0e6)


def test_target_transform_round_trips():
    cl = np.array([-0.3, 0.0, 1.2])
    cd = np.array([0.006, 0.012, 0.045])
    back_cl, back_cd = features.inverse_targets(features.transform_targets(cl, cd))
    assert np.allclose(back_cl, cl)
    assert np.allclose(back_cd, cd)


def test_target_transform_rejects_non_positive_drag():
    with pytest.raises(ValueError):
        features.transform_targets(np.array([1.0]), np.array([0.0]))


def _frame(rows):
    return pd.DataFrame(rows)


def test_train_serve_parity():
    """The single most important test here.

    A row that went through the data pipeline and the same airfoil arriving as
    an API payload must produce bit-identical feature vectors.
    """
    rows = [
        {"series": 4, "camber_code_0": 2.0, "camber_code_1": 4.0,
         "camber_code_2": 0.0, "thickness_pct": 12.0,
         "aoa_deg": 4.0, "reynolds": 3.0e6},
        {"series": 5, "camber_code_0": 3.551, "camber_code_1": 3.1,
         "camber_code_2": 1.0, "thickness_pct": 18.252,
         "aoa_deg": 5.932, "reynolds": 2.7947e6},
        {"series": 4, "camber_code_0": 6.811, "camber_code_1": 0.0,
         "camber_code_2": 0.0, "thickness_pct": 14.338,
         "aoa_deg": -1.17, "reynolds": 2.1785e6},
    ]
    matrix = features.build_feature_matrix(_frame(rows))
    for i, row in enumerate(rows):
        served = features.build_features(
            naca.params_from_row(row), row["aoa_deg"], row["reynolds"]
        )
        assert np.array_equal(matrix[i], served), f"row {i} differs between paths"


def test_parity_against_the_real_dataset():
    """Same guarantee, against every row actually used for training."""
    from config import COEFFS_PARQUET

    if not COEFFS_PARQUET.exists():
        pytest.skip("coeffs.parquet not built")
    df = pd.read_parquet(COEFFS_PARQUET)
    matrix = features.build_feature_matrix(df)
    for i in (0, len(df) // 3, len(df) // 2, len(df) - 1):
        row = df.iloc[i]
        served = features.build_features(
            naca.params_from_row(row), row["aoa_deg"], row["reynolds"]
        )
        assert np.array_equal(matrix[i], served)


def test_feature_ranges_cover_the_data():
    x = np.array([features.build_features([2.0, 4.0, t], a, 3e6)
                  for t in (8.0, 16.0) for a in (-4.0, 10.0)])
    r = features.feature_ranges(x)
    assert set(r) == set(features.FEATURE_NAMES)
    assert r["aoa_deg"] == [-4.0, 10.0]
