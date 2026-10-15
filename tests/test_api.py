"""The HTTP contract, the saved artifact, and the acceptance criterion."""

import json

import numpy as np
import pytest

from config import ARTIFACT_DIR, METRICS_JSON, MODEL_PT

pytestmark = pytest.mark.skipif(
    not MODEL_PT.exists(), reason="no trained model; run train.py first"
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    import api

    with TestClient(api.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["ensemble_members"] >= 1


def test_meta_exposes_the_training_envelope(client):
    body = client.get("/meta").json()
    assert body["feature_names"]
    assert "aoa_deg" in body["input_ranges"]
    lo, hi = body["input_ranges"]["aoa_deg"]
    assert lo < hi


def test_predict_matches_the_spec_contract(client):
    """The exact payload from the project spec, and exactly the keys it names."""
    r = client.post("/predict", json={
        "naca": [2.0, 4.0, 12.0], "aoa_deg": 4.0, "reynolds": 3.0e6
    })
    assert r.status_code == 200
    body = r.json()
    for key in ("cl", "cd", "inference_ms"):
        assert isinstance(body[key], float)
    assert body["inference_ms"] > 0
    assert body["cd"] > 0                       # log target guarantees positivity
    assert -2.0 < body["cl"] < 3.0


def test_five_digit_payload_is_accepted(client):
    r = client.post("/predict", json={
        "naca": [3.551, 3.1, 1.0, 18.252], "aoa_deg": 5.932, "reynolds": 2.7947e6
    })
    assert r.status_code == 200


@pytest.mark.parametrize("payload", [
    {"naca": [1, 2, 3, 4, 5], "aoa_deg": 4.0, "reynolds": 3e6},   # too many digits
    {"naca": [2.0, 4.0], "aoa_deg": 4.0, "reynolds": 3e6},        # too few
    {"naca": [2.0, 4.0, 0.0], "aoa_deg": 4.0, "reynolds": 3e6},   # zero thickness
    {"naca": [2.0, 4.0, 12.0], "aoa_deg": 4.0, "reynolds": -1},   # negative Re
    {"naca": [2.0, 3.0, 7.0, 12.0], "aoa_deg": 4.0, "reynolds": 3e6},  # Q not 0/1
    {"naca": [2.0, 4.0, 12.0], "aoa_deg": 400.0, "reynolds": 3e6},     # absurd angle
])
def test_impossible_input_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_out_of_range_input_is_answered_with_a_warning(client):
    """FR-12 is about showing where the model fails, so extrapolation must not
    be a hard error -- it is answered, flagged, and left to the caller."""
    r = client.post("/predict", json={
        "naca": [2.0, 4.0, 12.0], "aoa_deg": 45.0, "reynolds": 3.0e6
    })
    assert r.status_code == 200
    assert r.json()["extrapolation"]["in_range"] is False
    assert any("aoa_deg" in w for w in r.json()["extrapolation"]["warnings"])


def test_in_range_input_is_not_flagged(client):
    r = client.post("/predict", json={
        "naca": [2.0, 4.0, 12.0], "aoa_deg": 4.0, "reynolds": 3.0e6
    })
    assert r.json()["extrapolation"]["in_range"] is True


def test_sweep_returns_consistent_arrays(client):
    r = client.post("/sweep", json={
        "naca": [2.0, 4.0, 12.0], "reynolds": 3.0e6,
        "aoa_min": -5, "aoa_max": 15, "n": 41,
    })
    assert r.status_code == 200
    body = r.json()
    for key in ("aoa_deg", "cl", "cd", "cl_std", "cd_std", "in_range"):
        assert len(body[key]) == 41
    # Lift must increase with incidence over the linear range.
    assert body["cl"][-1] > body["cl"][0]


def test_sweep_agrees_with_predict(client):
    """One code path, two entry points -- they must not drift."""
    sw = client.post("/sweep", json={
        "naca": [2.0, 4.0, 12.0], "reynolds": 3.0e6,
        "aoa_min": 0.0, "aoa_max": 10.0, "n": 11,
    }).json()
    pr = client.post("/predict", json={
        "naca": [2.0, 4.0, 12.0], "aoa_deg": 5.0, "reynolds": 3.0e6
    }).json()
    assert sw["cl"][5] == pytest.approx(pr["cl"], rel=1e-9)
    assert sw["cd"][5] == pytest.approx(pr["cd"], rel=1e-9)


def test_geometry_returns_a_closed_section(client):
    body = client.get("/geometry", params={"naca_params": "2,4,12"}).json()
    assert len(body["x"]) == len(body["y"]) > 50
    assert body["x"][0] == pytest.approx(body["x"][-1])


def test_geometry_rejects_nonsense(client):
    assert client.get("/geometry", params={"naca_params": "nope"}).status_code == 422


def test_reference_lookup_finds_an_exact_test_case(client):
    """FR-10: feeding back a stored simulation must surface its CFD values."""
    cases = client.get("/test-cases").json()
    if not cases:
        pytest.skip("no test_reference.json; run evaluate.py first")
    case = cases[0]
    body = client.post("/predict", json={
        "naca": case["naca"], "aoa_deg": case["aoa_deg"], "reynolds": case["reynolds"]
    }).json()
    assert body["reference"] is not None
    assert body["reference"]["name"] == case["name"]
    assert body["reference"]["cl"] == pytest.approx(case["cl"])


def test_reference_lookup_returns_nothing_for_an_unseen_airfoil(client):
    body = client.post("/predict", json={
        "naca": [1.234, 5.678, 9.012], "aoa_deg": 3.21, "reynolds": 4.321e6
    }).json()
    assert body["reference"] is None


def test_artifact_loads_without_pickle_and_without_sklearn():
    """torch.load(weights_only=True) is the torch >= 2.6 default; the artifact
    must stay plain tensors so it keeps loading and cannot execute code."""
    import torch

    state = torch.load(MODEL_PT, map_location="cpu", weights_only=True)
    assert {"members", "x_mean", "x_std", "y_mean", "y_std"} <= set(state)
    meta = json.loads((ARTIFACT_DIR / "model_meta.json").read_text())
    assert meta["feature_version"]
    assert len(meta["feature_names"]) == state["x_mean"].numel()


def test_numpy_inference_matches_torch():
    """Serving runs the forward pass in numpy for speed; it must agree with the
    torch modules the weights were trained as."""
    import torch

    from model import load_bundle

    bundle = load_bundle()
    x = np.array([[0.02, 0.4, 0.12, -2.1, -0.06, 0.0, 4.0, 37.0, 6.48]])
    xs = (x - bundle.x_mean) / bundle.x_std
    with torch.inference_mode():
        ref = torch.stack([
            m(torch.tensor(xs, dtype=torch.float32)) for m in bundle.members
        ]).numpy()
    assert np.allclose(bundle._forward(xs), ref, atol=1e-5)


@pytest.mark.skipif(not METRICS_JSON.exists(), reason="run evaluate.py first")
def test_acceptance_criterion_holds():
    """The project's pass/fail bar, as an executable assertion."""
    m = json.loads(METRICS_JSON.read_text())
    if m["data"].get("synthetic"):
        pytest.skip("metrics were computed on synthetic data")
    assert m["models"]["mlp"]["cl"]["mae"] < m["models"]["linear"]["cl"]["mae"]
    assert m["models"]["mlp"]["cd"]["mae"] < m["models"]["linear"]["cd"]["mae"]
    # And the margin must exceed the noise of a 200-row test set.
    assert m["comparison_linear_vs_mlp"]["cl"]["significant"]
    assert m["comparison_linear_vs_mlp"]["cd"]["significant"]


@pytest.mark.skipif(not METRICS_JSON.exists(), reason="run evaluate.py first")
def test_lift_curve_slope_is_physical():
    """A surrogate that fits the numbers but gets the physics wrong is not useful.
    Thin-airfoil theory puts the lift slope at 2*pi/rad = 0.1097 per degree."""
    m = json.loads(METRICS_JSON.read_text())
    if m["data"].get("synthetic"):
        pytest.skip("metrics were computed on synthetic data")
    assert 0.75 < m["lift_curve"]["slope_ratio"] < 1.15
