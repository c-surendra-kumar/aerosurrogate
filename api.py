"""FastAPI service for AeroSurrogate, and host for the demo page.

    uvicorn api:app --reload      ->  http://127.0.0.1:8000

Needs only `artifacts/model.pt`, `artifacts/model_meta.json` and
`web/test_reference.json`, all of which are small enough to live in the repo.
No dataset, no pyvista, no scikit-learn.
"""

from __future__ import annotations

import json
import math
import time
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

import features
import naca
from config import ARTIFACT_DIR, TEST_REFERENCE_JSON, WEB_DIR
from model import load_bundle

#: A query counts as "this simulation" when no feature differs from the stored
#: one by more than 5% of a training standard deviation. L-infinity in
#: standardised space keeps that statement to one sentence, needs no per-feature
#: weights, and automatically ignores features the model ignores.
MATCH_TAU = 0.05


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(1)
    app.state.bundle = load_bundle(
        ARTIFACT_DIR / "model.pt", ARTIFACT_DIR / "model_meta.json"
    )
    try:
        app.state.reference = json.loads(TEST_REFERENCE_JSON.read_text(encoding="utf-8"))
    except FileNotFoundError:
        app.state.reference = []
    if app.state.reference:
        app.state.reference_x = np.vstack([
            features.build_features(r["naca"], r["aoa_deg"], r["reynolds"])
            for r in app.state.reference
        ])
    else:
        app.state.reference_x = np.empty((0, len(features.FEATURE_NAMES)))
    yield


app = FastAPI(
    title="AeroSurrogate",
    description="Neural surrogate for airfoil lift and drag coefficients.",
    version=features.FEATURE_VERSION,
    lifespan=lifespan,
)

# The page is served from this same app, so requests are same-origin and CORS
# is not involved at all. This block only exists so the page still works if
# someone opens web/index.html straight off disk during development.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|null)$",
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def validate_naca(v: list[float]) -> list[float]:
    """Reject physically impossible geometry, shared by every endpoint.

    This is the hard boundary. Merely *unusual* input -- an airfoil or an angle
    outside the training envelope -- is answered with a warning instead, since
    showing where the model stops being trustworthy is the point of the demo.
    """
    if not all(math.isfinite(x) for x in v):
        raise ValueError("every NACA parameter must be finite")
    if not 0.5 < v[-1] <= 40.0:
        raise ValueError(
            "the last NACA parameter is thickness in percent of chord and must "
            f"lie in (0.5, 40]; got {v[-1]}"
        )
    if len(v) == 4 and round(v[2]) not in (0, 1):
        raise ValueError("for the 5-digit series Q must be 0 (normal) or 1 (reflex)")
    return v


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    naca: Annotated[list[float], Field(min_length=3, max_length=4)] = Field(
        ..., description="4-digit [M, P, XX] or 5-digit [L, P, Q, XX]",
        examples=[[2.0, 4.0, 12.0]],
    )
    aoa_deg: float = Field(..., examples=[4.0])
    reynolds: float = Field(..., gt=0, examples=[3.0e6])

    @field_validator("naca")
    @classmethod
    def _check_naca(cls, v: list[float]) -> list[float]:
        return validate_naca(v)

    @field_validator("aoa_deg")
    @classmethod
    def _check_aoa(cls, v: float) -> float:
        if not math.isfinite(v) or abs(v) > 90:
            raise ValueError("aoa_deg must be finite and within +/-90 degrees")
        return v


class Extrapolation(BaseModel):
    in_range: bool
    warnings: list[str] = []
    fields: dict = {}


class Reference(BaseModel):
    name: str
    cl: float
    cd: float
    distance: float
    exact: bool


class PredictResponse(BaseModel):
    cl: float
    cd: float
    inference_ms: float
    cl_std: float
    cd_std: float
    extrapolation: Extrapolation
    reference: Reference | None = None


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    naca: Annotated[list[float], Field(min_length=3, max_length=4)]
    reynolds: float = Field(..., gt=0)
    aoa_min: float = -5.0
    aoa_max: float = 15.0
    n: int = Field(61, ge=2, le=501)

    @field_validator("naca")
    @classmethod
    def _check_naca(cls, v: list[float]) -> list[float]:
        return validate_naca(v)


# ---------------------------------------------------------------------------
# Routes  (all declared before the StaticFiles mount, which is a catch-all)
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "feature_version": features.FEATURE_VERSION,
        "ensemble_members": len(app.state.bundle.members),
        "reference_simulations": len(app.state.reference),
    }


@app.get("/meta")
def meta() -> dict:
    """Everything the page needs to describe the model and draw its limits."""
    b = app.state.bundle
    return {
        "feature_names": list(features.FEATURE_NAMES),
        "feature_version": features.FEATURE_VERSION,
        "target_transform": features.TARGET_TRANSFORM,
        "input_ranges": b.input_ranges,
        "arch": b.meta["arch"],
        "trained_on": b.meta["data"],
        "created_at": b.meta["created_at"],
        "inference_ms_definition": (
            "feature construction + standardisation + ensemble forward pass + "
            "inverse target transform; excludes HTTP and JSON handling"
        ),
    }


def _predict_one(naca_params, aoa_deg: float, reynolds: float) -> tuple[dict, np.ndarray]:
    bundle = app.state.bundle
    t0 = time.perf_counter()
    x = features.build_features(naca_params, aoa_deg, reynolds)
    out = bundle.predict(x)
    elapsed = (time.perf_counter() - t0) * 1000.0
    return {
        "cl": float(out.cl[0]), "cd": float(out.cd[0]),
        "cl_std": float(out.cl_std[0]), "cd_std": float(out.cd_std[0]),
        "inference_ms": elapsed,
    }, x


def _find_reference(x: np.ndarray) -> Reference | None:
    """Nearest stored test simulation, if the query is effectively on top of it."""
    ref_x = app.state.reference_x
    if len(ref_x) == 0:
        return None
    d = np.max(np.abs(ref_x - x) / app.state.bundle.x_std, axis=1)
    i = int(np.argmin(d))
    if d[i] > MATCH_TAU:
        return None
    r = app.state.reference[i]
    return Reference(name=r["name"], cl=r["cl"], cd=r["cd"],
                     distance=float(d[i]), exact=bool(d[i] < 1e-6))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result, x = _predict_one(req.naca, req.aoa_deg, req.reynolds)
    except naca.NacaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Out-of-range input is answered, not refused. Showing where the model stops
    # being trustworthy is the point of the demo; a 422 would hide it.
    return PredictResponse(
        **result,
        extrapolation=Extrapolation(**app.state.bundle.extrapolation_report(x)),
        reference=_find_reference(x),
    )


@app.post("/sweep")
def sweep(req: SweepRequest) -> dict:
    """A whole angle-of-attack sweep in one call.

    The page redraws its lift curve on every slider move; firing 61 separate
    POSTs for that would be slow and pointless.
    """
    bundle = app.state.bundle
    angles = np.linspace(req.aoa_min, req.aoa_max, req.n)
    t0 = time.perf_counter()
    try:
        x = np.vstack([
            features.build_features(req.naca, float(a), req.reynolds) for a in angles
        ])
    except naca.NacaError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out = bundle.predict(x)
    elapsed = (time.perf_counter() - t0) * 1000.0

    in_range = [bundle.extrapolation_report(row)["in_range"] for row in x]
    return {
        "aoa_deg": angles.tolist(),
        "cl": out.cl.tolist(), "cd": out.cd.tolist(),
        "cl_std": out.cl_std.tolist(), "cd_std": out.cd_std.tolist(),
        "in_range": in_range,
        "inference_ms": elapsed,
        "ms_per_point": elapsed / len(angles),
    }


@app.get("/geometry")
def geometry(naca_params: str, n: int = 160) -> dict:
    """Airfoil outline for the given digits, e.g. /geometry?naca_params=2,4,12.

    Drawing the section is not decoration: a wrong series branch or a misparsed
    parameter is instantly visible as a deformed shape.
    """
    try:
        params = [float(v) for v in naca_params.split(",")]
        pts = naca.naca_section(params, nb_samples=max(20, min(n, 1000)))
    except (ValueError, naca.NacaError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"x": pts[:, 0].tolist(), "y": pts[:, 1].tolist()}


@app.get("/test-cases")
def test_cases() -> list[dict]:
    """The 200 held-out simulations, for the demo's ground-truth comparison."""
    return app.state.reference


# Mounted LAST: a mount at "/" is a catch-all and would shadow every route
# declared after it.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
