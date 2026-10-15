# AeroSurrogate — What Was Built

A neural surrogate that predicts an airfoil's **lift (Cl)** and **drag (Cd)** coefficients from its shape, angle of attack and Reynolds number in **~0.4 ms**, instead of a RANS CFD simulation that costs ~25 minutes on 16 cores. Trained on the AirfRANS dataset (Bonnet et al., NeurIPS 2022, arXiv:2212.07564).

All numbers below come from `results/metrics.json` (reproducible bit-for-bit apart from wall-clock timings).

---

## 1. Headline results

Evaluated on the **200 held-out test simulations** of AirfRANS' `scarce` task.

| Model | Cl MAE | Cl rel. err | Cl R² | Cd MAE | Cd rel. err | Cd R² |
|---|---:|---:|---:|---:|---:|---:|
| Mean predictor | 0.4975 | 147.85% | −0.0001 | 0.003352 | 25.18% | −0.0106 |
| Linear baseline | 0.0205 | 5.18% | 0.9976 | 0.000793 | 5.85% | 0.9194 |
| **MLP (5-seed ensemble)** | **0.0090** | **2.30%** | **0.9995** | **0.000303** | **2.16%** | **0.9838** |

- Paired bootstrap (10,000 resamples): MLP wins 100% on both targets; both 95% CIs exclude zero (Cl [0.0088, 0.0141], Cd [0.00038, 0.00061]).
- Cl relative error excludes 10 of 200 cases with |Cl| < 0.05; the threshold-free normalised error is **1.21% (MLP) vs 2.72% (linear)**.
- Learned lift slope **0.1073/deg** vs thin-airfoil theory **0.1097/deg** (within 2.2%), with no physics in the loss.
- Latency: **0.39 ms** end to end (median, 1 CPU thread); ~2.7 ms including HTTP/JSON. Roughly 3.8 million× faster in wall clock than the 25-minute, 16-core solve (caveat: the solve produces the full flow field, the surrogate two numbers).

---

## 2. The data pipeline

### Streaming instead of downloading
`Dataset.zip` is 9.34 GiB (14 GiB unzipped), against a **1 GB disk limit**. No smaller mirror exists. So `remote_zip.py` exposes the remote archive as a seekable file over HTTP range requests; Python's `zipfile` then extracts one simulation at a time.

- Per simulation: fetch `_internal.vtu` + `_aerofoil.vtp`, compute Cl/Cd, delete immediately.
- Peak disk ~90 MB; 3.98 GB crosses the network; `data/` ends at ~0.3 MB.
- 6 worker processes, a single pool for the whole run, `OMP_NUM_THREADS=1` per worker.
- Resumable: 50-simulation parquet shards written atomically; failures go to `errors.jsonl`.
- Retries with exponential backoff on every request, including the opening HEAD (the host timed out once).
- Refuses to proceed if the server ignores `Range` (would silently download 9.34 GiB) or if the file size differs from the expected 10,029,067,577 bytes.

### Task choice
The 4 GB network budget buys 400 simulations → AirfRANS' own **`scarce` task**: 200 train + the standard 200 `full_test` set (processed first, so an interrupted run still has a full test set).

### How Cl and Cd are computed
Via `airfrans.Simulation.force_coefficient(compressible=False, reference=True)` — drag first, wind-axis rotated, unit chord. Pinned by a **golden case** from the AirfRANS docs: `airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252` → cd 0.013431 (published 0.0134), cl 0.809893 (published 0.8099). Per-row cross-checks confirm inlet velocity and angle of attack match the simulation name.

Viscous drag is ~62% of Cd (so drag genuinely needs the internal mesh); viscous lift is ~0.05% of Cl.

### Validation
`data.py check` runs 20 checks (uniqueness, no NaNs, split sizes/disjointness, physical ranges, Cd = Cdp + Cdv, geometry-vs-mesh agreement to 4e-4 chord, golden case, etc.) → `results/data_checks.txt`. All pass.

---

## 3. Features and targets

**Nine features** (`features.py`, the single definition used by training, evaluation and serving):

`max_camber`, `x_max_camber`, `max_thickness`, `alpha_L0_deg` (thin-airfoil zero-lift angle), `te_camber_slope`, `is_5digit`, `aoa_deg`, `aoa_eff_sq` = (aoa − α_L0)², `log10_reynolds`.

Raw NACA digits can't be used directly: 4-digit has 3 parameters, 5-digit has 4, and the numbers mean different things (200/200 split in the data). Physical descriptors mean the same thing for every row. `Re = U∞·c/ν` with ν = 1.56e-5 (the dataset's generation viscosity).

**Targets:** Cl raw; **log(Cd)** (positive, spans a decade, makes squared error ≈ relative error). The linear baseline gets the identical transform so the comparison isn't rigged.

**Train/serve parity:** `build_feature_matrix` goes through the same `build_features` the API calls, and a test asserts bit-identical vectors.

---

## 4. Model and training

- **Architecture:** MLP 9 → 128 → 128 → 128 → 2, SiLU, 34,562 parameters; 5-seed ensemble (spread shown as uncertainty in the demo).
- **Loss:** Huber (δ = 1). Adam, lr 1e-3, weight decay 1e-4, full batch.
- **Epochs:** 318, chosen by 5-fold cross-validation over all 200 training rows, scored on mean absolute error.
- Standardisation from training rows only; stored as plain tensors (no pickled sklearn scalers).
- Determinism: seeds for random/numpy/torch, `use_deterministic_algorithms(True)`, `set_num_threads(1)`.
- Artifact `artifacts/model.pt` (688 KB) contains only tensors → loads with `torch.load(weights_only=True)`. `model_meta.json` records features, architecture, CV info, data hash and the training-set input ranges that drive extrapolation warnings.
- Serving runs the forward pass in **numpy** (ensemble weights stacked, one matmul per layer): 1.73 ms → ~0.4 ms versus torch at batch size 1.

---

## 5. Evaluation (`evaluate.py`)

Generates everything in `results/`:

- `metrics.json` — all metrics, bootstrap CIs, latency breakdown, speedup, lift-slope check, extrapolation study
- `results_table.md`, `examples.md` (10 worked examples at angle-of-attack decile midpoints — a stated, non-cherry-picked rule)
- `scatter.png` (predicted vs true, Cd on log-log), `lift_curve.png` (all 200 test cases + a model sweep with ensemble band and thin-airfoil line), `residuals.png`
- `web/test_reference.json` — ground truth for the 200 test simulations (lets the demo run without the dataset)

**Extrapolation study (FR-12):** retrained only on angles within [−2.5°, 12.5°], tested on the 41 outside simulations. Cl MAE degraded ×2.97, Cd ×2.51. A prediction registered before running ("Cl mild, Cd bad") was **wrong** — Cl degraded more.

---

## 6. API and demo

### FastAPI (`api.py`)
| Route | Purpose |
|---|---|
| `POST /predict` | `{"naca":[...], "aoa_deg", "reynolds"}` → `cl, cd, inference_ms, cl_std, cd_std, extrapolation, reference` |
| `POST /sweep` | whole angle sweep in one call |
| `GET /geometry` | airfoil outline |
| `GET /meta`, `/health`, `/test-cases` | model info, status, the 200 ground-truth cases |

- 422 only for physically impossible input (wrong digit count, non-finite, thickness ∉ (0.5, 40], Re ≤ 0, 5-digit Q ∉ {0,1}).
- **Out-of-range input is answered with a warning, not refused**, so extrapolation behaviour stays visible.
- A query within 5% of a training standard deviation (L∞) of a stored test simulation returns its CFD ground truth.
- Serving imports no pandas, sklearn, pyvista or VTK; needs only `requirements-serve.txt` (numpy, torch, fastapi, uvicorn, pydantic).

### Demo page (`web/index.html`)
Single file, vanilla JS + inline SVG (no build step, no CDN, **no React**). Controls for series/shape/AoA/Re, live Cl/Cd/L-D/inference-time readout, in-envelope vs extrapolating badge, airfoil section drawing, lift curve with ensemble band and shaded out-of-range regions. A dropdown of the 200 held-out cases snaps controls to a real simulation and shows true vs predicted values; dragging the angle away detaches the ground-truth marker. In-flight requests are cancelled with `AbortController`.

---

## 7. Findings that changed the result

1. **24% of four-digit sections have a contradictory camber line.** Upstream's `P = 0` branch returns `y_c ≡ 0` with a non-zero slope. Inverting CFD lift for those 48 airfoils gave an implied zero-lift angle of +0.24° ± 0.46 (uncambered), versus an analytic mean of −3.1°. Differentiating `y_c` numerically fixed it and raised correlation with the implied zero-lift angle from 0.78 to 0.94.
2. **The 5-digit reflex digit carries no information.** `k ≡ 0` algebraically; Q=0 and Q=1 give identical camber lines, and the real meshes match this. `is_reflex` is stored but not used as a feature.
3. **One row dominated the loss.** A 5.2%-thick section at −4.4° (Cd 0.0459, +4σ) was ~99% of squared-error training loss. Huber loss cut cross-validated error 16% and test Cl MAE from 0.0109 to 0.0090. Capacity barely mattered once the loss was robust.
4. **A 20-row validation carve can't support early stopping.** Three of five seeds "early stopped" badly undertrained. Replaced with 5-fold CV on all 200 rows.

---

## 8. Tests (55, ~13 s, no dataset needed except where noted)

- `test_naca.py` — name parsing, Reynolds, textbook geometry (NACA 2412 α_L0 ≈ −2.08°), degenerate P=0, reflex degeneracy, vendored-vs-upstream equality
- `test_features.py` — feature values, caching safety, target round-trip, **train/serve parity** (including against the real parquet)
- `test_remote_zip.py` — range-request file object against a local stub server, size mismatch, no-Range refusal
- `test_api.py` — contract, validation, extrapolation warnings, sweep/predict agreement, reference lookup, artifact loads without pickle, numpy-vs-torch inference, **acceptance criterion** (MLP < linear on both targets, significant), lift slope in a physical range

---

## 9. Files

```
config.py          paths, constants, seeds, golden case
naca.py            name parsing + vendored NACA geometry (pure numpy)
features.py        the single feature/target definition
remote_zip.py      seekable HTTP range-request file object
data.py            peek | build | check | refresh | synthetic
model.py           MLP definition + numpy ensemble inference
train.py           CV epoch selection, baseline, ensemble, artifact
evaluate.py        metrics, bootstrap, figures, extrapolation, reference table
api.py             FastAPI app (also serves web/)
web/index.html     slider demo
artifacts/         model.pt, model_meta.json
results/           metrics.json, tables, figures, data_checks.txt
tests/             55 tests
requirements.txt, requirements-serve.txt, README.md, .gitignore
```

Total project footprint (excluding the venv): ~1.6 MB.

---

## 10. How to run

**Demo only (no dataset):**
```bash
pip install -r requirements-serve.txt
uvicorn api:app          # http://127.0.0.1:8000
```

**Full reproduction:**
```bash
pip install -r requirements.txt
python data.py build      # ~25 min, 3.98 GB streamed, resumable
python data.py check
python train.py --loss huber
python evaluate.py
pytest -q
```

Note: `train.py` defaults to `--loss mse`; the reported results use `--loss huber`.

---

## 11. Limitations

- 2D steady RANS only; the surrogate inherits the simulations' assumptions.
- Interpolation, not extrapolation (quantified above). Training envelope: AoA −4.8° to 14.8°, Re 2.0–6.0 M, thickness 5–20% chord.
- NACA 4/5-digit only; the reflex digit is degenerate, so the shape space is effectively smaller.
- Two scalars discard the flow field (no load distribution, separation or stall information).
- Thin sections at high incidence are the weak spot (worst Cd case: 5.7%-thick at 12.1°).
- 200 test rows give roughly ±7% on any MAE; hence the bootstrap intervals.
- Trained on the `scarce` task (200 training simulations), a consequence of the disk budget.
- **Not comparable to the AirfRANS paper's numbers**: this is a 9-parameter → (Cl, Cd) regression, not full-field prediction.

---

## 12. Not done / deviations from the spec

- No git repository was initialised (per the earlier decision).
- Deviations from the spec, each evidence-backed: Huber instead of MSE, 5-fold CV instead of a 10% validation carve with patience-100 early stopping, numpy instead of torch for inference.
- The optional stretch of fetching lift-only labels for the other 600 simulations was not done.
- The 2×64 architecture comparison from the plan was replaced by a broader capacity/loss CV sweep; `[32,32]` Huber was marginally best (0.0530 vs 0.0538 for 3×128), a 1.6% gap within noise, so the spec's 3×128 was kept.
