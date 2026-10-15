# AeroSurrogate — Project Spec

Oct 13, 2026 · @Surendra Kumar C

## Overview

AeroSurrogate is a neural surrogate that predicts an airfoil's lift (Cl) and drag (Cd) coefficients in milliseconds instead of running a CFD simulation. It is built on the public AirfRANS dataset in a single \~4-hour session with Claude Code.

**Problem.** A RANS CFD run for one airfoil design takes minutes to hours. Design exploration needs hundreds of runs, so engineers wait on simulation instead of iterating.

**Goal.** Train a model that maps airfoil shape, angle of attack and Reynolds number to Cl and Cd, beats a linear baseline on held-out simulations, and is wrapped in a demo an engineer can use without touching code.

**Who it is for.** A design engineer evaluating many candidate airfoils, as a forward-deployed engineer would demo it to a customer.

**Done means.** A public GitHub repo with reproducible training, an evaluation report with real measured errors, and a working FastAPI + web demo.

## Scope

The build predicts two scalars per simulation (Cl, Cd) from global parameters; full flow-field prediction is deliberately out of scope.

| In scope (v1, \~4 hours) | Out of scope (future work) |
| --- | --- |
| Download and load AirfRANS, `full` task split | Raw OpenFOAM files, re-running CFD |
| Compute Cl and Cd per simulation from surface fields | Predicting pressure or velocity fields on the mesh |
| Linear baseline + PyTorch MLP | Graph neural networks, PointNet, transformers |
| Evaluation on the 200 held-out test simulations | Extrapolation tasks (Reynolds, angle of attack) beyond one optional check |
| FastAPI `/predict` endpoint + single-page slider demo | Auth, deployment, CAD import |
| README with real results and limitations | Hyperparameter sweeps beyond a few manual runs |

## Dataset

AirfRANS has 1,000 incompressible RANS simulations of 2D NACA airfoils, split 800 train / 200 test in the `full` task ([AirfRANS docs](https://airfrans.readthedocs.io/en/latest/notes/dataset.html)).

| Property | Value |
| --- | --- |
| Simulations | 1,000 (OpenFOAM, steady incompressible RANS) |
| Airfoils | NACA 4-digit and 5-digit series |
| Reynolds number | 2 to 6 million |
| Angle of attack | -5° to 15° |
| `full` task | 800 train / 200 test, same distribution |
| `scarce` task | 200 train / same 200 test |
| Extrapolation tasks | Reynolds (train 3–5M), angle of attack (train -2.5° to 12.5°) |
| Python API | `airfrans.dataset.download(root, ...)`, `airfrans.dataset.load(root, task, train)` |

**Inputs (per simulation):** NACA shape parameters, angle of attack, Reynolds number (or inlet velocity). **Targets:** Cl and Cd, integrated from surface pressure and wall shear.

**To verify in hour 1:** the download size (start it first), how simulation names encode the inlet velocity, angle and NACA parameters, and which helper on `airfrans.Simulation` returns the force coefficients.

## Functional requirements

Twelve requirements cover data, models, evaluation and the demo; FR-1 to FR-9 are must-haves, FR-10 to FR-12 are stretch.

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-1 | Script downloads AirfRANS to `data/` and skips if already present | Must |
| FR-2 | Build a tabular dataset: one row per simulation with shape params, angle of attack, Reynolds number, Cl, Cd; save as `data/coeffs.parquet` | Must |
| FR-3 | Use the official `full` split (800 train / 200 test); carve 10% of train as validation | Must |
| FR-4 | Standardize inputs and targets using train statistics only | Must |
| FR-5 | Fit a linear regression baseline for Cl and Cd | Must |
| FR-6 | Train a PyTorch MLP with early stopping on validation loss; fix random seeds | Must |
| FR-7 | Report MAE and mean relative error for Cl and Cd on the 200 test simulations, for both models, to `results/metrics.json` | Must |
| FR-8 | Save a predicted-vs-true scatter plot and a Cl-vs-angle sweep for one airfoil | Must |
| FR-9 | FastAPI `POST /predict` returns Cl, Cd and inference time for given inputs | Must |
| FR-10 | Single-page demo: angle-of-attack slider, live Cl/Cd, and the true CFD value when the input matches a test simulation | Stretch |
| FR-11 | Check the model against 10 test simulations side by side in the README | Stretch |
| FR-12 | Optional run on the angle-of-attack extrapolation task to show where the model fails | Stretch |

## Technical design

A one-way pipeline: data and training run once and save `model.pt`; the API and demo only load it, so predictions take milliseconds.

&#91;embedded content: AeroSurrogate pipeline · offline training, serving, evaluation\]

The saved model plus input scalers are the hand-off point between training and everything a user touches.

**Repo layout**

```
aerosurrogate/
  data.py          # download, Cl/Cd extraction, coeffs.parquet
  train.py         # linear baseline + MLP, saves model.pt + scalers
  evaluate.py      # test metrics, plots -> results/
  api.py           # FastAPI app
  web/index.html   # slider demo
  results/         # metrics.json, scatter.png, lift_curve.png
  README.md
  requirements.txt # airfrans, torch, scikit-learn, pandas, fastapi, uvicorn
```

**Model spec**

| Item | Choice |
| --- | --- |
| Inputs | Shape parameters, angle of attack (deg), Reynolds number; standardized |
| Outputs | Cl and Cd, standardized (one network, two outputs) |
| Architecture | MLP, 3 hidden layers × 128 units, SiLU activations |
| Loss / optimizer | MSE, Adam, learning rate 1e-3 |
| Training | Up to 2,000 epochs, early stopping after 100 epochs without validation improvement |
| Baseline | scikit-learn `LinearRegression` on the same inputs |

**API contract**

```json
POST /predict
{"naca": [2.0, 4.0, 12.0], "aoa_deg": 4.0, "reynolds": 3.0e6}

200 OK
{"cl": 0.71, "cd": 0.0112, "inference_ms": 0.4}
```

The response values above are placeholders showing the format, not results.

## Evaluation

The project passes if the MLP beats the linear baseline on the 200 held-out test simulations for both Cl and Cd; the exact numbers are whatever the run produces, recorded as measured.

**Metrics** (computed on the test set, never on train or validation):

- MAE for Cl and Cd, in coefficient units
- Mean relative error, in %. Cl crosses zero near zero lift, so relative error excludes or flags test cases with |Cl| < 0.05
- R² for Cl and Cd
- Inference time per prediction on CPU, in ms, versus the CFD runtime stated in the AirfRANS paper

**Acceptance criteria:**

- [ ] MLP MAE lower than the linear baseline for both Cl and Cd
- [ ] Metrics written to `results/metrics.json` by one command, reproducible with a fixed seed
- [ ] Cl-vs-angle sweep shows a physically sensible curve (roughly linear lift slope at low angles)
- [ ] README states the measured numbers and at least two limitations

Result table to fill in after the run:

| Model | Cl MAE | Cl rel. error (%) | Cd MAE | Cd rel. error (%) |
| --- | --- | --- | --- | --- |
| Linear baseline |  |  |  |  |
| MLP |  |  |  |  |

## Build plan

Four one-hour blocks, each ending in a checkpoint you can commit; start the dataset download before anything else.

1. **Hour 1: data (FR-1 to FR-4).** Start the download. While it runs, have Claude Code write `data.py`. Inspect one simulation by hand: its name, its fields, how the force coefficients come out. Build `coeffs.parquet`.
   - Checkpoint: 1,000 rows, no NaNs, Cl roughly -1 to 2, Cd small and positive.
2. **Hour 2: models (FR-5 to FR-8).** Linear baseline, then the MLP. Run `evaluate.py`, save plots and `metrics.json`.
   - Checkpoint: the MLP beats the baseline on test; the lift curve looks sensible.
3. **Hour 3: demo (FR-9, FR-10).** FastAPI `/predict` loading the saved model and scalers; one HTML page with an angle slider calling it.
   - Checkpoint: the slider moves Cl live in the browser.
4. **Hour 4: write-up and understanding.** README with measured results, plots and limitations; push to GitHub. Then explain every step out loud without notes: data, targets, split, model, metrics, demo.
   - Checkpoint: you can answer every question in the risks section below.

Claude Code prompt to start each block: paste this spec, name the FR IDs for that hour, and ask it to stop after the checkpoint so you can read the code before moving on.

## Risks, limitations and next steps

The biggest build risk is the data step; the biggest modelling limitation is that two scalars throw away the flow field.

| Risk | Fallback |
| --- | --- |
| Download is slow or too large for your disk | Start it first; if blocked, use the `scarce` task's 200 training simulations |
| Force-coefficient helper is unclear or missing | Integrate surface pressure and wall shear over the airfoil boundary yourself; check one case against a published Cl |
| Shape parameters hard to parse from names | Use a fixed set of surface-point coordinates as the shape input instead |
| MLP no better than linear | Log-transform Cd, add angle² as a feature, widen the net; report honestly if it still loses |

**Limitations to state in the README:** 2D steady RANS only; interpolation within the training ranges, not extrapolation; only NACA shapes; predicts integrated forces, not where on the airfoil they come from.

**Next step to describe:** a graph neural network or point-cloud model on the mesh that predicts the full pressure and velocity fields, from which Cl and Cd follow. That is the problem class Neural Concept's platform works on.

**Questions to be able to answer before Friday:**

- What is a RANS simulation, and why is it slow?
- How are Cl and Cd computed from the surface fields?
- Why did you choose an MLP on global parameters, and what does it miss?
- How did you split the data, and why does the split matter for surrogates?
- Where would this model fail, and how would a customer find out?
- How would you deploy this for an engineering team?
