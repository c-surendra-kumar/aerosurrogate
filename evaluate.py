"""Measure the trained model on the 200 held-out test simulations.

Writes `results/metrics.json`, three figures, the FR-11 worked-example table,
and `web/test_reference.json` (the ground-truth lookup the demo uses, which is
also what lets the demo run on a clone with no dataset present).

    python evaluate.py
    python evaluate.py --data data/coeffs_synth.parquet --artifacts artifacts_synth

A note the README repeats: this is a 9-parameter -> (Cl, Cd) regression against
coefficients that AirfRANS integrated from its own ground-truth fields. It is
NOT the AirfRANS benchmark task, which is predicting the pressure and velocity
fields at every mesh node. The paper's baseline numbers -- and its well-known
Cd collapse -- describe mesh-field surrogates, and are not comparable to these
in either direction.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import features
import naca
from config import (ARTIFACT_DIR, CFD_CORES, CFD_MINUTES_PER_SIM, CFD_SOURCE,
                    CL_REL_ERR_FLOOR, COEFFS_PARQUET, METRICS_JSON, RESULTS_DIR,
                    TEST_REFERENCE_JSON, WEB_DIR)
from model import load_bundle

#: Thin-airfoil-theory lift slope, 2*pi per radian expressed per degree.
THIN_AIRFOIL_SLOPE = 2 * np.pi * np.pi / 180.0   # 0.10966 /deg
AOA_BAND = (-2.5, 12.5)                          # AirfRANS' own `aoa` task band
N_BOOTSTRAP = 10_000

C_TRUE, C_MLP, C_LIN = "#444444", "#1f77b4", "#d62728"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def target_metrics(
    true: np.ndarray, pred: np.ndarray, names: np.ndarray,
    rel_floor: float | None = None,
) -> dict:
    """MAE/RMSE/R^2/relative error, always in physical units."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    err = pred - true
    abs_err = np.abs(err)

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    if rel_floor is None:
        mask = np.ones_like(true, dtype=bool)
    else:
        # Relative error is meaningless where the quantity crosses zero, which
        # Cl does. Exclude those rows and say how many were excluded.
        mask = np.abs(true) >= rel_floor

    worst = int(np.argmax(abs_err))
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mean_rel_err_pct": float(
            np.mean(abs_err[mask] / np.abs(true[mask])) * 100
        ) if mask.any() else float("nan"),
        "n_excluded_from_rel_err": int((~mask).sum()),
        # Defined for every row and independent of any threshold, so this is
        # the number worth quoting when the threshold would be argued about.
        "normalised_err_pct": float(abs_err.mean() / np.abs(true).mean() * 100),
        "max_abs_err": float(abs_err[worst]),
        "worst_case": str(names[worst]),
        "bootstrap_mae_ci95": _bootstrap_mae_ci(abs_err),
    }


def _bootstrap_mae_ci(abs_err: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(abs_err), size=(N_BOOTSTRAP, len(abs_err)))
    maes = abs_err[idx].mean(axis=1)
    return [float(np.percentile(maes, 2.5)), float(np.percentile(maes, 97.5))]


def paired_bootstrap(
    true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, seed: int = 0
) -> dict:
    """Is model B's MAE advantage over A larger than the test-set noise?

    With 200 test rows the standard error on MAE is roughly 7%, so a bare
    "lower MAE" is close to a coin flip. Resampling the same rows for both
    models and looking at the *difference* answers the acceptance criterion
    properly: the win is real if the 95% interval excludes zero.
    """
    err_a = np.abs(np.asarray(pred_a, float) - true)
    err_b = np.abs(np.asarray(pred_b, float) - true)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(true), size=(N_BOOTSTRAP, len(true)))
    delta = err_a[idx].mean(axis=1) - err_b[idx].mean(axis=1)
    return {
        "mae_delta": float(err_a.mean() - err_b.mean()),
        "ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "win_fraction": float((delta > 0).mean()),
        "significant": bool(np.percentile(delta, 2.5) > 0),
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def measure_latency(bundle, sample_naca, aoa, re, n_warmup=100, n=1000) -> dict:
    def stats(fn) -> dict:
        for _ in range(n_warmup):
            fn()
        times = np.empty(n)
        for i in range(n):
            t0 = time.perf_counter_ns()
            fn()
            times[i] = (time.perf_counter_ns() - t0) / 1e6
        return {
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "mean_ms": float(times.mean()),
        }

    x = features.build_features(sample_naca, aoa, re)
    return {
        "torch_threads": torch.get_num_threads(),
        "n_warmup": n_warmup, "n_measured": n, "batch_size": 1,
        "ensemble_members": len(bundle.members),
        # Separated because the geometry rebuild can plausibly cost more than
        # the forward pass, and conflating them would hide that.
        "feature_build": stats(lambda: features.build_features(sample_naca, aoa, re)),
        "forward_only": stats(lambda: bundle.predict(x)),
        "end_to_end": stats(
            lambda: bundle.predict(features.build_features(sample_naca, aoa, re))
        ),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_scatter(path: Path, true_cl, true_cd, mlp_cl, mlp_cd, lin_cl, lin_cd, m) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.scatter(true_cl, lin_cl, s=18, facecolors="none", edgecolors=C_LIN,
               linewidths=0.8, label=f"linear  R2={m['linear']['cl']['r2']:.4f}")
    ax.scatter(true_cl, mlp_cl, s=14, color=C_MLP, alpha=0.8,
               label=f"MLP  R2={m['mlp']['cl']['r2']:.4f}")
    lims = [min(true_cl.min(), mlp_cl.min()) - 0.1, max(true_cl.max(), mlp_cl.max()) + 0.1]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(xlim=lims, ylim=lims, xlabel="true $C_l$ (CFD)", ylabel="predicted $C_l$",
           title="Lift coefficient")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.scatter(true_cd, lin_cd, s=18, facecolors="none", edgecolors=C_LIN,
               linewidths=0.8, label=f"linear  R2={m['linear']['cd']['r2']:.4f}")
    ax.scatter(true_cd, mlp_cd, s=14, color=C_MLP, alpha=0.8,
               label=f"MLP  R2={m['mlp']['cd']['r2']:.4f}")
    lims = [min(true_cd.min(), mlp_cd.min()) * 0.9, max(true_cd.max(), mlp_cd.max()) * 1.1]
    ax.plot(lims, lims, "k--", lw=1)
    # Log-log: Cd spans a decade and was modelled as log(Cd); on linear axes the
    # low-drag cases collapse into an unreadable blob.
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
           xlabel="true $C_d$ (CFD)", ylabel="predicted $C_d$",
           title="Drag coefficient (log-log)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.25, which="both")

    fig.suptitle("AeroSurrogate vs CFD on 200 held-out AirfRANS simulations")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lift_curve(path: Path, test: pd.DataFrame, mlp_cl: np.ndarray, bundle) -> dict:
    """Two panels, because a conventional lift curve is not available here.

    AirfRANS samples shape, angle of attack and Reynolds number jointly and
    continuously, so no geometry appears twice -- there is no sweep of one
    airfoil through angle in the data. The left panel is therefore the honest
    acceptance check (every test simulation against its own angle); the right
    panel is a model sweep with clearly-labelled nearest-neighbour CFD points.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(test["aoa_deg"], test["cl"], s=20, color=C_TRUE, alpha=0.6, label="CFD (true)")
    ax.scatter(test["aoa_deg"], mlp_cl, s=14, color=C_MLP, alpha=0.8, marker="x",
               label="AeroSurrogate")
    span = np.linspace(test["aoa_deg"].min(), test["aoa_deg"].max(), 50)
    ax.plot(span, THIN_AIRFOIL_SLOPE * span, "k:", lw=1.4,
            label=f"thin-airfoil slope {THIN_AIRFOIL_SLOPE:.4f}/deg")
    ax.set(xlabel="angle of attack (deg)", ylabel="$C_l$",
           title="All 200 test simulations\n(scatter in camber and Re is physical)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    # Right: a continuous sweep for the test airfoil closest to a canonical
    # NACA 2412, chosen deterministically.
    ref = naca.geometry_descriptors([2.0, 4.0, 12.0])
    dist = (
        ((test["max_camber"] - ref["max_camber"]) / 0.01) ** 2
        + ((test["x_max_camber"] - ref["x_max_camber"]) / 0.1) ** 2
        + ((test["max_thickness"] - ref["max_thickness"]) / 0.02) ** 2
    )
    row = test.loc[dist.idxmin()]
    params = naca_params_of(row)
    re = float(row["reynolds"])

    sweep = np.linspace(-5, 15, 201)
    x = np.array([features.build_features(params, a, re) for a in sweep])
    pred = bundle.predict(x)

    ax = axes[1]
    ax.plot(sweep, pred.cl, color=C_MLP, lw=2, label="AeroSurrogate")
    ax.fill_between(sweep, pred.cl - 2 * pred.cl_std, pred.cl + 2 * pred.cl_std,
                    color=C_MLP, alpha=0.2, label="ensemble spread (2$\\sigma$)")
    a0 = naca.geometry_descriptors(params)["alpha_L0_deg"]
    ax.plot(sweep, THIN_AIRFOIL_SLOPE * (sweep - a0), "k:", lw=1.4,
            label="thin-airfoil theory")
    ax.scatter([row["aoa_deg"]], [row["cl"]], s=110, color=C_LIN, marker="*",
               zorder=5, label="this simulation (CFD)")

    near = test[
        (np.abs(test["max_camber"] - row["max_camber"]) < 0.005)
        & (np.abs(test["max_thickness"] - row["max_thickness"]) < 0.015)
        & (test["name"] != row["name"])
    ]
    if len(near):
        ax.scatter(near["aoa_deg"], near["cl"], s=28, facecolors="none",
                   edgecolors=C_TRUE,
                   label=f"nearest-neighbour CFD, n={len(near)}\n(different geometry/Re)")

    label = ("NACA " + "/".join(f"{v:g}" for v in params))
    ax.set(xlabel="angle of attack (deg)", ylabel="$C_l$",
           title=f"Model sweep: {label}\nRe = {re/1e6:.2f}e6")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Measured lift slope of the sweep over the near-linear region.
    lin = (sweep > -2) & (sweep < 8)
    slope = float(np.polyfit(sweep[lin], pred.cl[lin], 1)[0])
    return {
        "airfoil": str(row["name"]), "naca": params, "reynolds": re,
        "modelled_slope_per_deg": slope,
        "thin_airfoil_slope_per_deg": float(THIN_AIRFOIL_SLOPE),
        "slope_ratio": slope / float(THIN_AIRFOIL_SLOPE),
    }


def plot_residuals(path: Path, test: pd.DataFrame, mlp_cd: np.ndarray) -> None:
    rel = np.abs(mlp_cd - test["cd"]) / test["cd"] * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, xlabel in (
        (axes[0], "aoa_deg", "angle of attack (deg)"),
        (axes[1], "reynolds", "Reynolds number"),
    ):
        s = ax.scatter(test[key], rel, c=test["max_thickness"] * 100, s=22,
                       cmap="viridis")
        ax.set(xlabel=xlabel, ylabel="$C_d$ relative error (%)")
        ax.grid(alpha=0.25)
        fig.colorbar(s, ax=ax, label="thickness (% chord)")
    fig.suptitle("Where the drag model is weakest")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


naca_params_of = naca.params_from_row


def describe_naca(row) -> str:
    p = naca_params_of(row)
    if len(p) == 4:
        return f"5-digit L{p[0]:.2f} P{p[1]:.2f} Q{p[2]:.0f} t{p[3]:.2f}%"
    return f"4-digit M{p[0]:.2f} P{p[1]:.2f} t{p[2]:.2f}%"


def write_examples(path: Path, test: pd.DataFrame, cl: np.ndarray, cd: np.ndarray) -> None:
    """FR-11: ten worked examples, chosen by a rule the README states.

    Sorted by angle of attack and sampled at the decile midpoints, so the table
    spans the physical range and is visibly not cherry-picked.
    """
    order = test.sort_values("aoa_deg").index
    pos = [int(round((k + 0.5) / 10 * (len(order) - 1))) for k in range(10)]
    pick = [order[p] for p in pos]
    loc = {idx: i for i, idx in enumerate(test.index)}

    lines = [
        "| Simulation | Shape | AoA (deg) | Re (x1e6) | Cl true | Cl pred | dCl | "
        "Cd true | Cd pred | Cd err (%) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx in pick:
        r = test.loc[idx]
        i = loc[idx]
        lines.append(
            f"| `{r['name']}` | {describe_naca(r)} | {r['aoa_deg']:.2f} | "
            f"{r['reynolds']/1e6:.2f} | {r['cl']:.4f} | {cl[i]:.4f} | "
            f"{cl[i]-r['cl']:+.4f} | {r['cd']:.5f} | {cd[i]:.5f} | "
            f"{(cd[i]-r['cd'])/r['cd']*100:+.2f} |"
        )
    lines += [
        "",
        "_Selection rule: the 200 test simulations sorted by angle of attack, "
        "sampled at each decile midpoint._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_results_table(path: Path, metrics: dict, comparison: dict) -> None:
    """The spec's results table, generated so the README can never drift."""
    rows = [
        "| Model | Cl MAE | Cl rel. error (%) | Cl R2 | Cd MAE | Cd rel. error (%) | Cd R2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"mean": "Mean predictor", "linear": "Linear baseline",
              "mlp": "**MLP (this work)**"}
    for key in ("mean", "linear", "mlp"):
        m = metrics[key]
        rows.append(
            f"| {labels[key]} | {m['cl']['mae']:.4f} | {m['cl']['mean_rel_err_pct']:.2f} "
            f"| {m['cl']['r2']:.4f} | {m['cd']['mae']:.6f} | "
            f"{m['cd']['mean_rel_err_pct']:.2f} | {m['cd']['r2']:.4f} |"
        )
    rows += ["", "Paired bootstrap over the 200 test simulations (10,000 resamples):", ""]
    for key, label in (("cl", "Cl"), ("cd", "Cd")):
        c = comparison[key]
        rows.append(
            f"- **{label}**: the MLP's MAE is lower by {c['mae_delta']:.6f}, "
            f"95% CI [{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}], winning "
            f"{c['win_fraction'] * 100:.1f}% of resamples "
            f"({'significant' if c['significant'] else 'NOT significant'})."
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_test_reference(path: Path, test: pd.DataFrame) -> None:
    """Ground truth for the demo's FR-10 overlay.

    Committed alongside the model so the API and the web page work on a fresh
    clone with no dataset and no streaming.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "name": r["name"], "naca": naca_params_of(r),
            "label": describe_naca(r),
            "aoa_deg": round(float(r["aoa_deg"]), 4),
            "reynolds": round(float(r["reynolds"]), 1),
            "cl": round(float(r["cl"]), 6), "cd": round(float(r["cd"]), 7),
        }
        for _, r in test.sort_values("name").iterrows()
    ]
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def extrapolation_study(
    df: pd.DataFrame, hidden: list[int], seeds: list[int], n_epochs: int
) -> dict:
    """FR-12, run within the data the network budget bought.

    AirfRANS' own `aoa` task needs 804 training simulations we did not fetch, so
    the same question is asked of the scarce set: train only on angles inside
    [-2.5, 12.5] degrees and measure what happens in the tails. Prediction made
    before running: Cl degrades mildly because it is near-linear in angle and an
    MLP extrapolates roughly affinely, while Cd degrades badly because it is
    quadratic and rising steeply.
    """
    import train as train_mod

    lo, hi = AOA_BAND
    pool = df[df["in_scarce_train"]]
    in_band = pool[pool["aoa_deg"].between(lo, hi)]
    test = df[df["in_full_test"]]
    test_in = test[test["aoa_deg"].between(lo, hi)]
    test_out = test[~test["aoa_deg"].between(lo, hi)]
    if len(test_out) < 5 or len(in_band) < 30:
        return {"skipped": "not enough rows for a meaningful extrapolation split"}

    x_tr, y_tr = train_mod.xy(in_band)
    x_mean, x_std = x_tr.mean(0), x_tr.std(0)
    x_std[x_std < 1e-12] = 1.0
    y_mean, y_std = y_tr.mean(0), y_tr.std(0)
    y_std[y_std < 1e-12] = 1.0

    # Same architecture, same seeds and the same epoch budget as the headline
    # model, so the only thing that differs is which angles were in training.
    nets = [
        train_mod._train_loop((x_tr - x_mean) / x_std, (y_tr - y_mean) / y_std,
                              hidden, s, n_epochs)[0]
        for s in seeds
    ]

    def predict(frame: pd.DataFrame):
        x = (features.build_feature_matrix(frame) - x_mean) / x_std
        with torch.inference_mode():
            out = torch.stack(
                [n(torch.tensor(x, dtype=torch.float32)) for n in nets]
            ).numpy().astype(np.float64)
        return features.inverse_targets(out.mean(axis=0) * y_std + y_mean)

    result = {
        "band_deg": list(AOA_BAND),
        "n_train_in_band": int(len(in_band)),
        "n_test_inside": int(len(test_in)),
        "n_test_outside": int(len(test_out)),
        "prediction_registered_before_running":
            "Cl degrades mildly, Cd degrades badly",
    }
    for label, frame in (("interpolation", test_in), ("extrapolation", test_out)):
        cl, cd = predict(frame)
        result[label] = {
            "cl": target_metrics(frame["cl"].to_numpy(), cl, frame["name"].to_numpy(),
                                 CL_REL_ERR_FLOOR),
            "cd": target_metrics(frame["cd"].to_numpy(), cd, frame["name"].to_numpy()),
        }
    for key in ("cl", "cd"):
        a = result["interpolation"][key]["mae"]
        b = result["extrapolation"][key]["mae"]
        result.setdefault("degradation_ratio", {})[key] = float(b / a) if a else None
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, default=COEFFS_PARQUET)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--skip-extrapolation", action="store_true")
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    bundle = load_bundle(args.artifacts / "model.pt", args.artifacts / "model_meta.json")

    test = df[df["in_full_test"]].sort_values("name").reset_index(drop=True)
    train = df[df["in_scarce_train"]]
    print(f"evaluating on {len(test)} held-out simulations")

    x = features.build_feature_matrix(test)
    true_cl = test["cl"].to_numpy()
    true_cd = test["cd"].to_numpy()
    names = test["name"].to_numpy()

    pred = bundle.predict(x)
    lin_cl, lin_cd = bundle.predict_linear(x)
    # Trivial reference so R^2 has something to mean. Cd uses the geometric
    # mean, matching the log target.
    mean_cl = np.full_like(true_cl, train["cl"].mean())
    mean_cd = np.full_like(true_cd, float(np.exp(np.log(train["cd"]).mean())))

    models = {
        "linear": {"cl": lin_cl, "cd": lin_cd},
        "mlp": {"cl": pred.cl, "cd": pred.cd},
        "mean": {"cl": mean_cl, "cd": mean_cd},
    }
    metrics = {
        name: {
            "cl": target_metrics(true_cl, p["cl"], names, CL_REL_ERR_FLOOR),
            "cd": target_metrics(true_cd, p["cd"], names),
        }
        for name, p in models.items()
    }

    comparison = {
        "cl": paired_bootstrap(true_cl, lin_cl, pred.cl),
        "cd": paired_bootstrap(true_cd, lin_cd, pred.cd),
    }
    acceptance = {
        "mlp_beats_linear_cl": bool(metrics["mlp"]["cl"]["mae"] < metrics["linear"]["cl"]["mae"]),
        "mlp_beats_linear_cd": bool(metrics["mlp"]["cd"]["mae"] < metrics["linear"]["cd"]["mae"]),
        "cl_win_significant": comparison["cl"]["significant"],
        "cd_win_significant": comparison["cd"]["significant"],
    }

    print("\n                 Cl MAE     Cl rel%    Cl R2      Cd MAE     Cd rel%    Cd R2")
    for name in ("mean", "linear", "mlp"):
        m = metrics[name]
        print(f"  {name:12s} {m['cl']['mae']:9.5f} {m['cl']['mean_rel_err_pct']:9.2f} "
              f"{m['cl']['r2']:9.5f} {m['cd']['mae']:9.6f} "
              f"{m['cd']['mean_rel_err_pct']:9.2f} {m['cd']['r2']:9.5f}")
    for key in ("cl", "cd"):
        c = comparison[key]
        print(f"  {key.upper()}: MLP beats linear by {c['mae_delta']:+.6f} MAE, "
              f"95% CI [{c['ci95'][0]:+.6f}, {c['ci95'][1]:+.6f}], "
              f"wins {c['win_fraction']*100:.1f}% of resamples "
              f"-> {'SIGNIFICANT' if c['significant'] else 'NOT significant'}")

    sample = test.iloc[len(test) // 2]
    latency = measure_latency(bundle, naca_params_of(sample),
                              float(sample["aoa_deg"]), float(sample["reynolds"]))
    cfd_sec = CFD_MINUTES_PER_SIM * 60
    e2e = latency["end_to_end"]["median_ms"] / 1000
    print(f"\n  latency: feature build {latency['feature_build']['median_ms']:.3f} ms, "
          f"forward {latency['forward_only']['median_ms']:.3f} ms, "
          f"end-to-end {latency['end_to_end']['median_ms']:.3f} ms (median, 1 thread)")

    print("\nwriting figures...")
    plot_scatter(results_dir / "scatter.png", true_cl, true_cd, pred.cl, pred.cd,
                 lin_cl, lin_cd, metrics)
    lift = plot_lift_curve(results_dir / "lift_curve.png", test, pred.cl, bundle)
    plot_residuals(results_dir / "residuals.png", test, pred.cd)
    print(f"  lift slope {lift['modelled_slope_per_deg']:.5f}/deg vs thin-airfoil "
          f"{lift['thin_airfoil_slope_per_deg']:.5f}/deg "
          f"(ratio {lift['slope_ratio']:.3f})")

    write_examples(results_dir / "examples.md", test, pred.cl, pred.cd)
    write_results_table(results_dir / "results_table.md", metrics, comparison)
    # Only a real run may touch the file the demo serves. A synthetic run keeps
    # its reference table beside its own results, so it cannot overwrite the
    # ground truth the web page shows.
    write_test_reference(
        TEST_REFERENCE_JSON if results_dir == RESULTS_DIR
        else results_dir / "test_reference.json",
        test,
    )

    extrapolation = {"skipped": "disabled"} if args.skip_extrapolation else \
        extrapolation_study(
            df, list(bundle.meta["arch"]["hidden"]),
            [h["seed"] for h in bundle.meta["train"]["per_seed"]],
            int(bundle.meta["train"]["cv"]["chosen_epochs"]),
        )
    if "extrapolation" in extrapolation:
        print(f"  extrapolation: Cl MAE x{extrapolation['degradation_ratio']['cl']:.2f}, "
              f"Cd MAE x{extrapolation['degradation_ratio']['cd']:.2f} outside "
              f"{AOA_BAND} deg")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "path": str(args.data),
            "n_test": int(len(test)), "n_train": int(len(train)),
            "task": "AirfRANS scarce (200 train / 200 test)",
            "synthetic": bool((df.get("label_kind") == "SYNTHETIC").any()),
        },
        "feature_version": features.FEATURE_VERSION,
        "feature_names": list(features.FEATURE_NAMES),
        "target_transform": features.TARGET_TRANSFORM,
        "model_meta": {k: bundle.meta[k] for k in ("arch", "train", "input_ranges")},
        "models": metrics,
        "comparison_linear_vs_mlp": comparison,
        "acceptance": acceptance,
        "lift_curve": lift,
        "latency": latency,
        "cfd_reference": {
            "minutes_per_simulation": CFD_MINUTES_PER_SIM,
            "cores": CFD_CORES,
            "core_hours": CFD_MINUTES_PER_SIM / 60 * CFD_CORES,
            "source": CFD_SOURCE,
        },
        "speedup": {
            "wall_clock_ratio": float(cfd_sec / e2e),
            "core_hour_ratio": float(cfd_sec * CFD_CORES / e2e),
            "caveat": (
                "Compares a 16-core RANS field solve against a single-threaded "
                "two-scalar inference. CFD produces the entire flow field; the "
                "surrogate produces two numbers."
            ),
        },
        "extrapolation_aoa": extrapolation,
        "not_comparable_to_paper": (
            "This is a 9-parameter -> (Cl, Cd) regression against AirfRANS' own "
            "integrated coefficients, not the AirfRANS benchmark task of predicting "
            "p and u at every mesh node. The paper's baselines are not comparable."
        ),
    }
    out = results_dir / METRICS_JSON.name
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
