"""Build `data/coeffs.parquet`: one row per AirfRANS simulation with its
shape parameters, flow conditions and ground-truth Cl/Cd.

The dataset is a 9.34 GiB zip that unpacks to 14.00 GiB, against a 1 GB disk
budget, so nothing is ever downloaded whole. `remote_zip.HttpFile` exposes the
archive as a seekable file, each simulation's two meshes are pulled out
individually, reduced to a handful of scalars, and deleted immediately. Peak
disk is one mesh per worker -- about 90 MB at six workers.

Commands::

    python data.py peek                 # central directory, manifest, exact cost
    python data.py build --limit 10     # smoke test incl. the golden case
    python data.py build                # the 400-simulation run
    python data.py check                # validate the parquet
    python data.py synthetic            # fabricate a parquet for pipeline work

`build` is resumable: results land in 50-simulation shards and an interrupted
run skips what it already has.
"""

from __future__ import annotations

# Must precede numpy/VTK import, in the parent *and* in spawned children (which
# re-import this module). Otherwise every worker starts 16 OpenMP threads and
# six workers thrash the machine 96 ways.
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import naca
from config import (COEFFS_META, COEFFS_PARQUET, DATASET_BYTES, DATASET_URL,
                    DATA_DIR, EXPECTED_SPLIT_SIZES, GOLDEN_ATOL, GOLDEN_CD,
                    GOLDEN_CL, GOLDEN_NAME, NU_GEN, RESULTS_DIR, RHO_REF,
                    SHARD_DIR, SIM_MEMBER_SUFFIXES, STAGING_DIR, TASK_SPLITS)
from remote_zip import HttpFile

CHUNK_SIZE = 50
SPLIT_COLUMNS = [f"in_{k}" for k in EXPECTED_SPLIT_SIZES]


# ---------------------------------------------------------------------------
# Remote archive access
# ---------------------------------------------------------------------------


def open_archive(url: str = DATASET_URL) -> tuple[HttpFile, zipfile.ZipFile, str]:
    """Open the remote zip and derive its internal path prefix.

    The prefix is read from the archive rather than hard-coded: assuming
    "Dataset/" would be an avoidable failure forty minutes into a run.
    """
    fh = HttpFile(url, expected_size=DATASET_BYTES)
    zf = zipfile.ZipFile(fh)
    manifest_member = next(n for n in zf.namelist() if n.endswith("manifest.json"))
    prefix = manifest_member[: -len("manifest.json")]
    return fh, zf, prefix


def read_manifest(zf: zipfile.ZipFile, prefix: str) -> dict[str, list[str]]:
    manifest = json.loads(zf.read(f"{prefix}manifest.json"))
    for key, expected in EXPECTED_SPLIT_SIZES.items():
        got = len(manifest.get(key, []))
        if got != expected:
            raise RuntimeError(
                f"manifest['{key}'] has {got} entries, expected {expected}. "
                "This is not the dataset version this project was built against."
            )
    return manifest


def work_list(manifest: dict[str, list[str]]) -> list[str]:
    """Simulations to process, test set first.

    Order matters: the run is resumable and stoppable, so putting the 200 test
    simulations first means an interruption still leaves a complete evaluation
    set plus however many training rows arrived -- a usable dataset rather than
    a useless partial one.
    """
    seen: set[str] = set()
    names: list[str] = []
    for split in TASK_SPLITS:
        for name in manifest[split]:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_ZF: zipfile.ZipFile | None = None
_PREFIX: str = ""


def _init_worker(url: str, prefix: str) -> None:
    """One archive handle per process. ZipFile is neither picklable nor
    thread-safe, so each worker builds its own from the URL."""
    global _ZF, _PREFIX
    _, _ZF, _ = open_archive(url)
    _PREFIX = prefix


def _geometry_mismatch(sim, naca_params) -> float:
    """Max distance from a real mesh surface point to our reconstructed section.

    Cheap, and it validates on every single simulation that the shape features
    the API will rebuild from NACA digits describe the geometry the CFD
    actually ran on. Also the evidence for whether the 5-digit reflex digit
    carries any information (see naca.camber_line).
    """
    from scipy.spatial import cKDTree

    section = naca.naca_section(naca_params, nb_samples=2000)
    return float(cKDTree(section).query(sim.airfoil_position)[0].max())


def _process_one(name: str) -> dict:
    """Fetch one simulation, reduce it to scalars, delete it."""
    import airfrans as af

    stage = STAGING_DIR / str(os.getpid())
    sim_dir = stage / name
    try:
        sim_dir.mkdir(parents=True, exist_ok=True)
        for suffix in SIM_MEMBER_SUFFIXES:
            member = f"{_PREFIX}{name}/{name}{suffix}"
            (sim_dir / f"{name}{suffix}").write_bytes(_ZF.read(member))  # CRC-checked

        parsed = naca.parse_simulation_name(name)
        sim = af.Simulation(root=str(stage), name=name)

        # reference=True reads the on-disk ground-truth fields.
        # compressible stays False: AirfRANS stores kinematic pressure, and
        # compressible=True would silently shrink both coefficients by ~15.5%.
        (cd, cdp, cdv), (cl, clp, clv) = sim.force_coefficient(
            compressible=False, reference=True
        )

        # Cross-checks against the name, per row rather than once.
        if not np.isclose(float(sim.inlet_velocity), parsed["u_inf"], rtol=1e-6):
            raise ValueError("inlet velocity disagrees with the simulation name")
        if not np.isclose(
            np.degrees(float(sim.angle_of_attack)), parsed["aoa_deg"], atol=1e-6
        ):
            raise ValueError("angle of attack disagrees with the simulation name")

        row = {
            "name": name,
            "series": parsed["series"],
            "u_inf": parsed["u_inf"],
            "aoa_deg": parsed["aoa_deg"],
            "reynolds": parsed["reynolds"],
            "camber_code_0": parsed["naca_params"][0],
            "camber_code_1": parsed["naca_params"][1],
            # 0.0 rather than NaN for 4-digit: the Hour-1 checkpoint requires a
            # NaN-free table, and Q == 0 means "non-reflex", which is correct.
            "camber_code_2": (
                parsed["naca_params"][2] if parsed["series"] == 5 else 0.0
            ),
            "thickness_pct": parsed["naca_params"][-1],
            "cl": float(cl), "cl_p": float(clp), "cl_v": float(clv),
            "cd": float(cd), "cd_p": float(cdp), "cd_v": float(cdv),
            "log_cd": float(np.log(cd)),
            "label_kind": "exact",
            "n_points_internal": int(sim.position.shape[0]),
            "n_points_airfoil": int(sim.airfoil_position.shape[0]),
            "geom_max_dist": _geometry_mismatch(sim, parsed["naca_params"]),
        }
        row.update(naca.geometry_descriptors(parsed["naca_params"]))
        del sim
        return row

    except Exception as exc:  # noqa: BLE001 - one bad mesh must not kill the run
        return {"name": name, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(sim_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_peek(args) -> int:
    t0 = time.perf_counter()
    fh, zf, prefix = open_archive()
    infos = zf.infolist()
    print(f"url            : {DATASET_URL}")
    print(f"size           : {fh.size:,} bytes (matches expected)")
    print(f"members        : {len(infos):,}")
    print(f"prefix         : {prefix!r}")

    manifest = read_manifest(zf, prefix)
    print("\nsplit sizes (all match expected):")
    for key in EXPECTED_SPLIT_SIZES:
        print(f"  {key:16s} {len(manifest[key]):4d}")

    names = work_list(manifest)
    wanted = {
        f"{prefix}{n}/{n}{s}" for n in names for s in SIM_MEMBER_SUFFIXES
    }
    by_name = {i.filename: i for i in infos}
    missing = wanted - by_name.keys()
    if missing:
        print(f"\n!! {len(missing)} expected members are absent, e.g. {sorted(missing)[:3]}")
        return 1

    comp = sum(by_name[m].compress_size for m in wanted)
    raw = sum(by_name[m].file_size for m in wanted)
    print(f"\nwork list      : {len(names)} simulations ({' + '.join(TASK_SPLITS)})")
    print(f"network cost   : {comp / 1e9:.3f} GB compressed")
    print(f"peak staging   : {raw / len(names) / 1e6:.1f} MB per worker")
    print(f"\npeek used {fh.bytes_fetched / 1e6:.2f} MB in {fh.requests_made} requests, "
          f"{time.perf_counter() - t0:.1f}s")
    return 0


def cmd_build(args) -> int:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    fh, zf, prefix = open_archive()
    manifest = read_manifest(zf, prefix)
    names = work_list(manifest)

    if args.limit:
        # Smoke run: the golden case first, then a deterministic spread.
        head = [GOLDEN_NAME] if GOLDEN_NAME in names else []
        rest = [n for n in names if n not in head]
        stride = max(1, len(rest) // max(1, args.limit - len(head)))
        names = (head + rest[::stride])[: args.limit]

    done = _already_done()
    todo = [n for n in names if n not in done]
    print(f"{len(names)} simulations in scope, {len(done)} already done, "
          f"{len(todo)} to fetch, {args.workers} workers")
    if not todo:
        print("nothing to do")
        return _consolidate(manifest)

    chunks = [todo[i : i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
    t_start = time.perf_counter()
    n_done = 0
    workers = max(1, min(args.workers, len(todo)))

    # One pool for the whole run. Each worker opens the archive once in its
    # initializer; building a pool per chunk would re-read the 2.1 MB central
    # directory in every worker on every chunk.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker,
        initargs=(DATASET_URL, prefix),
    ) as pool:
        for ci, chunk in enumerate(chunks):
            rows, errors = [], []
            for row in pool.map(_process_one, chunk):
                (errors if "error" in row else rows).append(row)
                n_done += 1
                rate = n_done / (time.perf_counter() - t_start)
                eta = (len(todo) - n_done) / rate if rate else 0
                print(f"\r  {n_done}/{len(todo)}  {rate*60:.1f}/min  "
                      f"eta {eta/60:.1f} min   ", end="", flush=True)
            print()

            # Shard per chunk, written atomically, so an interrupted run resumes
            # from the last completed chunk rather than from zero.
            if rows:
                shard = SHARD_DIR / f"chunk_{ci:04d}_{int(time.time())}.parquet"
                tmp = shard.with_suffix(".tmp")
                pd.DataFrame(rows).to_parquet(tmp, index=False)
                os.replace(tmp, shard)
            if errors:
                with (SHARD_DIR / "errors.jsonl").open("a", encoding="utf-8") as fp:
                    for err in errors:
                        fp.write(json.dumps(err) + "\n")
                print(f"  !! {len(errors)} failures recorded in errors.jsonl")

    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print(f"\nfetched {fh.bytes_fetched / 1e9:.3f} GB in "
          f"{(time.perf_counter() - t_start) / 60:.1f} min")
    return _consolidate(manifest)


def _already_done() -> set[str]:
    done: set[str] = set()
    for shard in SHARD_DIR.glob("chunk_*.parquet"):
        try:
            done |= set(pd.read_parquet(shard, columns=["name"])["name"])
        except Exception as exc:  # noqa: BLE001
            print(f"  (ignoring unreadable shard {shard.name}: {exc})")
    return done


def _consolidate(manifest: dict[str, list[str]]) -> int:
    shards = sorted(SHARD_DIR.glob("chunk_*.parquet"))
    if not shards:
        print("no shards to consolidate")
        return 1

    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df = df.drop_duplicates(subset="name").sort_values("name").reset_index(drop=True)

    for key in EXPECTED_SPLIT_SIZES:
        members = set(manifest[key])
        df[f"in_{key}"] = df["name"].isin(members)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(COEFFS_PARQUET, index=False, compression="zstd")

    import pyvista
    import importlib.metadata as md

    COEFFS_META.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_url": DATASET_URL,
                "source_bytes": DATASET_BYTES,
                "n_rows": int(len(df)),
                "task_splits": list(TASK_SPLITS),
                "nu_used": NU_GEN,
                "rho_reference": RHO_REF,
                "chord": 1.0,
                "compressible": False,
                "force_coefficient_reference": True,
                "versions": {
                    "airfrans": md.version("airfrans"),
                    "pyvista": pyvista.__version__,
                    "vtk": md.version("vtk"),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "python": sys.version.split()[0],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {COEFFS_PARQUET} ({len(df)} rows, {COEFFS_PARQUET.stat().st_size/1024:.0f} KB)")
    return 0


def cmd_refresh(args) -> int:
    """Recompute the stored geometry descriptors in place, no network.

    The descriptors are a pure function of the NACA digits, which are already
    in the parquet, so a change to their definition does not require re-fetching
    4 GB of meshes. The model does not read these columns -- `features.py`
    rebuilds them from the digits -- but the checks and the README do.
    """
    df = pd.read_parquet(COEFFS_PARQUET)
    updated = pd.DataFrame(
        [naca.geometry_descriptors(naca.params_from_row(r)) for _, r in df.iterrows()],
        index=df.index,
    )
    changed = [c for c in updated.columns if not np.allclose(df[c], updated[c])]
    df[updated.columns] = updated
    df.to_parquet(COEFFS_PARQUET, index=False, compression="zstd")
    print(f"refreshed {len(df)} rows; columns that changed: {changed or 'none'}")
    return 0


def cmd_check(args) -> int:
    df = pd.read_parquet(COEFFS_PARQUET)
    lines: list[str] = []
    failures = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if not ok:
            failures += 1
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))

    n = len(df)
    check("rows present", n > 0, f"n={n}")
    check("names unique", df["name"].is_unique)
    nan_cols = df.columns[df.isna().any()].tolist()
    check("no NaNs", not nan_cols, f"offenders={nan_cols}")

    check("test split complete", int(df["in_full_test"].sum()) == 200,
          f"n={int(df['in_full_test'].sum())}")
    check("train split complete", int(df["in_scarce_train"].sum()) == 200,
          f"n={int(df['in_scarce_train'].sum())}")
    check("splits disjoint", int((df["in_full_test"] & df["in_scarce_train"]).sum()) == 0)
    check("scarce_train subset of full_train",
          bool((~df["in_scarce_train"] | df["in_full_train"]).all()))

    check("cl in [-1.5, 2.5]", bool(df["cl"].between(-1.5, 2.5).all()),
          f"range=[{df['cl'].min():.4f}, {df['cl'].max():.4f}]")
    check("cd strictly positive and small", bool(df["cd"].between(1e-4, 0.5).all()),
          f"range=[{df['cd'].min():.6f}, {df['cd'].max():.6f}]")
    check("Re in [2.0e6, 6.0e6]", bool(df["reynolds"].between(2.0e6, 6.0e6).all()),
          f"range=[{df['reynolds'].min():.4e}, {df['reynolds'].max():.4e}]")
    check("aoa in [-5, 15] deg", bool(df["aoa_deg"].between(-5.0, 15.0).all()),
          f"range=[{df['aoa_deg'].min():.3f}, {df['aoa_deg'].max():.3f}]")

    check("cd == cd_p + cd_v", bool(np.allclose(df["cd"], df["cd_p"] + df["cd_v"])))
    check("cl == cl_p + cl_v", bool(np.allclose(df["cl"], df["cl_p"] + df["cl_v"])))

    five = df[df["series"] == 5]
    check("5-digit Q is binary",
          set(five["camber_code_2"].round(6).unique()) <= {0.0, 1.0},
          f"values={sorted(five['camber_code_2'].round(6).unique())[:6]}")
    # 24% of the 4-digit simulations have P exactly 0, which sends camber_line
    # down a branch whose returned slope contradicts its own camber line. We do
    # not reject those rows -- the CFD says they are simply uncambered -- but we
    # do assert the descriptors describe them consistently.
    four = df[df["series"] == 4]
    degenerate = four[four["camber_code_1"].abs() < 1e-9]
    check("P == 0 sections are described as uncambered",
          bool((degenerate["alpha_L0_deg"].abs() < 1e-9).all()),
          f"n={len(degenerate)} of {len(four)} four-digit rows")
    check("no 4-digit P in the numerically unstable (0, 0.05) band",
          bool(((four["camber_code_1"] == 0) | (four["camber_code_1"] > 0.05)).all()))
    check("zero camber implies zero zero-lift angle",
          bool((df.loc[df["max_camber"].abs() < 1e-12, "alpha_L0_deg"].abs() < 1e-9).all()))

    check("max_thickness == thickness_pct/100",
          bool(np.allclose(df["max_thickness"], df["thickness_pct"] / 100, atol=1e-3)))
    check("reconstructed geometry matches mesh", bool((df["geom_max_dist"] < 5e-3).all()),
          f"max={df['geom_max_dist'].max():.2e} chord")

    r = float(np.corrcoef(df["aoa_deg"], df["cl"])[0, 1])
    check("lift correlates with incidence", r > 0.8, f"pearson r={r:.4f}")

    if GOLDEN_NAME in set(df["name"]):
        g = df[df["name"] == GOLDEN_NAME].iloc[0]
        check("golden case cd", abs(g["cd"] - GOLDEN_CD) < GOLDEN_ATOL, f"{g['cd']:.6f}")
        check("golden case cl", abs(g["cl"] - GOLDEN_CL) < GOLDEN_ATOL, f"{g['cl']:.6f}")

    lines.append("")
    lines.append(f"series split: {int((df['series']==4).sum())} four-digit, "
                 f"{int((df['series']==5).sum())} five-digit")
    lines.append(f"viscous share of drag: mean {float((df['cd_v']/df['cd']).mean())*100:.1f}%")
    lines.append(f"viscous share of lift: mean {float((df['cl_v']/df['cl']).abs().mean())*100:.3f}%")

    report = "\n".join(lines)
    print(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "data_checks.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\n{failures} failure(s); report written to results/data_checks.txt")
    return 1 if failures else 0


def cmd_synthetic(args) -> int:
    """Fabricate a physically-shaped parquet so the rest of the pipeline can be
    built and tested while the real stream is still running.

    Targets come from thin-airfoil theory plus a drag polar -- plausible enough
    to exercise every downstream code path, and nowhere near good enough to
    report. Written to a distinct filename so it can never be mistaken for the
    real thing.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        five = i % 2 == 1
        u_inf = rng.uniform(31.283, 93.592)
        aoa = rng.uniform(-4.94, 14.93)
        thick = rng.uniform(6.0, 20.0)
        if five:
            params = [rng.uniform(0.1, 5.5), rng.uniform(1.5, 5.0),
                      float(rng.integers(0, 2)), thick]
        else:
            params = [rng.uniform(0.0, 6.0), rng.uniform(1.5, 9.0), thick]
        name = "airFoil2D_SST_" + "_".join(f"{v:.3f}" for v in [u_inf, aoa, *params])

        desc = naca.geometry_descriptors(params)
        aoa_eff = aoa - desc["alpha_L0_deg"]
        cl = 0.1097 * aoa_eff * (1 - 0.0015 * aoa_eff**2) + rng.normal(0, 0.004)
        cd = (0.006 + 0.010 * desc["max_thickness"] / 0.12 + 0.012 * cl**2
              + 0.0004 * (5.0e6 / (u_inf / NU_GEN)) + abs(rng.normal(0, 2e-4)))
        rows.append({
            "name": name, "series": 5 if five else 4, "u_inf": u_inf, "aoa_deg": aoa,
            "reynolds": u_inf / NU_GEN,
            "camber_code_0": params[0], "camber_code_1": params[1],
            "camber_code_2": params[2] if five else 0.0, "thickness_pct": thick,
            "cl": cl, "cl_p": cl * 0.9998, "cl_v": cl * 0.0002,
            "cd": cd, "cd_p": cd * 0.42, "cd_v": cd * 0.58, "log_cd": float(np.log(cd)),
            "label_kind": "SYNTHETIC", "n_points_internal": 180000,
            "n_points_airfoil": 1000, "geom_max_dist": 0.0, **desc,
        })

    df = pd.DataFrame(rows)
    df["in_full_test"] = np.arange(len(df)) < 200
    df["in_scarce_train"] = ~df["in_full_test"]
    df["in_full_train"] = df["in_scarce_train"]
    for key in ("aoa_train", "aoa_test", "reynolds_train", "reynolds_test"):
        df[f"in_{key}"] = False

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "coeffs_synth.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    print(f"wrote {out} ({len(df)} SYNTHETIC rows -- not real results)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("peek", help="inspect the remote archive without fetching meshes")

    p_build = sub.add_parser("build", help="stream simulations and compute Cl/Cd")
    p_build.add_argument("--limit", type=int, default=0, help="smoke-test this many")
    p_build.add_argument("--workers", type=int, default=6)

    sub.add_parser("check", help="validate data/coeffs.parquet")
    sub.add_parser("refresh", help="recompute stored geometry descriptors in place")
    sub.add_parser("synthetic", help="fabricate a parquet for pipeline development")

    args = parser.parse_args(argv)
    return {
        "peek": cmd_peek, "build": cmd_build, "check": cmd_check,
        "refresh": cmd_refresh, "synthetic": cmd_synthetic,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
