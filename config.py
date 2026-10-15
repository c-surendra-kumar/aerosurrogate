"""Paths, physical constants and seeds for AeroSurrogate.

Everything here is a value that more than one module needs to agree on. If a
number appears in two places in this project, it belongs in this file.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
SHARD_DIR = DATA_DIR / "_shards"
STAGING_DIR = DATA_DIR / "_staging"
COEFFS_PARQUET = DATA_DIR / "coeffs.parquet"
COEFFS_META = DATA_DIR / "coeffs_meta.json"

ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PT = ARTIFACT_DIR / "model.pt"
MODEL_META = ARTIFACT_DIR / "model_meta.json"

RESULTS_DIR = ROOT / "results"
METRICS_JSON = RESULTS_DIR / "metrics.json"

WEB_DIR = ROOT / "web"
TEST_REFERENCE_JSON = WEB_DIR / "test_reference.json"

# --------------------------------------------------------------------------
# Dataset source
# --------------------------------------------------------------------------

DATASET_URL = "https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip"

# Exact size of Dataset.zip, measured from the server's Content-Length.
# No official checksum is published, so this is our only cheap integrity signal.
DATASET_BYTES = 10_029_067_577

# Members per simulation. `_freestream.vtp` exists but airfrans.Simulation
# never reads it, so we do not fetch it.
SIM_MEMBER_SUFFIXES = ("_internal.vtu", "_aerofoil.vtp")

# The docs' worked example. Recomputing these two numbers end-to-end validates
# the range-request machinery, the zip layout, the pyvista/VTK stack, the
# drag-first tuple order and the compressible=False choice, all at once.
GOLDEN_NAME = "airFoil2D_SST_43.597_5.932_3.551_3.1_1.0_18.252"
GOLDEN_CD = 0.0134
GOLDEN_CL = 0.8099
GOLDEN_ATOL = 5e-4

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------

# Kinematic viscosity used when AirfRANS was GENERATED (paper Table 8 /
# Appendix J). Deliberately NOT airfrans.Simulation.NU == 1.5498e-5, which is
# derived from T=298.15 K and overshoots the dataset's advertised 2-6M Reynolds
# range. With 1.56e-5 the 1,000 inlet velocities map to Re in [2.005e6, 5.9995e6].
NU_GEN = 1.56e-5

# All AirfRANS airfoils have unit chord (paper Appendix H), and the paper takes
# the chord as the 1-D characteristic "area", so force_coefficient() carries no
# chord factor. Both facts only hold because this is 1.0.
CHORD = 1.0

# Reference values airfrans uses internally, recorded for provenance only.
# RHO cancels out of force_coefficient(), so it does not affect our targets.
RHO_REF = 1.1839305888

# CFD cost for the speedup comparison: Bonnet et al., AirfRANS,
# arXiv:2212.07564, Table 4 -- ~25 min on 16 cores of a Ryzen Threadripper 3960X.
CFD_MINUTES_PER_SIM = 25.0
CFD_CORES = 16
CFD_SOURCE = "Bonnet et al., AirfRANS, NeurIPS 2022 D&B, arXiv:2212.07564, Table 4"

# Observed envelope of the full 1,000-simulation dataset, used for sanity
# checks. These are dataset-wide bounds, NOT the training ranges that drive the
# API's extrapolation warnings -- those are measured at training time.
DATASET_AOA_RANGE = (-4.94, 14.93)
DATASET_RE_RANGE = (2.0e6, 6.0e6)

# --------------------------------------------------------------------------
# Splits and seeds
# --------------------------------------------------------------------------

# Our ~4 GB network budget buys 400 simulations, spent on AirfRANS' own
# `scarce` task: 200 training simulations against the same 200-simulation test
# set the `full` task uses. Test set first, so an interrupted run still leaves a
# complete evaluation set.
TASK_SPLITS = ("full_test", "scarce_train")

EXPECTED_SPLIT_SIZES = {
    "full_train": 800,
    "full_test": 200,
    "scarce_train": 200,
    "reynolds_train": 504,
    "reynolds_test": 496,
    "aoa_train": 804,
    "aoa_test": 196,
}

SPLIT_SEED = 1234          # train/validation carve; independent of model seeds
VAL_FRACTION = 0.1         # FR-3
MODEL_SEEDS = (0, 1, 2, 3, 4)

# Relative error for Cl is meaningless where Cl crosses zero (FR-7).
CL_REL_ERR_FLOOR = 0.05
