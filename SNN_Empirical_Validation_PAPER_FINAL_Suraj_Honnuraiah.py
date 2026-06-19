# ============================================================
# Empirical validation code for:
# "Spiking Neural Networks Are Not Transformers (Yet):
#  Feedforward Theory and Architectural Constraints in Modeling
#  Long-Range Dependencies"
#
# Code author:
#   Suraj Honnuraiah
#   Harvard University
#   Suraj_Honnuraiah@hms.harvard.edu
#
# Purpose and scope:
#   This script generates the empirical validation analyses accompanying the
#   paper. The experiments are finite synthetic validation studies; they are
#   not intended to prove the PAC/covering-number theorem. The analysis tests
#   whether the qualitative sequence-length phenotype predicted by the theory
#   is visible in controlled tasks:
#
#     1. Accuracy degradation as sequence length S increases.
#     2. Validation-selected samples-to-threshold M*(S), with right-censored
#        points marked explicitly.
#     3. Direct comparison among the seven paper-facing architectures:
#          - nLIF baseline
#          - nLIF + inhibition
#          - nLIF WTA16
#          - nLIF WTA16 + refractory constraint
#          - LIF
#          - GRU
#          - Transformer
#     4. Hidden spike-participation density as the mechanistic proxy for
#        effective causal-set growth.
#     5. Optional cue-recall token ablation at the largest M as a supplementary
#        proxy for diffuse causal-time participation.
#
# Main outputs:
#   - run_config.json
#   - variant_config_table.csv
#   - parameter_table.csv
#   - raw_results.csv
#   - mstar_samples_to_threshold.csv
#   - right_censoring_summary.csv
#   - accuracy_at_largest_M_summary.csv
#   - long_sequence_accuracy_summary.csv
#   - spike_participation_at_largest_M_summary.csv
#   - spike_participation_density_slopes.csv
#   - paired_accuracy_vs_nlif_largest_M.csv
#   - paired_accuracy_vs_nlif_longS_largest_M.csv
#   - paired_mstar_vs_nlif.csv
#   - beta_scaling_summary.csv
#   - sequence_robustness_auc.csv
#   - sequence_robustness_auc_stats_vs_nlif.csv
#   - manuscript-ready figures in PNG, PDF, and SVG formats
# ============================================================

import os
import json
import time
import math
import random
import shutil
import platform
import warnings
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for reproducible figure generation
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from IPython.display import display
except Exception:
    def display(x):
        print(x)

# ============================================================
# 0. Global configuration
# ============================================================

def env_flag(name: str, default: bool = False) -> bool:
    """Parse simple boolean environment flags."""
    val = os.environ.get(name, None)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)

def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return float(default)


def _cli_to_env():
    """
    Allow both environment-variable and command-line control.

    Environment variables and command-line flags are both supported so that the
    same paper analysis can be reproduced from a script or notebook. Unknown
    arguments are ignored so the file can run safely in managed notebook kernels.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Paper-final empirical validation for SNN sequence-length scaling. "
            "Environment variables remain supported; CLI flags override them."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Run preset: smoke_test, quick_test, focused_validation, "
            "paper_stats_beta_optimized, paper_stats_beta, or beta_scaling_core."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None, help="Execution device selection.")
    parser.add_argument("--compute-token-ablation", action="store_true", help="Enable cue-recall token ablation.")
    parser.add_argument("--no-token-ablation", action="store_true", help="Disable token ablation.")
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable progress bars.")
    parser.add_argument("--deterministic", action="store_true", help="Use strict deterministic torch algorithms when available.")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint CSV.")
    parser.add_argument("--num-threads", type=int, default=None, help="Torch thread count for local execution.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Evaluation batch size.")
    parser.add_argument("--test-n", type=int, default=None, help="Test set size per task/seed/S.")
    parser.add_argument("--calib-n", type=int, default=None, help="Teacher calibration set size.")
    parser.add_argument("--fixed-validation-n", type=int, default=None, help="Fixed external validation set size.")
    parser.add_argument("--stats-bootstraps", type=int, default=None, help="Bootstrap count for paired summaries.")
    parser.add_argument("--beta-bootstraps", type=int, default=None, help="Bootstrap count for empirical beta summaries.")
    parser.add_argument("--tobit-bootstraps", type=int, default=None, help="Bootstrap count for Tobit beta summaries.")
    parser.add_argument("--tobit-maxiter", type=int, default=None, help="Max optimizer iterations for Tobit fits.")

    args, _unknown = parser.parse_known_args()

    def set_env_if(name: str, value):
        if value is not None:
            os.environ[name] = str(value)

    set_env_if("SNN_RUN_PRESET", args.preset)
    set_env_if("SNN_OUTPUT_DIR", args.output_dir)
    set_env_if("SNN_NUM_THREADS", args.num_threads)
    set_env_if("SNN_EPOCHS", args.epochs)
    set_env_if("SNN_EARLY_STOPPING_PATIENCE", args.patience)
    set_env_if("SNN_BATCH_SIZE", args.batch_size)
    set_env_if("SNN_EVAL_BATCH_SIZE", args.eval_batch_size)
    set_env_if("SNN_TEST_N", args.test_n)
    set_env_if("SNN_CALIB_N", args.calib_n)
    set_env_if("SNN_FIXED_VALIDATION_N", args.fixed_validation_n)
    set_env_if("SNN_STATS_BOOTSTRAPS", args.stats_bootstraps)
    set_env_if("SNN_BETA_BOOTSTRAPS", args.beta_bootstraps)
    set_env_if("SNN_TOBIT_BOOTSTRAPS", args.tobit_bootstraps)
    set_env_if("SNN_TOBIT_MAXITER", args.tobit_maxiter)

    set_env_if("SNN_DEVICE", args.device)
    if args.compute_token_ablation and args.no_token_ablation:
        raise ValueError("Choose only one of --compute-token-ablation or --no-token-ablation.")
    if args.compute_token_ablation:
        os.environ["SNN_COMPUTE_TOKEN_ABLATION"] = "1"
    if args.no_token_ablation:
        os.environ["SNN_COMPUTE_TOKEN_ABLATION"] = "0"
    if args.disable_tqdm:
        os.environ["SNN_DISABLE_TQDM"] = "1"
    if args.deterministic:
        os.environ["SNN_DETERMINISTIC"] = "1"
    if args.no_resume:
        os.environ["SNN_RESUME"] = "0"


_cli_to_env()

# Execution device. Default "auto" uses CUDA when available and otherwise uses
# the standard PyTorch local device. Use SNN_DEVICE or --device to override.
DEVICE_MODE = os.environ.get("SNN_DEVICE", "auto").strip().lower()
if DEVICE_MODE not in {"auto", "cpu", "cuda"}:
    raise ValueError("SNN_DEVICE must be one of: auto, cpu, cuda.")
if DEVICE_MODE == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("SNN_DEVICE=cuda was requested, but CUDA is not available.")
    device = torch.device("cuda")
elif DEVICE_MODE == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cpu":
    requested_threads = env_int("SNN_NUM_THREADS", 0)
    torch_threads = requested_threads if requested_threads > 0 else min(8, os.cpu_count() or 1)
    try:
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

print("Using device:", device)

# Paper-analysis run by default.
# Options: smoke_test, quick_test, focused_validation, paper_stats_beta_optimized, beta_scaling_core
RUN_PRESET = os.environ.get("SNN_RUN_PRESET", "paper_stats_beta_optimized")

# For this theory paper, the experiments are auxiliary empirical validation.
# Smoke and paper deliberately include both the theorem-aligned teacher-student
# task and a delayed cue-recall task requested as a long-range dependency check.
TASKS_TO_RUN = ["teacher_student", "cue_recall"]

OUTPUT_DIR = os.environ.get("SNN_OUTPUT_DIR", f"snn_scaling_outputs_{RUN_PRESET}")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Main threshold used for sample-to-threshold curves.
# 0.75 is the main manuscript-facing threshold. Additional thresholds are used
# as robustness checks in the full paper run.
# Default; individual presets below may override this.
ACCURACY_THRESHOLDS = [0.75]

# Optional supplementary token-ablation settings.
# We compute this only for cue_recall at the largest M, because that is the
# cleanest long-range dependency setting and avoids distracting diagnostics.
SENS_MAX_SAMPLES = env_int("SNN_SENS_MAX_SAMPLES", 256)
CAUSAL_EFFECT_THRESHOLD = env_float("SNN_CAUSAL_EFFECT_THRESHOLD", 1e-3)
# For the optimized paper run, token ablation is disabled by default because it
# can add many forward passes. Enable it separately with:
#   $env:SNN_COMPUTE_TOKEN_ABLATION="1"
COMPUTE_TOKEN_ABLATION = env_flag("SNN_COMPUTE_TOKEN_ABLATION", False)
TOKEN_ABLATION_TASKS = ["cue_recall"]
TOKEN_ABLATION_ONLY_MAX_M = True

# Disable progress bars when running unattended or redirecting logs.
DISABLE_TQDM = env_flag("SNN_DISABLE_TQDM", False)

# Bootstrap settings. Paired stats can use many cheap bootstraps; empirical beta
# uses fewer, and Tobit uses far fewer because each bootstrap calls an optimizer.
STATS_BOOTSTRAPS = env_int("SNN_STATS_BOOTSTRAPS", 5000)
BETA_BOOTSTRAPS = env_int("SNN_BETA_BOOTSTRAPS", 1000)
TOBIT_BOOTSTRAPS = env_int("SNN_TOBIT_BOOTSTRAPS", 100)
TOBIT_MAXITER = env_int("SNN_TOBIT_MAXITER", 600)
STATS_RANDOM_SEED = 12345

# Keep model initialization fixed across training sample sizes M for each
# task/seed/S/variant. This makes M*(S) cleaner: the training set size changes,
# but the starting model does not. Training/shuffling seeds still depend on M.
MODEL_INIT_SHARED_ACROSS_M = True

# Full deterministic algorithm enforcement can increase runtime. Seeds are fixed by
# condition; set SNN_DETERMINISTIC=1 if stricter deterministic kernels are needed.
DETERMINISTIC_ALGORITHMS = env_flag("SNN_DETERMINISTIC", False)

# Use one fixed validation set per task/seed/S across all training sizes M.
# This makes M*(S) cleaner: model initialization and validation distribution are
# fixed while the training sample size changes. The validation set is generated
# separately from the M training pool, so M denotes the actual number of training
# examples used for fitting.
USE_FIXED_VALIDATION_SET = True
# Runtime optimization: keep validation large enough for stable early stopping,
# but do not scale it to 15% of 65,536 in paper mode. Override if needed:
#   $env:SNN_FIXED_VALIDATION_N="4096"
FIXED_VALIDATION_N = os.environ.get("SNN_FIXED_VALIDATION_N", None)
FIXED_VALIDATION_N = None if FIXED_VALIDATION_N in (None, "", "None", "none") else int(FIXED_VALIDATION_N)

# Robust long-run behavior: write each completed training row to disk and skip
# completed rows on restart. Disable with $env:SNN_RESUME="0".
RESUME_FROM_CHECKPOINT = env_flag("SNN_RESUME", True)
CHECKPOINT_EVERY_ROW = env_flag("SNN_CHECKPOINT_EVERY_ROW", True)

# ------------------------------------------------------------
# Presets
# ------------------------------------------------------------

if RUN_PRESET == "smoke_test":
    # Ultra-fast end-to-end test: all seven variants, one task/S/M/seed.
    # This checks model creation, training, checkpointing, analysis, plotting, and zip output.
    TASKS_TO_RUN = ["teacher_student"]
    SEQ_LENGTHS = [8]
    TRAIN_SIZES = [512]
    SEEDS = [0]
    EPOCHS = 1
    EARLY_STOPPING_PATIENCE = 1
    TEST_N = 256
    CALIB_N = 256
    ACCURACY_THRESHOLDS = [0.75]
    if FIXED_VALIDATION_N is None:
        FIXED_VALIDATION_N = 128
    VARIANTS_TO_RUN = [
        "nlif_baseline",
        "nlif_lateral_inh_only",
        "nlif_wta16",
        "nlif_wta16_ref",
        "lif_baseline",
        "gru",
        "transformer",
    ]

elif RUN_PRESET == "quick_test":
    # Fast sanity check: verifies environment, plotting, and output generation.
    SEQ_LENGTHS = [8, 16]
    TRAIN_SIZES = [512, 1024]
    SEEDS = [0]
    EPOCHS = 2
    EARLY_STOPPING_PATIENCE = 1
    TEST_N = 512
    CALIB_N = 512
    ACCURACY_THRESHOLDS = [0.75]
    VARIANTS_TO_RUN = [
        "nlif_baseline",
        "nlif_lateral_inh_only",
        "nlif_wta16",
        "nlif_wta16_ref",
        "lif_baseline",
        "gru",
        "transformer",
    ]

elif RUN_PRESET == "focused_validation":
    # Focused validation preset.
    # Goal: compact run grid that still tests the main empirical claims.
    SEQ_LENGTHS = [8, 16, 32, 64, 128]
    TRAIN_SIZES = [512, 1024, 2048, 4096, 8192, 16384]
    SEEDS = [0, 1, 2]
    EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 4
    TEST_N = 5000
    CALIB_N = 5000
    ACCURACY_THRESHOLDS = [0.75]
    VARIANTS_TO_RUN = [
        "nlif_baseline",
        "nlif_lateral_inh_only",
        "nlif_wta16",
        "nlif_wta16_ref",
        "lif_baseline",
        "gru",
        "transformer",
    ]

elif RUN_PRESET in {"paper_stats_beta", "paper_stats_beta_optimized"}:
    # Full all-seven-variant paper run.
    # This is the manuscript grid for M*(S), empirical beta summaries,
    # sequence-robustness AUC, and hidden spike-participation analyses.
    SEQ_LENGTHS = [8, 12, 16, 24, 32, 48, 64, 96, 128]
    TRAIN_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    SEEDS = [0, 1, 2, 3, 4]
    EPOCHS = env_int("SNN_EPOCHS", 20)
    EARLY_STOPPING_PATIENCE = env_int("SNN_EARLY_STOPPING_PATIENCE", 4)
    TEST_N = env_int("SNN_TEST_N", 6000)
    CALIB_N = env_int("SNN_CALIB_N", 6000)
    # Main threshold remains 0.75. 0.70 and 0.80 are robustness checks.
    ACCURACY_THRESHOLDS = [0.70, 0.75, 0.80]
    if FIXED_VALIDATION_N is None:
        FIXED_VALIDATION_N = 2048
    VARIANTS_TO_RUN = [
        "nlif_baseline",
        "nlif_lateral_inh_only",
        "nlif_wta16",
        "nlif_wta16_ref",
        "lif_baseline",
        "gru",
        "transformer",
    ]

elif RUN_PRESET == "beta_scaling_core":
    # Focused beta-scaling run. Use only if the full all-seven run is too heavy.
    SEQ_LENGTHS = [8, 12, 16, 24, 32, 48, 64, 96, 128]
    TRAIN_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    SEEDS = [0, 1, 2, 3, 4]
    EPOCHS = env_int("SNN_EPOCHS", 20)
    EARLY_STOPPING_PATIENCE = env_int("SNN_EARLY_STOPPING_PATIENCE", 4)
    TEST_N = env_int("SNN_TEST_N", 6000)
    CALIB_N = env_int("SNN_CALIB_N", 6000)
    ACCURACY_THRESHOLDS = [0.70, 0.75, 0.80]
    if FIXED_VALIDATION_N is None:
        FIXED_VALIDATION_N = 2048
    VARIANTS_TO_RUN = [
        "nlif_baseline",
        "lif_baseline",
        "gru",
        "transformer",
    ]

else:
    raise ValueError(
        "Unknown RUN_PRESET. Use smoke_test, quick_test, focused_validation, "
        "paper_stats_beta_optimized, paper_stats_beta, or beta_scaling_core."
    )
# ------------------------------------------------------------
# Model / data parameters
# ------------------------------------------------------------

D_IN = 16
# Parameter-matched model configuration.
# Training sample sizes are already identical across architectures. The main
# fairness issue is parameter count. We therefore use variant-specific hidden
# widths so all seven architectures stay in the same ~13k-parameter regime.
HIDDEN = 64  # retained as a fallback/default; most models use HIDDEN_BY_VARIANT below.
SNN_LAYERS = 2
N_CLASSES = 2

VARIANT_ORDER = [
    "nlif_baseline",
    "nlif_lateral_inh_only",
    "nlif_wta16",
    "nlif_wta16_ref",
    "lif_baseline",
    "gru",
    "transformer",
]

VARIANT_DISPLAY_NAMES = {
    "nlif_baseline": "nLIF baseline",
    "nlif_lateral_inh_only": "nLIF + inhibition",
    "nlif_wta16": "nLIF WTA16",
    "nlif_wta16_ref": "nLIF WTA16 + ref",
    "lif_baseline": "LIF",
    "gru": "GRU",
    "transformer": "Transformer",
}

# Chosen to bring all variants close to the same parameter budget:
#   nLIF/LIF baseline, H=106:       ~13.1k parameters
#   inhibitory/WTA SNNs, H=64:      ~13.4k parameters
#   GRU, H=58:                      ~13.3k parameters
#   Transformer, H=32:              ~13.3k parameters at max_len=128
HIDDEN_BY_VARIANT = {
    "nlif_baseline": 106,
    "nlif_lateral_inh_only": 64,
    "nlif_wta16": 64,
    "nlif_wta16_ref": 64,
    "lif_baseline": 106,
    "gru": 58,
    "transformer": 32,
}

SPIKE_PROB = 0.10

BATCH_SIZE = env_int("SNN_BATCH_SIZE", 512)  # override if memory-limited
EVAL_BATCH_SIZE = env_int("SNN_EVAL_BATCH_SIZE", 1024)
LR = 2e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

# Early stopping mitigates the fixed-epoch optimization confound. The experiments
# still remain auxiliary empirical checks, not optimizer-independent PAC estimates.
USE_EARLY_STOPPING = True
VALIDATION_FRACTION = 0.15
MIN_VAL_IMPROVEMENT = 1e-4

# Teacher parameters.
TEACHER_HIDDEN = 48
TEACHER_LAYERS = 1
TEACHER_WEIGHT_SCALE = 0.20
TEACHER_THRESHOLD = 0.75

# Student SNN parameters.
SNN_WEIGHT_SCALE = 0.15
SNN_THRESHOLD = 1.0

# Optional spike regularization.
DEFAULT_SPIKE_REG = 0.0

print("\nConfiguration:")
print("RUN_PRESET:", RUN_PRESET)
print("TASKS_TO_RUN:", TASKS_TO_RUN)
print("SEQ_LENGTHS:", SEQ_LENGTHS)
print("TRAIN_SIZES:", TRAIN_SIZES)
print("SEEDS:", SEEDS)
print("EPOCHS:", EPOCHS)
print("EARLY_STOPPING:", USE_EARLY_STOPPING)
print("VARIANTS:", VARIANTS_TO_RUN)
print("TOTAL_MODEL_TRAININGS:", len(TASKS_TO_RUN) * len(SEQ_LENGTHS) * len(TRAIN_SIZES) * len(SEEDS) * len(VARIANTS_TO_RUN))

# Save run metadata early so outputs are self-documenting.
def save_run_config():
    config = {
        "paper_title": "Spiking Neural Networks Are Not Transformers (Yet): Feedforward Theory and Architectural Constraints in Modeling Long-Range Dependencies",
        "code_author": "Suraj Honnuraiah",
        "code_affiliation": "Harvard University",
        "code_contact": "Suraj_Honnuraiah@hms.harvard.edu",
        "affiliation": "Harvard University",
        "contact_email": "Suraj_Honnuraiah@hms.harvard.edu",
        "created_unix_time": time.time(),
        "run_preset": RUN_PRESET,
        "tasks_to_run": TASKS_TO_RUN,
        "sequence_lengths": SEQ_LENGTHS,
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "use_early_stopping": USE_EARLY_STOPPING,
        "validation_fraction": VALIDATION_FRACTION,
        "test_n": TEST_N,
        "calib_n": CALIB_N,
        "batch_size": BATCH_SIZE,
        "eval_batch_size": EVAL_BATCH_SIZE,
        "variants_to_run": VARIANTS_TO_RUN,
        "accuracy_thresholds": ACCURACY_THRESHOLDS,
        "hidden_by_variant": HIDDEN_BY_VARIANT,
        "model_init_shared_across_M": MODEL_INIT_SHARED_ACROSS_M,
        "deterministic_algorithms": DETERMINISTIC_ALGORITHMS,
        "disable_tqdm": DISABLE_TQDM,
        "use_fixed_validation_set": USE_FIXED_VALIDATION_SET,
        "fixed_validation_n": FIXED_VALIDATION_N,
        "compute_token_ablation": COMPUTE_TOKEN_ABLATION,
        "token_ablation_tasks": TOKEN_ABLATION_TASKS,
        "token_ablation_only_max_M": TOKEN_ABLATION_ONLY_MAX_M,
        "causal_effect_threshold": CAUSAL_EFFECT_THRESHOLD,
        "stats_bootstraps": STATS_BOOTSTRAPS,
        "beta_bootstraps": BETA_BOOTSTRAPS,
        "tobit_bootstraps": TOBIT_BOOTSTRAPS,
        "tobit_maxiter": TOBIT_MAXITER,
        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
        "checkpoint_every_row": CHECKPOINT_EVERY_ROW,
        "device": str(device),
        "device_mode": DEVICE_MODE,
        "torch_num_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
    }
    path = os.path.join(OUTPUT_DIR, "run_config.json")
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print("Saved run config:", path)

def write_methods_readme():
    """Write a compact manuscript-facing methods/README file into the output directory."""
    path = os.path.join(OUTPUT_DIR, "README_methods_and_outputs.md")
    script_name = os.path.basename(__file__) if "__file__" in globals() else "SNN empirical validation script"
    content = f"""# Empirical validation for SNN sequence-length scaling

Paper: *Spiking Neural Networks Are Not Transformers (Yet): Feedforward Theory and Architectural Constraints in Modeling Long-Range Dependencies*

Code author: Suraj Honnuraiah, Harvard University  
Contact: Suraj_Honnuraiah@hms.harvard.edu

This folder was generated by `{script_name}`.

## Preset and run grid

- Run preset: `{RUN_PRESET}`
- Tasks: `{', '.join(TASKS_TO_RUN)}`
- Sequence lengths: `{SEQ_LENGTHS}`
- Training sizes: `{TRAIN_SIZES}`
- Seeds: `{SEEDS}`
- Max epochs: `{EPOCHS}`
- Early stopping patience: `{EARLY_STOPPING_PATIENCE}`
- Test set size per task/seed/S: `{TEST_N}`
- Calibration set size for teacher-student threshold: `{CALIB_N}`
- Fixed validation set size: `{FIXED_VALIDATION_N}`
- Accuracy thresholds for M*(S): `{ACCURACY_THRESHOLDS}`

## Interpretation

These experiments are finite synthetic validation experiments. They are not direct
estimates of formal PAC sample complexity and do not prove the covering-number
theorem. They test whether the qualitative phenotype predicted by the theory is
visible in controlled sequence-length sweeps: accuracy degradation with sequence
length, increased samples-to-threshold M*(S), right-censoring at finite sample
budgets, and changes in hidden spike participation under inhibitory constraints.

## Architecture comparison

The main all-seven run compares: nLIF baseline, nLIF with learned lateral
inhibition, nLIF WTA16, nLIF WTA16 plus refractory constraint, LIF, GRU, and
Transformer. Parameter counts are saved in `parameter_table.csv`.

Important scope note: the theorem-aligned baseline SNNs use positive feedforward
weights. The inhibitory/WTA variants are empirical circuit-level controls and
should be interpreted as constraints on effective causal participation, not as
members of the strict positive-weight theorem class.

## Right-censoring and beta estimates

`M_star_observed` is the first training size at which the validation-selected model reaches the
validation accuracy threshold. If the threshold is not reached, the condition is marked as
right-censored and `M_star_censored_as_max` is set to the largest tested M.

`beta_scaling_summary.csv` reports descriptive empirical beta fits from M*(S),
including observed-only, censored-as-max, and Tobit-style right-censored fits.
These beta estimates are finite-range, protocol-dependent summaries and should
not be interpreted as direct PAC-exponent estimates.

## Main outputs

- `raw_results.csv`: all per-run measurements.
- `mstar_samples_to_threshold.csv`: M*(S) and right-censoring information.
- `right_censoring_summary.csv`: summary of threshold reachability.
- `accuracy_at_largest_M_summary.csv`: mean/SEM accuracy at largest M.
- `paired_accuracy_vs_nlif_largest_M.csv`: paired accuracy deltas vs nLIF.
- `paired_mstar_vs_nlif.csv`: paired log2 M* deltas vs nLIF.
- `beta_scaling_summary.csv`: descriptive empirical beta estimates from M*(S).
- `sequence_robustness_auc.csv`: normalized sequence-robustness summaries.
- `spike_participation_at_largest_M_summary.csv`: activity-based proxy for causal participation.
- `parameter_table.csv` and `variant_config_table.csv`: model and variant definitions.
- PNG/PDF/SVG figure files for manuscript and supplementary figure assembly.
- `missing_conditions.csv`: empty when all requested conditions completed.

## Recommended manuscript wording

"We interpret M*(S), empirical beta, and Tobit summaries as finite-sample
descriptive signatures rather than direct PAC estimates. Right-censored points
indicate that threshold accuracy was not reached within the tested sample budget;
censored-as-max values therefore provide conservative descriptive summaries."
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved methods README:", path)


save_run_config()
write_methods_readme()

# ============================================================
# 1. Utilities
# ============================================================

TASK_SEED_OFFSET = {
    "teacher_student": 1009,
    "cue_recall": 2003,
}

VARIANT_SEED_OFFSET = {
    "nlif_baseline": 11,
    "nlif_lateral_inh_only": 18,
    "nlif_wta16": 19,
    "nlif_wta16_ref": 29,
    "lif_baseline": 31,
    "gru": 41,
    "transformer": 43,
}

ROLE_SEED_OFFSET = {
    "teacher": 101,
    "data": 211,
    "model": 307,
    "train": 401,
    "metrics": 503,
}


def set_seed(seed: int, deterministic: Optional[bool] = None):
    """
    Set all relevant seeds.

    Strict deterministic kernels can increase runtime. warn_only=True avoids crashes if a
    deterministic kernel is unavailable.
    """
    seed = int(seed) % (2**31 - 1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic is None:
        deterministic = DETERMINISTIC_ALGORITHMS

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def make_condition_seed(
    base_seed: int,
    S: int,
    M: int,
    variant: str,
    role: str = "model",
    task: str = "teacher_student",
) -> int:
    """
    Stable deterministic seed for each experimental condition.

    Includes task, variant, sequence length, sample size, and role. This avoids
    Python hash nondeterminism and makes reruns reproducible.
    """
    return int(
        base_seed * 1_000_003
        + TASK_SEED_OFFSET.get(task, 3001) * 10_000_019
        + S * 10_007
        + M * 101
        + VARIANT_SEED_OFFSET.get(variant, 997)
        + ROLE_SEED_OFFSET.get(role, 0)
    ) % (2**31 - 1)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def savefig(name: str):
    """Save publication-ready raster and vector versions of the current figure."""
    root, ext = os.path.splitext(name)
    if ext == "":
        ext = ".png"
    primary_path = os.path.join(OUTPUT_DIR, root + ext)
    plt.savefig(primary_path, dpi=300, bbox_inches="tight")
    print("Saved:", primary_path)

    # Save vector formats for manuscript assembly.
    for vector_ext in (".pdf", ".svg"):
        vector_path = os.path.join(OUTPUT_DIR, root + vector_ext)
        if os.path.abspath(vector_path) == os.path.abspath(primary_path):
            continue
        try:
            plt.savefig(vector_path, bbox_inches="tight")
            print("Saved:", vector_path)
        except Exception as exc:
            print(f"Could not save {vector_path}: {exc}")
    plt.close()


def make_binary_sequence_data(
    N: int,
    S: int,
    D: int,
    p: float = 0.1,
    device=device,
):
    return torch.bernoulli(torch.full((N, S, D), p, device=device))


def nan_metrics(prefix: str, keys: List[str]) -> Dict[str, float]:
    return {f"{prefix}{k}": float("nan") for k in keys}

# ============================================================
# 2. Surrogate spike function
# ============================================================

class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Fast-sigmoid-like surrogate gradient.
        grad = 1.0 / (1.0 + 25.0 * torch.abs(x)) ** 2
        return grad_output * grad


spike_fn = SpikeFn.apply

# ============================================================
# 3. nLIF / LIF layer with optional lateral inhibitory weights
# ============================================================

class SpikingLayer(nn.Module):
    """
    Feedforward spiking layer with optional same-layer lateral inhibition.

    Supports:
        - nLIF: leak_beta = 1.0
        - LIF:  leak_beta < 1.0
        - positive feedforward excitatory weights
        - optional actual negative lateral inhibitory weights
        - optional top-k hard WTA cap after inhibition
        - optional refractory period
        - optional divisive current normalization
        - reset after spike

    Notes for the paper:
        Baseline theorem-aligned SNNs use positive feedforward weights. WTA
        variants intentionally add negative lateral inhibitory weights as an
        empirical circuit-level control. Therefore, the WTA variants should be
        described as an empirical extension/proxy for inhibitory competition,
        not as part of the positive-weight theorem class.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        threshold: float = 1.0,
        weight_scale: float = 0.15,
        leak_beta: float = 1.0,
        refractory_steps: int = 0,
        normalize_current: bool = False,
        topk: Optional[int] = None,
        reset_mode: str = "zero",
        positive_weights: bool = True,
        use_lateral_inhibition: bool = False,
        inhibition_strength: float = 0.0,
        learn_inhibition: bool = False,
        normalize_inhibition_by_fan_in: bool = False,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.threshold = threshold
        self.weight_scale = weight_scale
        self.leak_beta = leak_beta
        self.refractory_steps = refractory_steps
        self.normalize_current = normalize_current
        self.topk = topk
        self.reset_mode = reset_mode
        self.positive_weights = positive_weights
        self.use_lateral_inhibition = use_lateral_inhibition
        self.inhibition_strength = inhibition_strength
        self.learn_inhibition = learn_inhibition
        self.normalize_inhibition_by_fan_in = normalize_inhibition_by_fan_in

        self.weight_raw = nn.Parameter(torch.randn(out_dim, in_dim) * 0.25)

        # Off-diagonal mask for lateral inhibition: no self-inhibition.
        offdiag = torch.ones(out_dim, out_dim) - torch.eye(out_dim)
        self.register_buffer("inh_offdiag_mask", offdiag)

        if self.use_lateral_inhibition:
            # Effective inhibitory weights are negative by construction:
            # W_inh = -strength * softplus(raw) * offdiag_mask.
            # Initial raw=0 gives a modest uniform inhibitory conductance.
            init = torch.zeros(out_dim, out_dim)
            if self.learn_inhibition:
                self.inh_weight_raw = nn.Parameter(init)
            else:
                self.register_buffer("inh_weight_raw", init)
        else:
            self.inh_weight_raw = None

    def effective_weight(self):
        if self.positive_weights:
            return self.weight_scale * F.softplus(self.weight_raw)
        return self.weight_scale * self.weight_raw

    def effective_inhibitory_weight(self):
        if not self.use_lateral_inhibition or self.inh_weight_raw is None:
            return None
        W_inh = -self.inhibition_strength * F.softplus(self.inh_weight_raw) * self.inh_offdiag_mask
        if self.normalize_inhibition_by_fan_in:
            W_inh = W_inh / math.sqrt(max(self.out_dim - 1, 1))
        return W_inh

    def forward(self, x_seq, return_stats: bool = False):
        B, T, _ = x_seq.shape
        W_exc = self.effective_weight()
        W_inh = self.effective_inhibitory_weight()

        mem = torch.zeros(B, self.out_dim, device=x_seq.device)
        refractory_count = torch.zeros(B, self.out_dim, device=x_seq.device)

        spike_record = []
        lateral_inhibition_record = []

        for t in range(T):
            cur = x_seq[:, t] @ W_exc.T

            if self.normalize_current:
                denom = torch.sqrt(1e-4 + cur.pow(2).mean(dim=1, keepdim=True))
                cur = cur / denom

            # nLIF if beta = 1.0; LIF if beta < 1.0.
            mem_candidate = self.leak_beta * mem + cur

            eligible = (refractory_count <= 0).float()
            candidate_spk = spike_fn(mem_candidate - self.threshold) * eligible

            if W_inh is not None:
                # Actual negative lateral inhibitory current from candidate spikes.
                # Shape: [B, out_dim]. W_inh[post, pre] <= 0.
                lateral_inh = candidate_spk @ W_inh.T
                mem_competed = mem_candidate + lateral_inh
            else:
                lateral_inh = torch.zeros_like(mem_candidate)
                mem_competed = mem_candidate

            raw_spk = spike_fn(mem_competed - self.threshold) * eligible

            # Optional hard WTA cap. This is a discrete causal-set cap layered on
            # top of the actual inhibitory current, useful for theorem-aligned
            # causal-cap controls.
            if self.topk is not None:
                k = min(self.topk, self.out_dim)
                _, idx = torch.topk(mem_competed, k=k, dim=1)
                mask = torch.zeros_like(raw_spk)
                mask.scatter_(1, idx, 1.0)
                spk = raw_spk * mask
            else:
                spk = raw_spk

            if self.reset_mode == "zero":
                mem = mem_competed * (1.0 - spk)
            elif self.reset_mode == "subtract":
                mem = mem_competed - spk * self.threshold
            else:
                raise ValueError("reset_mode must be 'zero' or 'subtract'.")

            if self.refractory_steps > 0:
                refractory_count = torch.clamp(refractory_count - 1, min=0)
                refractory_count = torch.where(
                    spk > 0,
                    torch.full_like(refractory_count, float(self.refractory_steps)),
                    refractory_count,
                )

            spike_record.append(spk)
            lateral_inhibition_record.append(lateral_inh)

        spikes = torch.stack(spike_record, dim=1)

        if return_stats:
            lateral_inh_seq = torch.stack(lateral_inhibition_record, dim=1)
            stats = {
                "total_spikes_per_sample": spikes.sum(dim=(1, 2)).detach(),
                "mean_spikes_per_neuron": spikes.sum(dim=1).mean(dim=1).detach(),
                "fraction_active_time_bins": (spikes.sum(dim=2) > 0).float().mean(dim=1).detach(),
                "mean_lateral_inhibition": lateral_inh_seq.abs().mean().detach(),
            }
            return spikes, stats

        return spikes

# ============================================================
# 4. Variant definitions
# ============================================================

@dataclass
class VariantConfig:
    model_type: str = "snn"  # "snn", "gru", "transformer"
    leak_beta: float = 1.0
    refractory_steps: int = 0
    normalize_current: bool = False
    topk: Optional[int] = None
    spike_reg_lambda: float = 0.0
    use_lateral_inhibition: bool = False
    inhibition_strength: float = 0.0
    learn_inhibition: bool = False
    normalize_inhibition_by_fan_in: bool = False
    description: str = ""


VARIANT_CONFIGS: Dict[str, VariantConfig] = {
    "nlif_baseline": VariantConfig(
        model_type="snn",
        leak_beta=1.0,
        refractory_steps=0,
        normalize_current=False,
        topk=None,
        spike_reg_lambda=0.0,
        use_lateral_inhibition=False,
        description="Baseline feedforward nLIF, positive feedforward weights",
    ),
    "nlif_lateral_inh_only": VariantConfig(
        model_type="snn",
        leak_beta=1.0,
        refractory_steps=0,
        normalize_current=False,
        topk=None,
        spike_reg_lambda=0.0,
        use_lateral_inhibition=True,
        inhibition_strength=0.040,
        learn_inhibition=True,
        normalize_inhibition_by_fan_in=False,
        description="nLIF with learned negative lateral inhibition only; no hard top-k cap",
    ),
    "nlif_wta16": VariantConfig(
        model_type="snn",
        leak_beta=1.0,
        refractory_steps=0,
        normalize_current=False,
        topk=16,
        spike_reg_lambda=0.0,
        use_lateral_inhibition=True,
        inhibition_strength=0.040,
        learn_inhibition=True,
        normalize_inhibition_by_fan_in=False,
        description="nLIF with learned negative lateral inhibition plus hard WTA cap, k=16",
    ),
    "nlif_wta16_ref": VariantConfig(
        model_type="snn",
        leak_beta=1.0,
        refractory_steps=2,
        normalize_current=False,
        topk=16,
        spike_reg_lambda=0.0,
        use_lateral_inhibition=True,
        inhibition_strength=0.050,
        learn_inhibition=True,
        normalize_inhibition_by_fan_in=False,
        description="nLIF with learned negative lateral inhibition, hard WTA cap k=16, and refractory constraint",
    ),
    "lif_baseline": VariantConfig(
        model_type="snn",
        leak_beta=0.90,
        refractory_steps=0,
        normalize_current=False,
        topk=None,
        spike_reg_lambda=0.0,
        use_lateral_inhibition=False,
        description="LIF temporal-forgetting control",
    ),
    "gru": VariantConfig(
        model_type="gru",
        description="GRU recurrent ANN baseline",
    ),
    "transformer": VariantConfig(
        model_type="transformer",
        description="Small Transformer encoder baseline",
    ),
}

# Safety check: keep the run list, display labels, and config table locked to the same seven variants.
# Keep variant definitions in the canonical seven-architecture order.
# Presets may run all seven variants or a computationally focused subset.
assert list(VARIANT_CONFIGS.keys()) == VARIANT_ORDER, "Variant configs must match the canonical seven manuscript-facing variants."
VARIANTS_TO_RUN = [v for v in VARIANT_ORDER if v in VARIANTS_TO_RUN]

# Save the exact model/variant table for the supplement.
variant_table = pd.DataFrame([
    {
        "variant": name,
        "display_name": VARIANT_DISPLAY_NAMES[name],
        "hidden_dim": HIDDEN_BY_VARIANT[name],
        **asdict(cfg),
    }
    for name, cfg in VARIANT_CONFIGS.items()
])
variant_table_path = os.path.join(OUTPUT_DIR, "variant_config_table.csv")
variant_table.to_csv(variant_table_path, index=False)
print("Saved variant config table:", variant_table_path)

# ============================================================
# 5. Models
# ============================================================

class SNNSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_classes: int,
        cfg: VariantConfig,
    ):
        super().__init__()

        self.is_snn = True
        self.cfg = cfg
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        dims = [input_dim] + [hidden_dim] * n_layers

        self.layers = nn.ModuleList([
            SpikingLayer(
                dims[i],
                dims[i + 1],
                threshold=SNN_THRESHOLD,
                weight_scale=SNN_WEIGHT_SCALE,
                leak_beta=cfg.leak_beta,
                refractory_steps=cfg.refractory_steps,
                normalize_current=cfg.normalize_current,
                topk=cfg.topk,
                reset_mode="zero",
                positive_weights=True,
                use_lateral_inhibition=cfg.use_lateral_inhibition,
                inhibition_strength=cfg.inhibition_strength,
                learn_inhibition=cfg.learn_inhibition,
                normalize_inhibition_by_fan_in=cfg.normalize_inhibition_by_fan_in,
            )
            for i in range(n_layers)
        ])

        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_seq, return_stats: bool = False):
        out = x_seq

        total_hidden_spikes = None
        active_time_fraction_list = []
        inhibition_list = []

        for layer in self.layers:
            if return_stats:
                out, layer_stats = layer(out, return_stats=True)
                if total_hidden_spikes is None:
                    total_hidden_spikes = layer_stats["total_spikes_per_sample"]
                else:
                    total_hidden_spikes = total_hidden_spikes + layer_stats["total_spikes_per_sample"]
                active_time_fraction_list.append(layer_stats["fraction_active_time_bins"])
                inhibition_list.append(layer_stats["mean_lateral_inhibition"])
            else:
                out = layer(out)

        # Spike-count decoder. This is deliberately simple because the empirical
        # section is auxiliary to the theory, not a SOTA sequence-modeling claim.
        features = out.sum(dim=1)
        logits = self.readout(features)

        if return_stats:
            if len(active_time_fraction_list) > 0:
                active_time_fraction = torch.stack(active_time_fraction_list, dim=0).mean(dim=0)
            else:
                active_time_fraction = torch.full((x_seq.shape[0],), float("nan"), device=x_seq.device)

            if len(inhibition_list) > 0:
                mean_lateral_inhibition = torch.stack(inhibition_list).mean().item()
            else:
                mean_lateral_inhibition = float("nan")

            norm_denom = max(1, x_seq.shape[1] * self.hidden_dim * self.n_layers)
            hidden_spike_density_per_sample = total_hidden_spikes / float(norm_denom)

            stats = {
                "total_hidden_spikes_per_sample": total_hidden_spikes,
                "hidden_spike_density_per_sample": hidden_spike_density_per_sample,
                "mean_total_hidden_spikes": total_hidden_spikes.mean().item(),
                "mean_hidden_spike_density": hidden_spike_density_per_sample.mean().item(),
                "active_time_fraction": active_time_fraction.mean().item(),
                "mean_lateral_inhibition": mean_lateral_inhibition,
            }
            return logits, stats

        return logits


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.is_snn = False
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_seq, return_stats: bool = False):
        out, h = self.gru(x_seq)
        logits = self.readout(h[-1])

        if return_stats:
            stats = {
                "total_hidden_spikes_per_sample": torch.full((x_seq.shape[0],), float("nan"), device=x_seq.device),
                "hidden_spike_density_per_sample": torch.full((x_seq.shape[0],), float("nan"), device=x_seq.device),
                "mean_total_hidden_spikes": float("nan"),
                "mean_hidden_spike_density": float("nan"),
                "active_time_fraction": float("nan"),
                "mean_lateral_inhibition": float("nan"),
            }
            return logits, stats

        return logits


class TinyTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        max_len: int,
        n_heads: int = 4,
        n_layers: int = 1,
    ):
        super().__init__()
        self.is_snn = False
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, hidden_dim) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,  # keep regularization simple/fair for this controlled synthetic comparison
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.readout = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_seq, return_stats: bool = False):
        B, S, _ = x_seq.shape
        x = self.input_proj(x_seq)

        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : S + 1, :]

        h = self.encoder(x)
        logits = self.readout(h[:, 0, :])

        if return_stats:
            stats = {
                "total_hidden_spikes_per_sample": torch.full((x_seq.shape[0],), float("nan"), device=x_seq.device),
                "hidden_spike_density_per_sample": torch.full((x_seq.shape[0],), float("nan"), device=x_seq.device),
                "mean_total_hidden_spikes": float("nan"),
                "mean_hidden_spike_density": float("nan"),
                "active_time_fraction": float("nan"),
                "mean_lateral_inhibition": float("nan"),
            }
            return logits, stats

        return logits


def make_model(variant: str, max_len: int):
    cfg = VARIANT_CONFIGS[variant]
    hidden_dim = HIDDEN_BY_VARIANT.get(variant, HIDDEN)

    if cfg.model_type == "snn":
        model = SNNSequenceClassifier(
            input_dim=D_IN,
            hidden_dim=hidden_dim,
            n_layers=SNN_LAYERS,
            n_classes=N_CLASSES,
            cfg=cfg,
        )
    elif cfg.model_type == "gru":
        model = GRUClassifier(
            input_dim=D_IN,
            hidden_dim=hidden_dim,
            n_classes=N_CLASSES,
        )
    elif cfg.model_type == "transformer":
        # hidden_dim=32 remains divisible by four heads and keeps this baseline parameter-matched.
        model = TinyTransformerClassifier(
            input_dim=D_IN,
            hidden_dim=hidden_dim,
            n_classes=N_CLASSES,
            max_len=max_len,
            n_heads=4,
            n_layers=1,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    return model.to(device)


def save_parameter_table(max_len: int):
    rows = []
    for variant in VARIANTS_TO_RUN:
        set_seed(make_condition_seed(0, S=max_len, M=0, variant=variant, role="model", task="teacher_student"))
        model = make_model(variant, max_len=max_len)
        cfg = VARIANT_CONFIGS[variant]
        rows.append({
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "model_type": cfg.model_type,
            "hidden_dim": HIDDEN_BY_VARIANT.get(variant, HIDDEN),
            "n_params": count_trainable_params(model),
            "leak_beta": cfg.leak_beta,
            "refractory_steps": cfg.refractory_steps,
            "normalize_current": cfg.normalize_current,
            "topk_hard_cap": cfg.topk,
            "use_lateral_inhibition": cfg.use_lateral_inhibition,
            "inhibition_strength": cfg.inhibition_strength,
            "learn_inhibition": cfg.learn_inhibition,
            "description": cfg.description,
        })
    table = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "parameter_table.csv")
    table.to_csv(path, index=False)
    print("Saved parameter table:", path)
    display(table)

save_parameter_table(max_len=max(SEQ_LENGTHS))

# ============================================================
# 6. Teacher and datasets
# ============================================================

class TeacherSNN(nn.Module):
    """
    Fixed nLIF teacher for the theorem-aligned teacher-student task.

    The teacher is intentionally simpler than the student so that all sequence
    lengths can eventually cross threshold with enough samples.
    """

    def __init__(
        self,
        input_dim: int = D_IN,
        hidden_dim: int = TEACHER_HIDDEN,
        n_layers: int = TEACHER_LAYERS,
    ):
        super().__init__()

        dims = [input_dim] + [hidden_dim] * n_layers
        self.layers = nn.ModuleList([
            SpikingLayer(
                dims[i],
                dims[i + 1],
                threshold=TEACHER_THRESHOLD,
                weight_scale=TEACHER_WEIGHT_SCALE,
                leak_beta=1.0,
                refractory_steps=0,
                normalize_current=False,
                topk=None,
                reset_mode="zero",
                positive_weights=True,
                use_lateral_inhibition=False,
            )
            for i in range(n_layers)
        ])

        self.readout_vec = nn.Parameter(torch.randn(hidden_dim), requires_grad=False)
        self.bias = nn.Parameter(torch.randn(()) * 0.05, requires_grad=False)

        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def score(self, x_seq):
        out = x_seq
        for layer in self.layers:
            out = layer(out)
        features = out.sum(dim=1)
        score = features @ self.readout_vec + self.bias
        return score


@torch.no_grad()
def calibrate_teacher_threshold(teacher: TeacherSNN, S: int, D: int, N: int = 5000):
    x = make_binary_sequence_data(N, S, D, p=SPIKE_PROB, device=device)
    score = teacher.score(x)
    threshold = score.median().item()
    return threshold


@torch.no_grad()
def make_teacher_dataset(
    teacher: TeacherSNN,
    N: int,
    S: int,
    D: int,
    threshold: float,
):
    x = make_binary_sequence_data(N, S, D, p=SPIKE_PROB, device=device)
    score = teacher.score(x)
    y = (score > threshold).long()
    return x, y


def make_cue_recall_dataset(N: int, S: int, D: int):
    """
    Delayed cue-recall / long-range memory task.

    The first token contains a binary cue encoded in channels 0/1. The model must
    report that first-token cue at the final query token. Intermediate timesteps
    contain one-hot distractor bits in the same channels, so the task is not
    solved by simply counting channel activity. This makes sequence length a
    true delay variable while remaining simple enough for a theory supplement.

    Channel convention:
        channel 0/1: cue or distractor bit
        channel 2: final query marker
        channel 3: start/cue marker
        channels >= 4: sparse Bernoulli background spikes
    """
    assert D >= 4, "cue_recall requires D_IN >= 4"

    x = torch.bernoulli(torch.full((N, S, D), SPIKE_PROB, device=device))
    y = torch.randint(0, 2, (N,), device=device)

    # Random distractor stream in the same channels as the cue.
    distractor = torch.randint(0, 2, (N, S), device=device)
    x[:, :, 0] = (distractor == 0).float()
    x[:, :, 1] = (distractor == 1).float()

    # True cue at t=0.
    x[:, 0, 0] = (y == 0).float()
    x[:, 0, 1] = (y == 1).float()

    # Explicit markers.
    x[:, :, 2] = 0.0
    x[:, :, 3] = 0.0
    x[:, -1, 2] = 1.0   # query marker
    x[:, 0, 3] = 1.0    # cue/start marker

    return x, y


def make_dataset(
    N: int,
    S: int,
    D: int,
    task: str,
    teacher: Optional[TeacherSNN] = None,
    threshold: Optional[float] = None,
):
    if task == "teacher_student":
        assert teacher is not None
        assert threshold is not None
        return make_teacher_dataset(teacher, N, S, D, threshold)
    if task == "cue_recall":
        return make_cue_recall_dataset(N, S, D)
    raise ValueError(f"Unknown task: {task}")

# ============================================================
# 7. Training / evaluation / metrics
# ============================================================

def _evaluate_loss_acc(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = EVAL_BATCH_SIZE):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            xb = x[start:end]
            yb = y[start:end]
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            pred = logits.argmax(dim=1)
            total_loss += loss.item()
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_model(
    model: nn.Module,
    variant: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    x_val_fixed: Optional[torch.Tensor] = None,
    y_val_fixed: Optional[torch.Tensor] = None,
):
    """
    Train a model with optional validation-based early stopping.

    Early stopping reduces the risk that M*(S) reflects undertraining rather than
    sample availability. This still remains a controlled empirical sanity check,
    not an optimizer-independent PAC estimate.
    """
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    cfg = VARIANT_CONFIGS[variant]
    N = x_train.shape[0]

    validation_mode = "none"

    if USE_EARLY_STOPPING and x_val_fixed is not None and y_val_fixed is not None:
        # Preferred manuscript-facing mode: fixed validation set shared across all
        # M for this task/seed/S condition. M remains the number of actual
        # fitting examples, while validation examples are held out separately.
        x_tr, y_tr = x_train, y_train
        x_val, y_val = x_val_fixed, y_val_fixed
        validation_mode = "fixed_external"
    elif USE_EARLY_STOPPING and N >= 256:
        # Fallback mode for standalone reuse if no external validation set is
        # passed. This branch is deterministic because the caller sets train_seed
        # immediately before training.
        n_val = max(64, int(round(VALIDATION_FRACTION * N)))
        n_val = min(n_val, max(64, N // 3))
        perm0 = torch.randperm(N, device=device)
        val_idx = perm0[:n_val]
        tr_idx = perm0[n_val:]
        x_val, y_val = x_train[val_idx], y_train[val_idx]
        x_tr, y_tr = x_train[tr_idx], y_train[tr_idx]
        validation_mode = "internal_split"
    else:
        x_tr, y_tr = x_train, y_train
        x_val, y_val = None, None

    best_val_loss = float("inf")
    best_val_acc = float("nan")
    best_state = None
    patience_left = EARLY_STOPPING_PATIENCE
    epochs_trained = 0
    updates = 0

    for epoch in range(epochs):
        model.train()
        Ntr = x_tr.shape[0]
        perm = torch.randperm(Ntr, device=device)
        x_train_epoch = x_tr[perm]
        y_train_epoch = y_tr[perm]

        for start in range(0, Ntr, batch_size):
            end = min(start + batch_size, Ntr)
            xb = x_train_epoch[start:end]
            yb = y_train_epoch[start:end]

            if getattr(model, "is_snn", False):
                logits, stats = model(xb, return_stats=True)
                loss = F.cross_entropy(logits, yb)

                # Optional spike regularization for causal-cap style experiments.
                if cfg.spike_reg_lambda > 0:
                    mean_spikes = stats["total_hidden_spikes_per_sample"].mean()
                    norm_factor = xb.shape[1] * getattr(model, "hidden_dim", HIDDEN) * getattr(model, "n_layers", SNN_LAYERS)
                    loss = loss + cfg.spike_reg_lambda * mean_spikes / max(float(norm_factor), 1.0)
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            updates += 1

        epochs_trained = epoch + 1

        if x_val is not None:
            val_loss, val_acc = _evaluate_loss_acc(model, x_val, y_val)
            if val_loss < best_val_loss - MIN_VAL_IMPROVEMENT:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = EARLY_STOPPING_PATIENCE
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    elif x_val is not None:
        best_val_loss, best_val_acc = _evaluate_loss_acc(model, x_val, y_val)

    train_info = {
        "epochs_trained": epochs_trained,
        "optimizer_updates": updates,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "validation_mode": validation_mode,
        "n_train_effective": int(x_tr.shape[0]),
        "n_val": int(x_val.shape[0]) if x_val is not None else 0,
    }
    return model, train_info


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = EVAL_BATCH_SIZE,
):
    model.eval()

    correct = 0
    total = 0
    losses = []

    spike_values = []
    spike_density_values = []
    active_fractions = []
    inhibition_values = []

    for start in range(0, x_test.shape[0], batch_size):
        end = min(start + batch_size, x_test.shape[0])
        xb = x_test[start:end]
        yb = y_test[start:end]

        logits, stats = model(xb, return_stats=True)
        loss = F.cross_entropy(logits, yb)

        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        losses.append(loss.item())

        if getattr(model, "is_snn", False):
            spike_values.append(stats["total_hidden_spikes_per_sample"].detach().cpu())
            spike_density_values.append(stats["hidden_spike_density_per_sample"].detach().cpu())
            active_fractions.append(stats["active_time_fraction"])
            inhibition_values.append(stats.get("mean_lateral_inhibition", float("nan")))

    acc = correct / total
    mean_loss = float(np.mean(losses))

    if getattr(model, "is_snn", False) and len(spike_values) > 0:
        participation = torch.cat(spike_values).float().mean().item()
        participation_density = torch.cat(spike_density_values).float().mean().item()
        active_time_fraction = float(np.mean(active_fractions))
        mean_lateral_inhibition = float(np.nanmean(inhibition_values))
    else:
        participation = np.nan
        participation_density = np.nan
        active_time_fraction = np.nan
        mean_lateral_inhibition = np.nan

    return {
        "test_acc": acc,
        "test_loss": mean_loss,
        "spike_participation_proxy": participation,
        "spike_participation_density": participation_density,
        "active_time_fraction": active_time_fraction,
        "mean_lateral_inhibition": mean_lateral_inhibition,
    }



@torch.inference_mode()
def token_ablation_causal_participation(
    model: nn.Module,
    x: torch.Tensor,
    max_samples: int = SENS_MAX_SAMPLES,
    effect_threshold: float = CAUSAL_EFFECT_THRESHOLD,
):
    """
    Supplementary intervention-based causal-time participation proxy.

    For each time bin t, zero out x[:, t, :] and measure output-probability
    change. This is intentionally limited to cue_recall at largest M in the
    focused validation run.
    """
    model.eval()

    x = x[:max_samples]
    B, S, D = x.shape

    base_logits = model(x)
    base_probs = F.softmax(base_logits, dim=1)

    effects = []
    for t in range(S):
        x_abl = x.clone()
        x_abl[:, t, :] = 0.0

        abl_logits = model(x_abl)
        abl_probs = F.softmax(abl_logits, dim=1)

        effect_t = torch.sum(torch.abs(abl_probs - base_probs), dim=1)
        effects.append(effect_t)

    effects = torch.stack(effects, dim=1)  # [B, S]
    active = effects > effect_threshold

    causal_time_count = active.float().sum(dim=1)
    causal_time_fraction = active.float().mean()

    return {
        "intervention_causal_time_count_mean": causal_time_count.mean().item(),
        "intervention_causal_time_count_p95": torch.quantile(causal_time_count, 0.95).item(),
        "intervention_causal_time_fraction": causal_time_fraction.item(),
        "intervention_effect_p50": torch.quantile(effects.reshape(-1), 0.50).item(),
        "intervention_effect_p95": torch.quantile(effects.reshape(-1), 0.95).item(),
        "intervention_effect_max_mean": effects.max(dim=1).values.mean().item(),
        "causal_effect_threshold": effect_threshold,
    }


def empty_causal_metrics():
    return {
        "intervention_causal_time_count_mean": float("nan"),
        "intervention_causal_time_count_p95": float("nan"),
        "intervention_causal_time_fraction": float("nan"),
        "intervention_effect_p50": float("nan"),
        "intervention_effect_p95": float("nan"),
        "intervention_effect_max_mean": float("nan"),
        "causal_effect_threshold": CAUSAL_EFFECT_THRESHOLD,
    }

# ============================================================
# 8. Main experiment
# ============================================================

CHECKPOINT_CSV = os.path.join(OUTPUT_DIR, "raw_results_checkpoint.csv")
FINAL_RAW_CSV = os.path.join(OUTPUT_DIR, "raw_results.csv")
CHECKPOINT_CONFIG_JSON = os.path.join(OUTPUT_DIR, "checkpoint_config_signature.json")

def _condition_key_from_values(task, seed, S, variant, M):
    return (str(task), int(seed), int(S), str(variant), int(M))

def _condition_key_from_row(row: dict):
    return _condition_key_from_values(row["task"], row["seed"], row["S"], row["variant"], row["M"])

def _current_checkpoint_signature():
    """Configuration fields that must match before checkpoint rows are reused."""
    return {
        "run_preset": RUN_PRESET,
        "tasks_to_run": TASKS_TO_RUN,
        "sequence_lengths": SEQ_LENGTHS,
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "variants_to_run": VARIANTS_TO_RUN,
        "hidden_by_variant": HIDDEN_BY_VARIANT,
        "d_in": D_IN,
        "snn_layers": SNN_LAYERS,
        "n_classes": N_CLASSES,
        "spike_prob": SPIKE_PROB,
        "teacher_hidden": TEACHER_HIDDEN,
        "teacher_layers": TEACHER_LAYERS,
        "teacher_weight_scale": TEACHER_WEIGHT_SCALE,
        "teacher_threshold": TEACHER_THRESHOLD,
        "snn_weight_scale": SNN_WEIGHT_SCALE,
        "snn_threshold": SNN_THRESHOLD,
        "model_init_shared_across_M": MODEL_INIT_SHARED_ACROSS_M,
        "use_fixed_validation_set": USE_FIXED_VALIDATION_SET,
        "fixed_validation_n": FIXED_VALIDATION_N,
        "test_n": TEST_N,
        "calib_n": CALIB_N,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    }

def _checkpoint_signature_matches() -> bool:
    if not os.path.exists(CHECKPOINT_CSV):
        return False
    if not os.path.exists(CHECKPOINT_CONFIG_JSON):
        print(
            "Checkpoint CSV exists but no checkpoint_config_signature.json was found. "
            "Ignoring checkpoint for safety. Set SNN_ALLOW_UNSAFE_RESUME=1 to override."
        )
        return env_flag("SNN_ALLOW_UNSAFE_RESUME", False)
    try:
        with open(CHECKPOINT_CONFIG_JSON, "r", encoding="utf-8") as f:
            old = json.load(f)
    except Exception as exc:
        print(f"Could not read checkpoint signature ({exc}). Ignoring checkpoint for safety.")
        return False
    new = _current_checkpoint_signature()
    if old != new:
        print(
            "Checkpoint configuration does not match current run. Ignoring checkpoint for safety. "
            "Use a new --output-dir or delete the old output directory for a fresh run."
        )
        return False
    return True

def _write_checkpoint_signature():
    with open(CHECKPOINT_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(_current_checkpoint_signature(), f, indent=2, sort_keys=True)

def _load_checkpoint_rows():
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_CSV) and _checkpoint_signature_matches():
        try:
            ckpt = pd.read_csv(CHECKPOINT_CSV)
            if len(ckpt) > 0:
                key_cols = ["task", "seed", "S", "variant", "M"]
                before = len(ckpt)
                ckpt = ckpt.drop_duplicates(subset=key_cols, keep="last")
                if len(ckpt) != before:
                    print(f"Dropped {before - len(ckpt)} duplicate checkpoint rows; kept the last occurrence.")
            rows = ckpt.to_dict("records")
            completed = {_condition_key_from_row(r) for r in rows}
            print(f"Loaded checkpoint with {len(rows)} completed rows: {CHECKPOINT_CSV}")
            return rows, completed
        except Exception as exc:
            print(f"Could not load checkpoint {CHECKPOINT_CSV}: {exc}. Starting fresh.")
    _write_checkpoint_signature()
    return [], set()

def _append_checkpoint_row(row: dict):
    if not CHECKPOINT_EVERY_ROW:
        return
    header = not os.path.exists(CHECKPOINT_CSV)
    pd.DataFrame([row]).to_csv(CHECKPOINT_CSV, mode="a", header=header, index=False)

def should_compute_token_ablation(task: str, M: int, max_train_size: int) -> bool:
    return (
        COMPUTE_TOKEN_ABLATION
        and task in TOKEN_ABLATION_TASKS
        and (not TOKEN_ABLATION_ONLY_MAX_M or M == max_train_size)
    )


def run_experiment():
    results, completed_keys = _load_checkpoint_rows()
    max_train_size = max(TRAIN_SIZES)
    max_len = max(SEQ_LENGTHS)

    if USE_EARLY_STOPPING and USE_FIXED_VALIDATION_SET:
        fixed_validation_n = (
            int(FIXED_VALIDATION_N)
            if FIXED_VALIDATION_N is not None
            else max(512, int(round(VALIDATION_FRACTION * max_train_size)))
        )
    else:
        fixed_validation_n = 0

    for task in TASKS_TO_RUN:
        print("\n" + "#" * 80)
        print(f"TASK: {task}")
        print("#" * 80)

        for seed in SEEDS:
            set_seed(make_condition_seed(seed, S=0, M=0, variant="nlif_baseline", role="teacher", task=task))

            teacher = None
            if task == "teacher_student":
                teacher = TeacherSNN().to(device)
                teacher.eval()

            for S in SEQ_LENGTHS:
                print("\n" + "=" * 70)
                print(f"Task {task} | Seed {seed} | Sequence length S={S}")
                print("=" * 70)

                threshold = None
                if task == "teacher_student":
                    set_seed(make_condition_seed(seed, S=S, M=0, variant="nlif_baseline", role="data", task=task))
                    threshold = calibrate_teacher_threshold(teacher, S, D_IN, N=CALIB_N)

                # For each task/seed/S, use one fixed test set, one fixed validation set,
                # and one nested training pool. Larger M sees a superset of smaller M,
                # and the validation distribution is unchanged across M.
                set_seed(make_condition_seed(seed, S=S, M=max_train_size, variant="nlif_baseline", role="data", task=task))
                x_test, y_test = make_dataset(TEST_N, S, D_IN, task=task, teacher=teacher, threshold=threshold)

                trainval_N = max_train_size + fixed_validation_n
                x_trainval_pool, y_trainval_pool = make_dataset(
                    trainval_N, S, D_IN, task=task, teacher=teacher, threshold=threshold
                )

                if fixed_validation_n > 0:
                    x_val_fixed = x_trainval_pool[:fixed_validation_n]
                    y_val_fixed = y_trainval_pool[:fixed_validation_n]
                    x_train_pool = x_trainval_pool[fixed_validation_n:]
                    y_train_pool = y_trainval_pool[fixed_validation_n:]
                else:
                    x_val_fixed = None
                    y_val_fixed = None
                    x_train_pool = x_trainval_pool
                    y_train_pool = y_trainval_pool

                pos_frac = y_test.float().mean().item()
                print(f"Class-1 fraction in test set: {pos_frac:.3f}")
                if fixed_validation_n > 0:
                    print(f"Fixed validation N: {fixed_validation_n}; nested training pool N: {x_train_pool.shape[0]}")

                for variant in VARIANTS_TO_RUN:
                    cfg = VARIANT_CONFIGS[variant]
                    print(f"\nVariant: {VARIANT_DISPLAY_NAMES[variant]} | {cfg.description}")

                    for M in tqdm(TRAIN_SIZES, desc=f"{task}, S={S}, {VARIANT_DISPLAY_NAMES[variant]}", leave=False, disable=DISABLE_TQDM):
                        cond_key = _condition_key_from_values(task, seed, S, variant, M)
                        if cond_key in completed_keys:
                            print(f"Skipping completed: task={task} seed={seed} S={S} variant={variant} M={M}")
                            continue

                        model_seed_M = 0 if MODEL_INIT_SHARED_ACROSS_M else M
                        model_seed = make_condition_seed(seed, S=S, M=model_seed_M, variant=variant, role="model", task=task)
                        train_seed = make_condition_seed(seed, S=S, M=M, variant=variant, role="train", task=task)
                        metric_seed = make_condition_seed(seed, S=S, M=M, variant=variant, role="metrics", task=task)

                        set_seed(model_seed)
                        model = make_model(variant, max_len=max_len)
                        n_params = count_trainable_params(model)

                        x_train = x_train_pool[:M]
                        y_train = y_train_pool[:M]

                        set_seed(train_seed)
                        model, train_info = train_one_model(
                            model,
                            variant,
                            x_train,
                            y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            lr=LR,
                            x_val_fixed=x_val_fixed,
                            y_val_fixed=y_val_fixed,
                        )

                        eval_metrics = evaluate_model(model, x_test, y_test)

                        if should_compute_token_ablation(task, M, max_train_size):
                            set_seed(metric_seed)
                            causal_metrics = token_ablation_causal_participation(model, x_test)
                        else:
                            causal_metrics = empty_causal_metrics()

                        row = {
                            "task": task,
                            "preset": RUN_PRESET,
                            "seed": seed,
                            "variant": variant,
                            "display_name": VARIANT_DISPLAY_NAMES[variant],
                            "description": cfg.description,
                            "model_type": cfg.model_type,
                            "hidden_dim": HIDDEN_BY_VARIANT.get(variant, HIDDEN),
                            "S": S,
                            "M": M,
                            "epochs_max": EPOCHS,
                            "epochs_trained": train_info["epochs_trained"],
                            "optimizer_updates": train_info["optimizer_updates"],
                            "best_val_loss": train_info["best_val_loss"],
                            "best_val_acc": train_info["best_val_acc"],
                            "validation_mode": train_info["validation_mode"],
                            "n_train_effective": train_info["n_train_effective"],
                            "n_val": train_info["n_val"],
                            "n_params": n_params,
                            "leak_beta": cfg.leak_beta,
                            "refractory_steps": cfg.refractory_steps,
                            "normalize_current": cfg.normalize_current,
                            "topk_hard_cap": cfg.topk,
                            "use_lateral_inhibition": cfg.use_lateral_inhibition,
                            "inhibition_strength": cfg.inhibition_strength,
                            "learn_inhibition": cfg.learn_inhibition,
                            "model_seed": model_seed,
                            "train_seed": train_seed,
                            "metric_seed": metric_seed,
                            **eval_metrics,
                            **causal_metrics,
                        }
                        results.append(row)
                        completed_keys.add(cond_key)
                        _append_checkpoint_row(row)

                        print(
                            f"M={M:6d} | "
                            f"acc={eval_metrics['test_acc']:.3f} | "
                            f"val={train_info['best_val_acc']:.3f} | "
                            f"spikes={eval_metrics['spike_participation_proxy']:.1f} | "
                            f"spike_density={eval_metrics['spike_participation_density']:.4f} | "
                            f"active={eval_metrics['active_time_fraction']:.3f} | "
                            f"inh={eval_metrics['mean_lateral_inhibition']:.3f} | "
                            f"causalT={causal_metrics['intervention_causal_time_count_mean']:.1f}"
                        )

    return pd.DataFrame(results)


df = run_experiment()

raw_csv = FINAL_RAW_CSV
df.to_csv(raw_csv, index=False)
print("\nSaved raw results:", raw_csv)
display(df.head())


def expected_condition_table():
    return pd.DataFrame([
        {"task": task, "seed": seed, "S": S, "variant": variant, "M": M}
        for task in TASKS_TO_RUN
        for seed in SEEDS
        for S in SEQ_LENGTHS
        for variant in VARIANTS_TO_RUN
        for M in TRAIN_SIZES
    ])

def save_completeness_report(df_in: pd.DataFrame):
    expected = expected_condition_table()
    actual = df_in[["task", "seed", "S", "variant", "M"]].drop_duplicates() if len(df_in) else expected.iloc[0:0]
    merged = expected.merge(actual, on=["task", "seed", "S", "variant", "M"], how="left", indicator=True)
    missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
    path = os.path.join(OUTPUT_DIR, "missing_conditions.csv")
    missing.to_csv(path, index=False)
    print(f"Completeness: {len(actual)}/{len(expected)} conditions completed. Missing: {len(missing)}")
    if len(missing) > 0:
        print(f"Saved missing-condition report: {path}")
    return missing

missing_conditions = save_completeness_report(df)

# ============================================================
# 9. Analysis: M*(S), censoring, paired statistics, slopes
# ============================================================

def compute_mstar(df: pd.DataFrame, threshold: float, selection_metric_col: str = "best_val_acc"):
    """
    Compute validation-selected samples-to-threshold M*(S).

    M*(S) is defined as the smallest tested training-set size whose validation
    accuracy reaches the requested threshold. Held-out test accuracy is retained
    only as a final performance metric and is not used to select M*(S).

    If the model never reaches threshold within TRAIN_SIZES:
        - M_star_observed is NaN.
        - M_star_censored_as_max is max(TRAIN_SIZES).
        - right_censored is True.

    This matches the manuscript language: threshold crossing is selected on the
    fixed validation set, while held-out test accuracy is reserved for final
    performance summaries.
    """
    if selection_metric_col not in df.columns:
        raise KeyError(f"Missing selection metric column for M*(S): {selection_metric_col}")

    rows = []
    max_M = int(df["M"].max())

    for (task, variant, seed, S), g in df.groupby(["task", "variant", "seed", "S"]):
        g = g.sort_values("M")
        selection_acc = pd.to_numeric(g[selection_metric_col], errors="coerce")
        reached_rows = g[selection_acc >= threshold]
        reached = len(reached_rows) > 0

        if reached:
            M_obs = int(reached_rows.iloc[0]["M"])
        else:
            M_obs = np.nan

        rows.append({
            "threshold": threshold,
            "task": task,
            "seed": int(seed),
            "S": int(S),
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "selection_metric": selection_metric_col,
            "M_star_observed": M_obs,
            "M_star_censored_as_max": M_obs if reached else max_M,
            "reached": bool(reached),
            "right_censored": bool(not reached),
            "max_selection_acc": float(selection_acc.max()),
            "max_test_acc": float(pd.to_numeric(g["test_acc"], errors="coerce").max()) if "test_acc" in g.columns else np.nan,
            "max_M_tested": max_M,
        })

    return pd.DataFrame(rows)


all_mstar = pd.concat([compute_mstar(df, thr) for thr in ACCURACY_THRESHOLDS], ignore_index=True)
mstar_csv = os.path.join(OUTPUT_DIR, "mstar_samples_to_threshold.csv")
all_mstar.to_csv(mstar_csv, index=False)
print("Saved M* table:", mstar_csv)
display(all_mstar.head())


def bootstrap_ci(values: np.ndarray, stat_fn=np.mean, n_boot: int = STATS_BOOTSTRAPS, seed: int = STATS_RANDOM_SEED):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        v = float(stat_fn(values))
        return v, v

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(stat_fn(sample))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def signed_rank_p(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, 0):
        return np.nan
    if scipy_stats is None:
        return np.nan
    try:
        return float(scipy_stats.wilcoxon(values).pvalue)
    except Exception:
        return np.nan


def sign_flip_p(values: np.ndarray, n_perm: int = 20000, seed: int = STATS_RANDOM_SEED):
    """
    Paired nonparametric sign-flip test on the mean delta.
    This is useful with small n because it tests whether the paired effect is
    consistently positive/negative without distributional assumptions.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.allclose(values, 0):
        return np.nan
    obs = abs(np.mean(values))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(values)))
    null = np.abs((signs * values[None, :]).mean(axis=1))
    return float((np.sum(null >= obs) + 1.0) / (n_perm + 1.0))


def paired_delta_summary(values: np.ndarray):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ci_low, ci_high = bootstrap_ci(values)
    mean_delta = float(np.mean(values)) if len(values) else np.nan
    std_delta = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    # paired standardized effect, often called Cohen's dz
    cohen_dz = float(mean_delta / std_delta) if np.isfinite(std_delta) and std_delta > 0 else np.nan
    return {
        "n_pairs": int(len(values)),
        "mean_delta": mean_delta,
        "median_delta": float(np.median(values)) if len(values) else np.nan,
        "std_delta": std_delta,
        "cohen_dz": cohen_dz,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "signed_rank_p": signed_rank_p(values),
        "sign_flip_p": sign_flip_p(values),
    }


def add_bh_qvalues(table: pd.DataFrame, p_col: str = "signed_rank_p") -> pd.DataFrame:
    """Add Benjamini-Hochberg FDR q-values to statistical summary tables."""
    table = table.copy()
    if p_col not in table.columns:
        return table
    pvals = pd.to_numeric(table[p_col], errors="coerce").astype(float).values
    qvals = np.full_like(pvals, np.nan, dtype=float)
    finite = np.isfinite(pvals)
    if finite.sum() > 0:
        idx = np.where(finite)[0]
        order = idx[np.argsort(pvals[finite])]
        ranked_p = pvals[order]
        m = len(ranked_p)
        adjusted = ranked_p * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)
        qvals[order] = adjusted
    q_col = p_col.replace("_p", "_q_bh") if p_col.endswith("_p") else f"{p_col}_q_bh"
    table[q_col] = qvals
    return table


def summary_mean_sem(df_in: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    """Manuscript-friendly mean/SEM/count summary for selected metrics."""
    rows = []
    for keys, g in df_in.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for col in value_cols:
            vals = pd.to_numeric(g[col], errors="coerce").astype(float)
            vals = vals[np.isfinite(vals)]
            row[f"{col}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{col}_sem"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            row[f"{col}_n"] = int(len(vals))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_accuracy_vs_baseline(df: pd.DataFrame, longS_only: bool = False):
    """
    Paired accuracy differences versus nLIF baseline at largest M.
    Pairing keys: task, seed, S.
    """
    max_M = df["M"].max()
    d = df[df["M"] == max_M].copy()
    if longS_only:
        d = d[d["S"] >= 64].copy()

    base = d[d["variant"] == "nlif_baseline"][
        ["task", "seed", "S", "test_acc"]
    ].rename(columns={"test_acc": "baseline_acc"})

    merged = d.merge(base, on=["task", "seed", "S"], how="inner")
    merged["delta_acc_vs_nlif"] = merged["test_acc"] - merged["baseline_acc"]

    rows = []
    for (task, variant), g in merged.groupby(["task", "variant"]):
        if variant == "nlif_baseline":
            continue
        stats_row = paired_delta_summary(g["delta_acc_vs_nlif"].values)
        rows.append({
            "task": task,
            "comparison": f"{VARIANT_DISPLAY_NAMES[variant]} - nLIF baseline",
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "longS_only": bool(longS_only),
            **stats_row,
        })

    return pd.DataFrame(rows)


def accuracy_degradation_slopes(df: pd.DataFrame):
    """
    Fit test_acc = alpha + slope * log2(S) at largest M for each task/variant/seed.
    More negative slope means stronger sequence-length degradation.
    """
    max_M = df["M"].max()
    d = df[df["M"] == max_M].copy()
    rows = []

    for (task, variant, seed), g in d.groupby(["task", "variant", "seed"]):
        g = g.sort_values("S")
        if g["S"].nunique() < 2:
            continue
        x = np.log2(g["S"].values.astype(float))
        y = g["test_acc"].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({
            "task": task,
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "seed": int(seed),
            "accuracy_slope_per_log2S": float(slope),
            "accuracy_intercept": float(intercept),
        })

    slope_cols = [
        "task", "variant", "display_name", "seed",
        "accuracy_slope_per_log2S", "accuracy_intercept",
    ]
    stat_cols = [
        "task", "comparison", "variant", "display_name",
        "n_pairs", "mean_delta", "median_delta", "std_delta", "cohen_dz",
        "ci_low", "ci_high", "signed_rank_p", "sign_flip_p",
    ]
    slopes = pd.DataFrame(rows, columns=slope_cols)

    if slopes.empty or "nlif_baseline" not in set(slopes["variant"]):
        return slopes, pd.DataFrame(columns=stat_cols)

    base = slopes[slopes["variant"] == "nlif_baseline"][
        ["task", "seed", "accuracy_slope_per_log2S"]
    ].rename(columns={"accuracy_slope_per_log2S": "baseline_slope"})

    merged = slopes.merge(base, on=["task", "seed"], how="inner")
    merged["delta_slope_vs_nlif"] = merged["accuracy_slope_per_log2S"] - merged["baseline_slope"]

    stat_rows = []
    for (task, variant), g in merged.groupby(["task", "variant"]):
        if variant == "nlif_baseline":
            continue
        stats_row = paired_delta_summary(g["delta_slope_vs_nlif"].values)
        stat_rows.append({
            "task": task,
            "comparison": f"{VARIANT_DISPLAY_NAMES[variant]} slope - nLIF baseline slope",
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            **stats_row,
        })

    return slopes, pd.DataFrame(stat_rows, columns=stat_cols)


def right_censoring_summary(mstar: pd.DataFrame):
    rows = []
    for (threshold, task, variant), g in mstar.groupby(["threshold", "task", "variant"]):
        rows.append({
            "threshold": threshold,
            "task": task,
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "n_conditions": int(len(g)),
            "n_reached": int(g["reached"].sum()),
            "n_right_censored": int(g["right_censored"].sum()),
            "frac_right_censored": float(g["right_censored"].mean()),
            "median_Mstar_censored_as_max": float(g["M_star_censored_as_max"].median()),
            "median_max_selection_acc": float(g["max_selection_acc"].median()) if "max_selection_acc" in g.columns else np.nan,
            "median_max_test_acc": float(g["max_test_acc"].median()) if "max_test_acc" in g.columns else np.nan,
        })
    return pd.DataFrame(rows)


def paired_mstar_vs_baseline(mstar: pd.DataFrame, threshold: float = 0.75):
    """
    Paired log2 M* differences using censored-as-max values.
    Negative delta means the variant reaches threshold with fewer samples than nLIF baseline.
    This is descriptive when either condition is censored.
    """
    d = mstar[mstar["threshold"] == threshold].copy()
    base = d[d["variant"] == "nlif_baseline"][
        ["task", "seed", "S", "M_star_censored_as_max", "right_censored"]
    ].rename(columns={
        "M_star_censored_as_max": "baseline_Mstar",
        "right_censored": "baseline_right_censored",
    })

    merged = d.merge(base, on=["task", "seed", "S"], how="inner")
    merged["delta_log2_Mstar_vs_nlif"] = (
        np.log2(merged["M_star_censored_as_max"].astype(float))
        - np.log2(merged["baseline_Mstar"].astype(float))
    )

    rows = []
    for (task, variant), g in merged.groupby(["task", "variant"]):
        if variant == "nlif_baseline":
            continue
        stats_row = paired_delta_summary(g["delta_log2_Mstar_vs_nlif"].values)
        rows.append({
            "threshold": threshold,
            "task": task,
            "comparison": f"{VARIANT_DISPLAY_NAMES[variant]} log2 M* - nLIF baseline log2 M*",
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "note": "Uses censored-as-max values; descriptive when points are right-censored.",
            **stats_row,
        })

    return pd.DataFrame(rows)



def _fit_loglog_beta(x_log2: np.ndarray, y_log2: np.ndarray):
    """Simple OLS beta for log2 M* = alpha + beta log2 S."""
    x_log2 = np.asarray(x_log2, dtype=float)
    y_log2 = np.asarray(y_log2, dtype=float)
    ok = np.isfinite(x_log2) & np.isfinite(y_log2)
    x_log2 = x_log2[ok]
    y_log2 = y_log2[ok]
    if len(x_log2) < 2 or np.unique(x_log2).size < 2:
        return np.nan, np.nan, np.nan
    beta, alpha = np.polyfit(x_log2, y_log2, 1)
    y_hat = alpha + beta * x_log2
    ss_res = float(np.sum((y_log2 - y_hat) ** 2))
    ss_tot = float(np.sum((y_log2 - np.mean(y_log2)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(beta), float(alpha), float(r2)


def _tobit_right_censored_beta(x_log2: np.ndarray, y_log2: np.ndarray, censored: np.ndarray, censor_log2: float):
    """
    Tobit-style MLE for right-censored log2 M*.

    Observed points contribute Normal pdf log-likelihood.
    Right-censored points contribute log P(Y >= censor_log2).
    This gives a more principled finite-range beta estimate than simply setting
    censored values equal to M_max. It is still descriptive and protocol-dependent.
    """
    x = np.asarray(x_log2, dtype=float)
    y = np.asarray(y_log2, dtype=float)
    cens = np.asarray(censored, dtype=bool)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, cens = x[ok], y[ok], cens[ok]
    if len(x) < 3 or np.unique(x).size < 2 or scipy_stats is None:
        return np.nan, np.nan, np.nan, np.nan

    # Initialize with censored-as-max OLS.
    beta0, alpha0, _ = _fit_loglog_beta(x, y)
    if not np.isfinite(beta0):
        beta0, alpha0 = 0.0, np.nanmean(y)
    sigma0 = np.nanstd(y - (alpha0 + beta0 * x))
    sigma0 = max(float(sigma0), 0.25)

    try:
        from scipy.optimize import minimize
    except Exception:
        return np.nan, np.nan, np.nan, np.nan

    def nll(params):
        alpha, beta, log_sigma = params
        sigma = np.exp(log_sigma) + 1e-8
        mu = alpha + beta * x
        ll = np.zeros_like(y)
        obs = ~cens
        if np.any(obs):
            ll[obs] = scipy_stats.norm.logpdf(y[obs], loc=mu[obs], scale=sigma)
        if np.any(cens):
            z = (censor_log2 - mu[cens]) / sigma
            # survival function P(Y >= censor)
            ll[cens] = scipy_stats.norm.logsf(z)
        if not np.all(np.isfinite(ll)):
            return 1e12
        return -float(np.sum(ll))

    res = minimize(
        nll,
        x0=np.array([alpha0, beta0, math.log(sigma0)]),
        method="Nelder-Mead",
        options={"maxiter": TOBIT_MAXITER, "xatol": 1e-5, "fatol": 1e-5},
    )
    if not res.success:
        # Still return the best estimate if finite.
        pass
    alpha, beta, log_sigma = res.x
    sigma = float(np.exp(log_sigma))
    return float(beta), float(alpha), sigma, float(res.fun)


def empirical_beta_scaling_summary(
    mstar: pd.DataFrame,
    n_boot: int = BETA_BOOTSTRAPS,
    tobit_boot: int = TOBIT_BOOTSTRAPS,
    seed: int = STATS_RANDOM_SEED,
):
    """
    Estimate empirical beta in M*(S) ∝ S^beta.

    Fit types:
      observed_only:
          Only S/seed points that actually reached threshold.
      censored_as_max_lower_bound:
          Replace unreached M* by M_max. Conservative finite-range descriptor.
      tobit_right_censored:
          Right-censored likelihood in log2 M*. Descriptive but more appropriate
          when many points are censored.

    All beta estimates remain empirical/protocol-dependent validation summaries,
    not direct estimates of the formal PAC exponent.
    """
    rows = []
    rng = np.random.default_rng(seed)

    for (threshold, task, variant), g in mstar.groupby(["threshold", "task", "variant"]):
        g = g.sort_values(["seed", "S"]).copy()
        max_M = float(g["max_M_tested"].max())
        censor_log2 = float(np.log2(max_M))
        n_total = int(len(g))
        n_reached = int(g["reached"].sum())
        frac_censored = float(g["right_censored"].mean())

        fit_inputs = []
        # observed only
        obs = g[g["reached"]].copy()
        fit_inputs.append(("observed_only", obs, False))
        # censored as max
        fit_inputs.append(("censored_as_max_lower_bound", g, True))
        # tobit
        fit_inputs.append(("tobit_right_censored", g, True))

        for fit_type, gg, use_censored_as_max in fit_inputs:
            if fit_type == "observed_only":
                x = np.log2(gg["S"].astype(float).values)
                y = np.log2(gg["M_star_observed"].astype(float).values)
                beta, alpha, r2 = _fit_loglog_beta(x, y)
                sigma = np.nan
                nll = np.nan
            elif fit_type == "censored_as_max_lower_bound":
                x = np.log2(gg["S"].astype(float).values)
                y = np.log2(gg["M_star_censored_as_max"].astype(float).values)
                beta, alpha, r2 = _fit_loglog_beta(x, y)
                sigma = np.nan
                nll = np.nan
            else:
                x = np.log2(gg["S"].astype(float).values)
                y = np.log2(gg["M_star_censored_as_max"].astype(float).values)
                cens = gg["right_censored"].astype(bool).values
                # Tobit is only meaningful when there are enough observed points
                # and the data are not almost completely censored. Fully or
                # nearly fully censored curves are reported as not identifiable.
                if n_reached < 3 or int(obs["S"].nunique()) < 2 or frac_censored >= 0.90 or frac_censored <= 0.0:
                    beta, alpha, sigma, nll = np.nan, np.nan, np.nan, np.nan
                    tobit_identifiable = False
                else:
                    beta, alpha, sigma, nll = _tobit_right_censored_beta(x, y, cens, censor_log2)
                    tobit_identifiable = bool(np.isfinite(beta))
                r2 = np.nan

            # Bootstrap over seeds to respect pairing/correlation within seed.
            boot_betas = []
            seeds_unique = np.array(sorted(g["seed"].unique()))
            this_n_boot = int(tobit_boot if fit_type == "tobit_right_censored" else n_boot)
            if len(seeds_unique) >= 2 and np.isfinite(beta):
                for _ in range(this_n_boot):
                    sampled = rng.choice(seeds_unique, size=len(seeds_unique), replace=True)
                    gb = pd.concat([g[g["seed"] == s] for s in sampled], ignore_index=True)
                    if fit_type == "observed_only":
                        gb = gb[gb["reached"]]
                        xb = np.log2(gb["S"].astype(float).values)
                        yb = np.log2(gb["M_star_observed"].astype(float).values)
                        b, _, _ = _fit_loglog_beta(xb, yb)
                    elif fit_type == "censored_as_max_lower_bound":
                        xb = np.log2(gb["S"].astype(float).values)
                        yb = np.log2(gb["M_star_censored_as_max"].astype(float).values)
                        b, _, _ = _fit_loglog_beta(xb, yb)
                    else:
                        gb_obs = gb[gb["reached"]]
                        if int(gb_obs.shape[0]) < 3 or int(gb_obs["S"].nunique()) < 2 or float(gb["right_censored"].mean()) >= 0.90 or float(gb["right_censored"].mean()) <= 0.0:
                            b = np.nan
                        else:
                            xb = np.log2(gb["S"].astype(float).values)
                            yb = np.log2(gb["M_star_censored_as_max"].astype(float).values)
                            cb = gb["right_censored"].astype(bool).values
                            b, _, _, _ = _tobit_right_censored_beta(xb, yb, cb, censor_log2)
                    if np.isfinite(b):
                        boot_betas.append(float(b))

            if len(boot_betas) >= 10:
                beta_ci_low = float(np.percentile(boot_betas, 2.5))
                beta_ci_high = float(np.percentile(boot_betas, 97.5))
                beta_boot_std = float(np.std(boot_betas, ddof=1))
            else:
                beta_ci_low = np.nan
                beta_ci_high = np.nan
                beta_boot_std = np.nan

            rows.append({
                "threshold": float(threshold),
                "task": task,
                "variant": variant,
                "display_name": VARIANT_DISPLAY_NAMES[variant],
                "fit_type": fit_type,
                "beta": float(beta) if np.isfinite(beta) else np.nan,
                "alpha_log2": float(alpha) if np.isfinite(alpha) else np.nan,
                "r2_loglog": float(r2) if np.isfinite(r2) else np.nan,
                "sigma_log2": float(sigma) if np.isfinite(sigma) else np.nan,
                "negative_log_likelihood": float(nll) if np.isfinite(nll) else np.nan,
                "beta_ci_low": beta_ci_low,
                "beta_ci_high": beta_ci_high,
                "beta_boot_std": beta_boot_std,
                "n_total_points": n_total,
                "n_reached_points": n_reached,
                "frac_right_censored": frac_censored,
                "n_unique_S_total": int(g["S"].nunique()),
                "n_unique_S_reached": int(obs["S"].nunique()) if len(obs) else 0,
                "max_M_tested": max_M,
                "n_boot_requested": this_n_boot,
                "tobit_identifiable": (
                    bool(np.isfinite(beta)) if fit_type == "tobit_right_censored" else np.nan
                ),
                "note": (
                    "Descriptive finite-range empirical beta from M*(S); "
                    "right-censoring means beta is not a direct estimate of the formal PAC exponent. "
                    "Tobit rows are set to NaN when too few thresholds are reached or censoring is extreme."
                ),
            })

    return pd.DataFrame(rows)


def sequence_robustness_auc(df: pd.DataFrame):
    """
    Area under accuracy-vs-log2(S) curve at largest M, normalized by log2(S) range.
    Higher AUC means greater sequence-length robustness. Computed per task/variant/seed.
    """
    max_M = df["M"].max()
    d = df[df["M"] == max_M].copy()
    rows = []
    for (task, variant, seed), g in d.groupby(["task", "variant", "seed"]):
        g = g.sort_values("S")
        if g["S"].nunique() < 2:
            continue
        x = np.log2(g["S"].astype(float).values)
        y = g["test_acc"].astype(float).values
        trapz_fn = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc = float(trapz_fn(y, x) / (x.max() - x.min()))
        rows.append({
            "task": task,
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "seed": int(seed),
            "accuracy_auc_over_log2S": auc,
        })
    auc_df = pd.DataFrame(rows)
    if auc_df.empty or "nlif_baseline" not in set(auc_df["variant"]):
        return auc_df, pd.DataFrame()
    base = auc_df[auc_df["variant"] == "nlif_baseline"][
        ["task", "seed", "accuracy_auc_over_log2S"]
    ].rename(columns={"accuracy_auc_over_log2S": "baseline_auc"})
    merged = auc_df.merge(base, on=["task", "seed"], how="inner")
    merged["delta_auc_vs_nlif"] = merged["accuracy_auc_over_log2S"] - merged["baseline_auc"]
    stat_rows = []
    for (task, variant), g in merged.groupby(["task", "variant"]):
        if variant == "nlif_baseline":
            continue
        stats_row = paired_delta_summary(g["delta_auc_vs_nlif"].values)
        stat_rows.append({
            "task": task,
            "comparison": f"{VARIANT_DISPLAY_NAMES[variant]} AUC - nLIF baseline AUC",
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            **stats_row,
        })
    return auc_df, pd.DataFrame(stat_rows)



def spike_participation_slopes(df: pd.DataFrame, metric_col: str = "spike_participation_density"):
    """
    Mechanistic slope: fit log2(spike metric) = alpha + slope * log2(S) at
    largest M for SNN variants only. The density metric is the main mechanistic
    measure because it normalizes for sequence length, hidden width, and number
    of SNN layers.
    """
    max_M = df["M"].max()
    d = df[(df["M"] == max_M) & df[metric_col].notna()].copy()
    d = d[d[metric_col] > 0].copy()

    rows = []
    for (task, variant, seed), g in d.groupby(["task", "variant", "seed"]):
        g = g.sort_values("S")
        if g["S"].nunique() < 2:
            continue
        x = np.log2(g["S"].values.astype(float))
        y = np.log2(g[metric_col].values.astype(float))
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({
            "task": task,
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            "seed": int(seed),
            "metric_col": metric_col,
            "log2_spike_metric_slope_per_log2S": float(slope),
            "intercept": float(intercept),
        })

    slope_cols = [
        "task", "variant", "display_name", "seed", "metric_col",
        "log2_spike_metric_slope_per_log2S", "intercept",
    ]
    stat_cols = [
        "task", "metric_col", "comparison", "variant", "display_name",
        "n_pairs", "mean_delta", "median_delta", "std_delta", "cohen_dz",
        "ci_low", "ci_high", "signed_rank_p", "sign_flip_p",
    ]
    slopes = pd.DataFrame(rows, columns=slope_cols)

    if slopes.empty or "nlif_baseline" not in set(slopes["variant"]):
        return slopes, pd.DataFrame(columns=stat_cols)

    base = slopes[slopes["variant"] == "nlif_baseline"][
        ["task", "seed", "log2_spike_metric_slope_per_log2S"]
    ].rename(columns={"log2_spike_metric_slope_per_log2S": "baseline_spike_slope"})

    merged = slopes.merge(base, on=["task", "seed"], how="inner")
    merged["delta_spike_slope_vs_nlif"] = (
        merged["log2_spike_metric_slope_per_log2S"] - merged["baseline_spike_slope"]
    )

    stat_rows = []
    for (task, variant), g in merged.groupby(["task", "variant"]):
        if variant == "nlif_baseline":
            continue
        stats_row = paired_delta_summary(g["delta_spike_slope_vs_nlif"].values)
        stat_rows.append({
            "task": task,
            "metric_col": metric_col,
            "comparison": f"{VARIANT_DISPLAY_NAMES[variant]} {metric_col} growth slope - nLIF baseline slope",
            "variant": variant,
            "display_name": VARIANT_DISPLAY_NAMES[variant],
            **stats_row,
        })

    return slopes, pd.DataFrame(stat_rows, columns=stat_cols)


paired_acc = add_bh_qvalues(paired_accuracy_vs_baseline(df, longS_only=False), p_col="signed_rank_p")
paired_acc = add_bh_qvalues(paired_acc, p_col="sign_flip_p")

paired_acc_longS = add_bh_qvalues(paired_accuracy_vs_baseline(df, longS_only=True), p_col="signed_rank_p")
paired_acc_longS = add_bh_qvalues(paired_acc_longS, p_col="sign_flip_p")

acc_slopes, acc_slope_stats = accuracy_degradation_slopes(df)
acc_slope_stats = add_bh_qvalues(acc_slope_stats, p_col="signed_rank_p")
acc_slope_stats = add_bh_qvalues(acc_slope_stats, p_col="sign_flip_p")

censor_summary = right_censoring_summary(all_mstar)
paired_mstar = add_bh_qvalues(paired_mstar_vs_baseline(all_mstar, threshold=0.75 if 0.75 in ACCURACY_THRESHOLDS else ACCURACY_THRESHOLDS[0]), p_col="signed_rank_p")
paired_mstar = add_bh_qvalues(paired_mstar, p_col="sign_flip_p")

beta_scaling_summary = empirical_beta_scaling_summary(all_mstar)
auc_df, auc_stats = sequence_robustness_auc(df)
auc_stats = add_bh_qvalues(auc_stats, p_col="signed_rank_p")
auc_stats = add_bh_qvalues(auc_stats, p_col="sign_flip_p")

spike_density_slopes, spike_density_slope_stats = spike_participation_slopes(df, metric_col="spike_participation_density")
spike_density_slope_stats = add_bh_qvalues(spike_density_slope_stats, p_col="signed_rank_p")
spike_density_slope_stats = add_bh_qvalues(spike_density_slope_stats, p_col="sign_flip_p")

spike_slopes_raw, spike_slope_stats_raw = spike_participation_slopes(df, metric_col="spike_participation_proxy")
spike_slope_stats_raw = add_bh_qvalues(spike_slope_stats_raw, p_col="signed_rank_p")
spike_slope_stats_raw = add_bh_qvalues(spike_slope_stats_raw, p_col="sign_flip_p")

max_M_for_summary = df["M"].max()
df_largest_M = df[df["M"] == max_M_for_summary].copy()
accuracy_at_largest_M_summary = summary_mean_sem(
    df_largest_M,
    ["task", "S", "variant", "display_name"],
    ["test_acc", "test_loss"],
)
long_sequence_accuracy_summary = summary_mean_sem(
    df_largest_M[df_largest_M["S"] >= 64],
    ["task", "variant", "display_name"],
    ["test_acc"],
)
spike_participation_at_largest_M_summary = summary_mean_sem(
    df_largest_M[df_largest_M["spike_participation_proxy"].notna()],
    ["task", "S", "variant", "display_name"],
    ["spike_participation_proxy", "spike_participation_density", "active_time_fraction", "mean_lateral_inhibition"],
)

stats_tables = {
    "accuracy_at_largest_M_summary.csv": accuracy_at_largest_M_summary,
    "long_sequence_accuracy_summary.csv": long_sequence_accuracy_summary,
    "spike_participation_at_largest_M_summary.csv": spike_participation_at_largest_M_summary,
    "paired_accuracy_vs_nlif_largest_M.csv": paired_acc,
    "paired_accuracy_vs_nlif_longS_largest_M.csv": paired_acc_longS,
    "accuracy_degradation_slopes.csv": acc_slopes,
    "accuracy_degradation_slope_stats_vs_nlif.csv": acc_slope_stats,
    "right_censoring_summary.csv": censor_summary,
    "paired_mstar_vs_nlif.csv": paired_mstar,
    "beta_scaling_summary.csv": beta_scaling_summary,
    "sequence_robustness_auc.csv": auc_df,
    "sequence_robustness_auc_stats_vs_nlif.csv": auc_stats,
    "spike_participation_density_slopes.csv": spike_density_slopes,
    "spike_participation_density_slope_stats_vs_nlif.csv": spike_density_slope_stats,
    "spike_participation_slopes_raw.csv": spike_slopes_raw,
    "spike_participation_slope_stats_raw_vs_nlif.csv": spike_slope_stats_raw,
}

for filename, table in stats_tables.items():
    path = os.path.join(OUTPUT_DIR, filename)
    table.to_csv(path, index=False)
    print("Saved:", path)

print("\nPaired accuracy vs nLIF baseline at largest M:")
display(paired_acc)
print("\nPaired accuracy vs nLIF baseline at largest M, long sequences only S>=64:")
display(paired_acc_longS)
print("\nRight-censoring summary:")
display(censor_summary)
print("\nSequence robustness AUC stats:")
display(auc_stats)
print("\nEmpirical beta scaling summary:")
display(beta_scaling_summary.sort_values(["threshold", "task", "variant", "fit_type"]))
print("\nSpike participation density slope stats:")
display(spike_density_slope_stats)
print("\nRaw spike participation slope stats:")
display(spike_slope_stats_raw)

# ============================================================
# 10. Focused plots
# ============================================================

def vlabel(variant: str) -> str:
    return VARIANT_DISPLAY_NAMES.get(variant, variant)


def _ordered_variants_present(d: pd.DataFrame):
    return [v for v in VARIANT_ORDER if v in set(d["variant"])]


def plot_accuracy_vs_S_at_largest_M(df: pd.DataFrame):
    max_M = df["M"].max()
    d = df[df["M"] == max_M].copy()

    for task in TASKS_TO_RUN:
        dt = d[d["task"] == task]
        if dt.empty:
            print(f"No largest-M accuracy data for task={task}; skipping plot.")
            continue
        plt.figure(figsize=(8.8, 5.8))

        for variant in _ordered_variants_present(dt):
            g = (
                dt[dt["variant"] == variant]
                .groupby("S")["test_acc"]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("S")
            )
            if len(g) == 0:
                continue
            sem = g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
            plt.errorbar(g["S"], g["mean"], yerr=sem, marker="o", capsize=3, label=vlabel(variant))

        plt.xscale("log", base=2)
        plt.xticks(SEQ_LENGTHS, [str(s) for s in SEQ_LENGTHS])
        plt.ylim(0.45, 1.02)
        plt.xlabel("Sequence length S")
        plt.ylabel(f"Test accuracy at largest M = {max_M}")
        plt.title(f"Accuracy degradation with sequence length: {task}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        savefig(f"main_accuracy_vs_S_{task}.png")


def plot_mstar_focused(mstar: pd.DataFrame, threshold: float = 0.75):
    d = mstar[mstar["threshold"] == threshold].copy()
    max_M = d["max_M_tested"].max()

    for task in TASKS_TO_RUN:
        dt = d[d["task"] == task]
        if dt.empty:
            print(f"No M* data for task={task}; skipping M* plot.")
            continue
        plt.figure(figsize=(9.2, 6.0))

        for variant in _ordered_variants_present(dt):
            gv = dt[dt["variant"] == variant]
            summary = (
                gv.groupby("S")["M_star_censored_as_max"]
                .agg(["median", "count"])
                .reset_index()
                .sort_values("S")
            )
            cens = gv.groupby("S")["right_censored"].mean().reset_index(name="frac_censored")
            summary = summary.merge(cens, on="S", how="left")
            if len(summary) == 0:
                continue

            plt.plot(summary["S"], summary["median"], marker="o", label=vlabel(variant))

            # Mark partially/fully censored S values with open circles and upward arrows.
            censored_summary = summary[summary["frac_censored"] > 0]
            if len(censored_summary) > 0:
                plt.scatter(
                    censored_summary["S"],
                    censored_summary["median"],
                    facecolors="none",
                    edgecolors="black",
                    s=70,
                    linewidths=1.2,
                    zorder=5,
                )
                for _, row in censored_summary.iterrows():
                    if row["median"] >= max_M:
                        plt.annotate(
                            "",
                            xy=(row["S"], row["median"] * 1.13),
                            xytext=(row["S"], row["median"]),
                            arrowprops=dict(arrowstyle="->", lw=1.0),
                        )

        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.xticks(SEQ_LENGTHS, [str(s) for s in SEQ_LENGTHS])
        plt.yticks(TRAIN_SIZES, [str(m) for m in TRAIN_SIZES])
        plt.xlabel("Sequence length S")
        plt.ylabel(f"M*(S), samples to reach validation accuracy ≥ {threshold:.2f}")
        plt.title(f"Samples-to-threshold with right-censoring: {task}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        savefig(f"main_mstar_{task}_{int(threshold * 100)}.png")


def plot_spike_participation_vs_S(df: pd.DataFrame, metric_col: str = "spike_participation_density", main: bool = True):
    max_M = df["M"].max()
    d = df[(df["M"] == max_M) & df[metric_col].notna()].copy()

    metric_label = (
        "Normalized hidden spike participation density"
        if metric_col == "spike_participation_density"
        else "Raw hidden spike participation proxy"
    )
    prefix = "main" if main else "supp"

    for task in TASKS_TO_RUN:
        dt = d[d["task"] == task]
        if dt.empty:
            print(f"No spike-participation data for task={task}, metric={metric_col}; skipping plot.")
            continue
        plt.figure(figsize=(8.8, 5.8))

        for variant in _ordered_variants_present(dt):
            gv = dt[dt["variant"] == variant]
            if gv[metric_col].notna().sum() == 0:
                continue
            g = (
                gv.groupby("S")[metric_col]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("S")
            )
            if len(g) == 0:
                continue
            sem = g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
            plt.errorbar(g["S"], g["mean"], yerr=sem, marker="o", capsize=3, label=vlabel(variant))

        plt.xscale("log", base=2)
        plt.yscale("log", base=10)
        plt.xticks(SEQ_LENGTHS, [str(s) for s in SEQ_LENGTHS])
        plt.xlabel("Sequence length S")
        plt.ylabel(metric_label)
        plt.title(f"Causal-participation proxy at largest M: {task}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend(fontsize=8, ncol=2)
        savefig(f"{prefix}_{metric_col}_{task}.png")


def plot_token_ablation_cue_recall(df: pd.DataFrame):
    d = df[
        (df["task"] == "cue_recall")
        & (df["M"] == df["M"].max())
        & df["intervention_causal_time_fraction"].notna()
    ].copy()

    if len(d) == 0:
        print("No token-ablation data found; skipping supplementary token-ablation plot.")
        return

    plt.figure(figsize=(8.8, 5.8))
    for variant in _ordered_variants_present(d):
        gv = d[d["variant"] == variant]
        g = (
            gv.groupby("S")["intervention_causal_time_fraction"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("S")
        )
        if len(g) == 0:
            continue
        sem = g["std"].fillna(0) / np.sqrt(g["count"].clip(lower=1))
        plt.errorbar(g["S"], g["mean"], yerr=sem, marker="o", capsize=3, label=vlabel(variant))

    plt.xscale("log", base=2)
    plt.xticks(SEQ_LENGTHS, [str(s) for s in SEQ_LENGTHS])
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Sequence length S")
    plt.ylabel("Fraction of time bins with causal effect")
    plt.title("Supplementary cue-recall token-ablation causal-time fraction")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    savefig("supp_token_ablation_cue_recall.png")



def plot_beta_scaling(beta_df: pd.DataFrame, threshold: float = 0.75):
    """
    Supplementary beta plot. Keep this out of the main figure when censoring is high.
    """
    if beta_df is None or beta_df.empty:
        print("No beta summary available; skipping beta plots.")
        return

    d = beta_df[beta_df["threshold"] == threshold].copy()
    if d.empty:
        print(f"No beta rows for threshold {threshold}; skipping beta plots.")
        return

    fit_order = ["observed_only", "censored_as_max_lower_bound", "tobit_right_censored"]
    fit_labels = {
        "observed_only": "Observed only",
        "censored_as_max_lower_bound": "Censored-as-max",
        "tobit_right_censored": "Tobit right-censored",
    }

    for task in TASKS_TO_RUN:
        dt = d[d["task"] == task].copy()
        if dt.empty:
            continue

        variants = _ordered_variants_present(dt)
        x = np.arange(len(variants))
        width = 0.23

        plt.figure(figsize=(10.5, 6.0))
        for i, fit_type in enumerate(fit_order):
            sub = dt[dt["fit_type"] == fit_type].set_index("variant")
            means = [sub.loc[v, "beta"] if v in sub.index else np.nan for v in variants]
            lows = [sub.loc[v, "beta_ci_low"] if v in sub.index else np.nan for v in variants]
            highs = [sub.loc[v, "beta_ci_high"] if v in sub.index else np.nan for v in variants]
            yerr_low = [m - lo if np.isfinite(m) and np.isfinite(lo) else 0 for m, lo in zip(means, lows)]
            yerr_high = [hi - m if np.isfinite(m) and np.isfinite(hi) else 0 for m, hi in zip(means, highs)]
            plt.errorbar(
                x + (i - 1) * width,
                means,
                yerr=[yerr_low, yerr_high],
                fmt="o",
                capsize=3,
                label=fit_labels[fit_type],
            )

        plt.axhline(2.0, linestyle="--", linewidth=1.2, alpha=0.65, label="Quadratic reference β=2")
        plt.xticks(x, [vlabel(v) for v in variants], rotation=35, ha="right")
        plt.ylabel(r"Empirical exponent $\beta$ in $M^*(S) \propto S^\beta$")
        plt.title(f"Supplementary empirical beta from M*(S): {task}, threshold={threshold}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        savefig(f"supp_beta_scaling_{task}_{int(threshold * 100)}.png")


plot_accuracy_vs_S_at_largest_M(df)
plot_mstar_focused(all_mstar, threshold=0.75 if 0.75 in ACCURACY_THRESHOLDS else ACCURACY_THRESHOLDS[0])
plot_spike_participation_vs_S(df, metric_col="spike_participation_density", main=True)
plot_spike_participation_vs_S(df, metric_col="spike_participation_proxy", main=False)
plot_token_ablation_cue_recall(df)
for _thr in ACCURACY_THRESHOLDS:
    plot_beta_scaling(beta_scaling_summary, threshold=_thr)

# ============================================================
# 11. Concise manuscript-facing summary
# ============================================================

def summarize_findings(df: pd.DataFrame, mstar: pd.DataFrame, beta_df: Optional[pd.DataFrame] = None):
    print("\n" + "=" * 80)
    print("MANUSCRIPT EMPIRICAL SUMMARY")
    print("=" * 80)

    max_M = df["M"].max()
    d = df[df["M"] == max_M].copy()

    print("\n1. Accuracy at largest M by task, S, and architecture")
    acc_summary = (
        d.groupby(["task", "S", "variant", "display_name"])["test_acc"]
        .mean()
        .reset_index()
        .sort_values(["task", "S", "variant"])
    )
    display(acc_summary)

    print("\n2. Right-censoring summary for M*(S)")
    display(right_censoring_summary(mstar).sort_values(["threshold", "task", "variant"]))

    print("\n3. Long-sequence paired accuracy improvements over nLIF baseline")
    display(paired_accuracy_vs_baseline(df, longS_only=True))

    print("\n4. Mechanistic spike-participation density slopes")
    _, spike_stats = spike_participation_slopes(df, metric_col="spike_participation_density")
    display(add_bh_qvalues(spike_stats))

    if beta_df is not None and len(beta_df) > 0:
        print("\n5. Supplementary empirical beta summaries")
        beta_cols = [
            "threshold", "task", "variant", "display_name", "fit_type", "beta",
            "beta_ci_low", "beta_ci_high", "n_reached_points", "frac_right_censored",
        ]
        display(beta_df[beta_cols].sort_values(["threshold", "task", "variant", "fit_type"]))

    print("\nSuggested manuscript framing:")
    print(
        "These experiments are finite synthetic sanity checks, not proofs of the PAC exponent. "
        "They show the predicted qualitative signature: baseline nLIF degrades sharply with "
        "sequence length and becomes right-censored in validation-selected M*(S), inhibitory/WTA/refractory variants "
        "partially rescue long-sequence performance by constraining normalized hidden spike participation, "
        "and GRU/Transformer baselines maintain stronger threshold reachability."
    )

summarize_findings(df, all_mstar, beta_scaling_summary)

# ============================================================
# 12. Zip outputs
# ============================================================

zip_base = OUTPUT_DIR
zip_path = shutil.make_archive(zip_base, "zip", OUTPUT_DIR)
print("\nZipped outputs:", zip_path)

print("\nDone.")
