# Empirical Validation for SNN Sequence-Length Scaling

**Associated paper:** *SNNs Are Not Transformers (Yet): The Architectural Problems for SNNs in Modeling Long-Range Dependencies*  
**Authors:** William Fishell, Gord Fishell, and Suraj Honnuraiah  
**Corresponding author and lead contact:** Suraj Honnuraiah  
**Contact:** suraj_honnuraiah@hms.harvard.edu

**Public-release script:** `SNN_Empirical_Validation_PAPER_FINAL_Suraj_Honnuraiah.py`

This repository contains a self-contained Python implementation of the empirical validation analyses accompanying the paper. The script generates synthetic sequence-learning tasks, trains the model variants used in the manuscript, computes validation-selected samples-to-threshold statistics, summarizes right-censoring, estimates finite-range empirical scaling summaries, and exports CSV tables and figures.

These experiments are **finite synthetic validation experiments**. They are designed to test whether the qualitative sequence-length phenotype predicted by the theory is visible in controlled settings. They are **not** direct PAC sample-complexity estimates and do **not** prove the covering-number theorem.

No external dataset is required. All data are generated synthetically by the script using fixed seeds.

---

## Files

```text
SNN_Empirical_Validation_PAPER_FINAL_Suraj_Honnuraiah.py
README.md
requirements.txt
LICENSE
```

---

## Installation

Use Python 3.10 or newer.

Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` file should contain:

```text
numpy
pandas
matplotlib
scipy
tqdm
torch
ipython
```

---

## Quick verification run

First confirm that the script parses and compiles:

```bash
python -m py_compile SNN_Empirical_Validation_PAPER_FINAL_Suraj_Honnuraiah.py
```

Then run the smoke test:

```bash
python SNN_Empirical_Validation_PAPER_FINAL_Suraj_Honnuraiah.py \
  --preset smoke_test \
  --output-dir smoke_test_outputs \
  --device auto \
  --disable-tqdm \
  --no-resume
```

Expected smoke-test outputs:

```text
smoke_test_outputs/
smoke_test_outputs.zip
```

Expected smoke-test checks:

- `raw_results.csv` contains **7 rows**.
- `missing_conditions.csv` is empty.
- `mstar_samples_to_threshold.csv` uses `selection_metric = best_val_acc`.
- The output directory contains CSV summaries, figures, `run_config.json`, and `README_methods_and_outputs.md`.

Verify the smoke test:

```bash
python - <<'PY'
import pandas as pd

out = "smoke_test_outputs"

raw = pd.read_csv(f"{out}/raw_results.csv")
missing = pd.read_csv(f"{out}/missing_conditions.csv")
mstar = pd.read_csv(f"{out}/mstar_samples_to_threshold.csv")

print("raw rows:", len(raw))
print("missing rows:", len(missing))
print("selection metrics:", sorted(mstar["selection_metric"].dropna().unique()))

assert len(raw) == 7
assert len(missing) == 0
assert set(mstar["selection_metric"].dropna().unique()) <= {"best_val_acc"}

print("Smoke test passed.")
PY
```

---

## Full paper run

The default full preset is `paper_stats_beta_optimized`.

Run the full paper-facing grid:

```bash
python SNN_Empirical_Validation_PAPER_FINAL_Suraj_Honnuraiah.py \
  --preset paper_stats_beta_optimized \
  --output-dir snn_scaling_outputs_paper_stats_beta_optimized \
  --device auto \
  --disable-tqdm
```

Full-grid design:

| Setting | Value |
|---|---:|
| Tasks | 2: `teacher_student`, `cue_recall` |
| Sequence lengths | 9: 8, 12, 16, 24, 32, 48, 64, 96, 128 |
| Training sizes | 8: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 |
| Seeds | 5: 0, 1, 2, 3, 4 |
| Architectures | 7 |
| Total training conditions | 5040 |
| Accuracy thresholds for M\*(S) | 0.70, 0.75, 0.80 |
| Main plotted M\*(S) threshold | 0.75 |

The full run is computationally expensive. Runtime depends on hardware, PyTorch installation, selected device, thread count, and whether deterministic kernels are requested. Runs can take many hours to multiple days on local hardware.

Implementation note: the SNN forward pass uses explicit time-step updates for clarity and auditability of membrane state, reset, leak, lateral inhibition, WTA competition, and refractory dynamics. This can introduce Python or GPU kernel-launch overhead, but the sequence lengths and model sizes used here are modest and appropriate for the finite synthetic validation experiments reported in the paper.

For a stricter deterministic CPU run:

```bash
python SNN_Empirical_Validation_PAPER_FINAL_Suraj_Honnuraiah.py \
  --preset paper_stats_beta_optimized \
  --output-dir snn_scaling_outputs_paper_stats_beta_optimized_deterministic \
  --device cpu \
  --deterministic \
  --disable-tqdm \
  --no-resume
```

Small numerical differences can occur across operating systems, PyTorch versions, BLAS libraries, and accelerator backends even with fixed random seeds. The run metadata are saved in `run_config.json`.

---

## Available presets

| Preset | Purpose | Expected raw rows |
|---|---|---:|
| `smoke_test` | Fast end-to-end functionality check | 7 |
| `quick_test` | Small two-task sanity run | 56 |
| `focused_validation` | Compact validation grid with all seven architectures | 1260 |
| `paper_stats_beta_optimized` | Full paper-facing grid; default | 5040 |
| `paper_stats_beta` | Alias for the full paper-facing grid | 5040 |
| `beta_scaling_core` | Focused beta-scaling grid using nLIF, LIF, GRU, and Transformer | 2880 |

Select a preset with:

```bash
--preset paper_stats_beta_optimized
```

or with an environment variable:

```bash
export SNN_RUN_PRESET=paper_stats_beta_optimized
```

Command-line flags override environment variables.

---

## Command-line options

| CLI option | Environment variable | Meaning |
|---|---|---|
| `--preset` | `SNN_RUN_PRESET` | Run preset |
| `--output-dir` | `SNN_OUTPUT_DIR` | Output directory |
| `--device auto/cpu/cuda` | `SNN_DEVICE` | Device selection |
| `--epochs` | `SNN_EPOCHS` | Maximum training epochs |
| `--patience` | `SNN_EARLY_STOPPING_PATIENCE` | Early-stopping patience |
| `--batch-size` | `SNN_BATCH_SIZE` | Training batch size |
| `--eval-batch-size` | `SNN_EVAL_BATCH_SIZE` | Evaluation batch size |
| `--test-n` | `SNN_TEST_N` | Test-set size per task/seed/S |
| `--calib-n` | `SNN_CALIB_N` | Teacher calibration set size |
| `--fixed-validation-n` | `SNN_FIXED_VALIDATION_N` | Fixed external validation-set size |
| `--stats-bootstraps` | `SNN_STATS_BOOTSTRAPS` | Bootstrap count for paired statistics |
| `--beta-bootstraps` | `SNN_BETA_BOOTSTRAPS` | Bootstrap count for empirical beta summaries |
| `--tobit-bootstraps` | `SNN_TOBIT_BOOTSTRAPS` | Bootstrap count for Tobit beta summaries |
| `--tobit-maxiter` | `SNN_TOBIT_MAXITER` | Maximum optimizer iterations for Tobit fits |
| `--compute-token-ablation` | `SNN_COMPUTE_TOKEN_ABLATION=1` | Enable supplementary cue-recall token ablation |
| `--no-token-ablation` | `SNN_COMPUTE_TOKEN_ABLATION=0` | Disable token ablation |
| `--disable-tqdm` | `SNN_DISABLE_TQDM=1` | Disable progress bars |
| `--deterministic` | `SNN_DETERMINISTIC=1` | Request deterministic PyTorch algorithms where available |
| `--no-resume` | `SNN_RESUME=0` | Ignore checkpoint resume behavior |
| `--num-threads` | `SNN_NUM_THREADS` | Torch thread count for local execution |

---

## Tasks and model variants

The script evaluates sequence-length scaling on two controlled binary classification tasks.

### Teacher-student task

A fixed nLIF teacher generates labels from sparse binary input sequences. Student models are trained to approximate the teacher. This task is closest to the theorem-facing feedforward nLIF setting.

### Cue-recall task

The first token contains a binary cue, the final token is a query marker, and intermediate tokens contain distractors. The task tests delayed long-range dependency learning.

The full comparison includes seven model variants:

| Internal variant | Display name |
|---|---|
| `nlif_baseline` | nLIF baseline |
| `nlif_lateral_inh_only` | nLIF + inhibition |
| `nlif_wta16` | nLIF WTA16 |
| `nlif_wta16_ref` | nLIF WTA16 + ref |
| `lif_baseline` | LIF |
| `gru` | GRU |
| `transformer` | Transformer |

The theorem-aligned baseline SNNs use positive feedforward weights. The inhibitory and WTA variants are empirical circuit-level controls that test whether constraining effective causal participation improves long-sequence behavior.

---

## Reproducibility design

### Fixed seed construction

The script constructs deterministic seeds from:

- task
- seed index
- sequence length `S`
- training size `M`
- model variant
- role: teacher, data, model initialization, training, or metrics

This prevents accidental reuse of the same random stream across conceptually distinct parts of the experiment.

### Fixed validation sets

By default, the script uses one fixed validation set per `task/seed/S` condition across all training sizes `M`. This keeps M\*(S) cleaner: the training sample size changes, while validation distribution and model initialization policy are controlled.

### Validation-selected M\*(S)

M\*(S) is selected using validation accuracy:

```text
best_val_acc >= threshold
```

Here `best_val_acc` is a backward-compatible column name for the validation accuracy at the checkpoint selected by minimum validation loss. Held-out test accuracy is used for final performance summaries and accuracy-versus-sequence-length plots. It is **not** used to decide M\*(S). This avoids test-set leakage.

### Checkpoint safety

The script writes completed conditions to `raw_results_checkpoint.csv` and stores a configuration signature in `checkpoint_config_signature.json`. A checkpoint is reused only when the current run configuration matches the saved signature.

To force a fresh run, use:

```bash
--no-resume
```

or delete the output directory before rerunning.

---

## Generated output files

Each run writes an output directory and a zipped copy:

```text
<output-dir>/
<output-dir>.zip
```

Core generated files:

| File | Purpose |
|---|---|
| `run_config.json` | Complete run metadata: preset, grid, seeds, device, package versions, and major hyperparameters |
| `README_methods_and_outputs.md` | Auto-generated methods summary for the completed run |
| `variant_config_table.csv` | Model-variant definitions |
| `parameter_table.csv` | Parameter counts and architecture settings |
| `raw_results.csv` | One row per completed training condition |
| `raw_results_checkpoint.csv` | Incremental checkpoint copy of raw results |
| `checkpoint_config_signature.json` | Configuration signature used for checkpoint safety |
| `missing_conditions.csv` | Conditions expected by the preset but missing from `raw_results.csv`; should be empty for a completed run |
| `accuracy_at_largest_M_summary.csv` | Mean/SEM test accuracy at largest training size |
| `long_sequence_accuracy_summary.csv` | Accuracy summaries for long sequences |
| `mstar_samples_to_threshold.csv` | Validation-selected M\*(S) and right-censoring status |
| `right_censoring_summary.csv` | Fraction of conditions that failed to reach threshold within the tested sample budget |
| `paired_mstar_vs_nlif.csv` | Paired log2 M\*(S) comparisons versus nLIF baseline |
| `beta_scaling_summary.csv` | Descriptive finite-range empirical beta summaries from M\*(S) |
| `sequence_robustness_auc.csv` | Normalized sequence-robustness AUC summaries |
| `sequence_robustness_auc_stats_vs_nlif.csv` | Paired sequence-robustness statistics versus nLIF baseline |
| `spike_participation_at_largest_M_summary.csv` | Hidden spike-participation summaries at largest `M` |
| `spike_participation_density_slopes.csv` | Slopes of normalized hidden spike-participation density versus sequence length |

Figure files are saved in PNG, PDF, and SVG formats.

Main figure patterns:

```text
main_accuracy_vs_S_<task>.*
main_mstar_<task>_75.*
main_spike_participation_density_<task>.*
```

Supplementary figure patterns:

```text
supp_beta_scaling_<task>_70.*
supp_beta_scaling_<task>_75.*
supp_beta_scaling_<task>_80.*
supp_spike_participation_proxy_<task>.*
supp_token_ablation_cue_recall.*
```

---

## Full-run integrity check

After a full paper run, execute:

```bash
python - <<'PY'
import json
import pandas as pd

out = "snn_scaling_outputs_paper_stats_beta_optimized"

raw = pd.read_csv(f"{out}/raw_results.csv")
missing = pd.read_csv(f"{out}/missing_conditions.csv")
mstar = pd.read_csv(f"{out}/mstar_samples_to_threshold.csv")
params = pd.read_csv(f"{out}/parameter_table.csv")
variants = pd.read_csv(f"{out}/variant_config_table.csv")

with open(f"{out}/run_config.json") as f:
    config = json.load(f)

print("raw rows:", len(raw))
print("missing rows:", len(missing))
print("variants:", sorted(raw["variant"].unique()))
print("tasks:", sorted(raw["task"].unique()))
print("sequence lengths:", sorted(raw["S"].unique()))
print("training sizes:", sorted(raw["M"].unique()))
print("thresholds:", sorted(mstar["threshold"].unique()))
print("run preset:", config["run_preset"])

expected_rows = (
    len(config["tasks_to_run"])
    * len(config["sequence_lengths"])
    * len(config["train_sizes"])
    * len(config["seeds"])
    * len(config["variants_to_run"])
)

assert len(raw) == expected_rows, (len(raw), expected_rows)
assert len(missing) == 0
assert sorted(params["variant"].tolist()) == sorted(config["variants_to_run"])
assert set(config["variants_to_run"]).issubset(set(variants["variant"]))
assert "best_val_acc" in raw.columns
assert "val_acc_at_best_val_loss" in raw.columns
assert "mstar_selection_acc" in raw.columns
assert "test_acc" in raw.columns
assert "selection_metric" in mstar.columns
assert set(mstar["selection_metric"].dropna().unique()) <= {"best_val_acc"}

print("All checks passed.")
PY
```

For the default full paper run, the expected result is:

```text
raw rows: 5040
missing rows: 0
All checks passed.
```

---

## Interpretation of key metrics

### Test accuracy at largest M

This measures final held-out performance after training with the largest sample budget.

### M\*(S)

M\*(S) is the smallest training-set size at which validation accuracy reaches a specified threshold. In the output tables, `best_val_acc` denotes validation accuracy at the best-validation-loss checkpoint. If the threshold is not reached within the tested sample grid, the point is marked as right-censored.

### Right-censoring

A right-censored point means that the model did not reach threshold accuracy within the largest tested training size.

### Empirical beta

The beta summaries are finite-range descriptive slopes fitted from M\*(S). They should be interpreted as protocol-dependent empirical summaries, not as formal PAC exponents.

### Hidden spike-participation density

This is a normalized activity-based proxy for hidden causal participation. It is used to assess whether inhibitory, WTA, or refractory constraints reduce diffuse hidden participation as sequence length increases.

---

## License

This repository is released under the MIT License. See the `LICENSE` file for details.
