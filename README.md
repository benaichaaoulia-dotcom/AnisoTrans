# AnisoTrans-CCS v2 (compact reproduction)

Self-contained PyTorch port of the AnisoTrans v2 model and its training/evaluation protocol (R6 per-cell split). Parameter count: 6,364,709.

## Scope

- Included: model (`src/anisotrans_ccs/model.py`), data loading and normalisation (`data.py`), executed objective (`objective.py`), training and evaluation (`train.py`), config (`configs/v2.yaml`), tests.
- Not included: model checkpoints, dataset files, `v2_small`, `v2b`, and other ablations.

## Data

Set `ANISOTRANS_DATA_DIR` (or pass `--data-dir`) to a directory containing:

| File | sha256 |
|---|---|
| `cache_densify_r6.npz` | `3de1d736340e3709ab17a3c14be18121ecdaf5c3325cb72d78d6e1793a148ad7` |
| `Kfield_gridordered.npy` | `837f1fe05f1eba869ee769937c4db2f5d580a7d44c5cf9d5a6e8147483fd5573` |
| `grid_coords_r6.npy` | `9b33cf3a92ec000bdaedb48d50ea56f01a667b3cb9a6950258c91352197473c8` |

Hashes are verified at load time. Split: 378 configs (263 fit / 40 val / 75 test), 13,334 cells.

## Setup

Requires Python 3.12 and uv. The torch pin (2.11.0) uses the cu130 index.

```bash
cd approved_repos/anisotrans-ccs
uv sync --extra dev
```

## Run

```bash
export ANISOTRANS_DATA_DIR=/path/to/percell_field

# Train (writes checkpoints and eval_frozen_test.json to --out-dir)
uv run anisotrans-ccs --config configs/v2.yaml --out-dir artifacts/v2

# Evaluate an existing checkpoint (strict state_dict load)
uv run anisotrans-ccs --config configs/v2.yaml --eval-only path/to/best.pt --out-dir artifacts/eval

# Tests
uv run pytest -q
```

Use `--device cpu` or `--device cuda:N` to override `hardware.device`.

## Expected result

Evaluating the reference checkpoint of the source run (`--eval-only`) gives:

- Pooled test relative RMSE: 0.11424796198618126
- Max per-config test relative RMSE: 0.33499641870511193

These match the source run's `eval_frozen_test.json` exactly.

## Discrepancies from the source

- The `lambda_smooth` term is not applied in the executed objective of the source run, so it is omitted here. The objective is masked MSE over valid fit cells plus the anchor term (`lambda_anchor` 0.05).
- Config scaler `exp_clip` is `[-3, 3]` (the source class default is `[-6, 6]`). The config value is used.
- kNN candidate selection may break ties differently across devices.
- Checkpoints are not bundled. Train, or supply your own `best.pt`.
- No license is chosen yet.
