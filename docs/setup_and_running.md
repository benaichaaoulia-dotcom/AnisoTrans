# Installation, environment, and running

Data layout and model details are in [README.md](../README.md) and [model_and_training.md](model_and_training.md).

## Requirements

- Python 3.12 (`requires-python = ">=3.12,<3.13"`).
- [uv](https://docs.astral.sh/uv/) for environment and dependency management.
- An NVIDIA GPU with a driver that supports CUDA 13.0 for the default `torch==2.11.0` wheel from the `pytorch-cu130` index. CPU runs work with `--device cpu`.

## Installation

```bash
git clone https://github.com/benaichaaoulia-dotcom/AnisoTrans.git
cd AnisoTrans
uv sync --extra dev
```

`uv sync` creates the virtual environment at `.venv/` inside the repository and installs the runtime and dev dependencies. Do not use `pip` or `python -m venv` with this project.

To run commands inside the environment, either prefix them with `uv run` or activate it:

```bash
source .venv/bin/activate
```

## Dependencies

Declared in [pyproject.toml](../pyproject.toml):

| Group | Package | Constraint |
|---|---|---|
| runtime | `numpy` | `>=2.0,<3` |
| runtime | `pyyaml` | `>=6.0` |
| runtime | `torch` | `==2.11.0` (from `pytorch-cu130`) |
| dev (`--extra dev`) | `pytest` | `>=8.0` |
| dev (`--extra dev`) | `ruff` | `>=0.6` |

The repository does not include a `uv.lock`, so transitive dependency versions are resolved at install time. To pin them, run `uv lock` once and commit the resulting `uv.lock`.

## Data

Set the data directory once per shell, or pass `--data-dir` to each command:

```bash
export ANISOTRANS_DATA_DIR=/path/to/percell_field
```

The directory must contain `cache_densify_r6.npz`, `Kfield_gridordered.npy`, and `grid_coords_r6.npy`. Their SHA-256 hashes are checked when the data is loaded. Data files are not part of this repository.

## Training

```bash
uv run anisotrans-ccs --config configs/v2.yaml --out-dir artifacts/v2
```

- Trains with the settings in the config, keeps the best validation checkpoint as `best.pt` in `--out-dir`, then evaluates the test split once and writes `eval_frozen_test.json`.
- `--out-dir` defaults to `artifacts/<config stem>`.
- `--device cpu` or `--device cuda:N` overrides `hardware.device` from the config.
- `--max-epochs N` overrides `training.max_epochs`, for example for a short smoke run:

```bash
uv run anisotrans-ccs --config configs/v2.yaml --out-dir artifacts/smoke --max-epochs 2
```

## Evaluation

Evaluate an existing checkpoint without training:

```bash
uv run anisotrans-ccs --config configs/v2.yaml --eval-only path/to/best.pt --out-dir artifacts/eval
```

The checkpoint is loaded with a strict `state_dict` match, so it must come from the same architecture as the config. Output is written to `artifacts/eval/eval_frozen_test.json`.

## Ablation sweeps

This repository does not include ablation configs or a sweep runner. The ablation variants of the source work (`v2_small`, `v2b`, and the no-bias, no-positional-encoding, and no-scaler variants) are not included.

To run an ablation yourself, copy the config and change the values you want, then give each run its own output directory:

```bash
cp configs/v2.yaml configs/my_ablation.yaml
# edit configs/my_ablation.yaml, for example model.knn_k or model.n_layers
uv run anisotrans-ccs --config configs/my_ablation.yaml --out-dir artifacts/my_ablation
```

Keep the `data` section unchanged so that all runs use the same split. Record the config path with each result, because the output JSON stores it.

## Tests and linting

```bash
uv run pytest -q
uv run ruff check src tests
```

Ruff uses a line length of 100 (see [pyproject.toml](../pyproject.toml)).

## Outputs

Generated files go to `artifacts/` and are ignored by git. Checkpoints (`*.pt`, `*.pth`), data (`*.npz`, `*.npy`), and logs (`*.log`, `*.jsonl`) are also ignored.
