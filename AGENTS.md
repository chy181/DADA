# Repository Guidelines

## Project Structure & Module Organization

This repository is a Python time-series anomaly detection project centered on `run.py`. Experiment orchestration lives in `exp/`, model definitions in `models/`, shared neural network components in `layers/`, data loading in `data_provider/` and `data_provider_private/`, metrics and helpers in `utils/`, and evaluation code in `ts_ad_evaluation/`. Dataset assets belong in `dataset/evaluation_dataset/data/`, with metadata in `dataset/evaluation_dataset/DETECT_META.csv`. Dataset-specific training scripts are under `scripts/fine-tune/stage1/` and `scripts/fine-tune/stage2/`. Generated checkpoints are stored in `checkpoints/`; experiment outputs are expected in `test_results/`.

## Build, Test, and Development Commands

Use Python 3.8, as specified in `readme.md`.

```sh
pip install -r requirements.txt
```

Installs runtime dependencies including PyTorch, pandas, scikit-learn, matplotlib, and Transformers.

```sh
sh ./scripts/fine-tune/stage1/SMAP/DADA.sh
sh ./scripts/fine-tune/stage2/MSL/DADA.sh
```

Runs dataset-specific fine-tuning experiments. Adjust `CUDA_VISIBLE_DEVICES`, `--dataset`, `--n_channel`, and hyperparameters in the script before launching.

```sh
python -m py_compile run.py exp/*.py models/*.py layers/*.py utils/*.py
```

Performs a lightweight syntax check when a full test suite is unavailable.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, snake_case functions and variables, PascalCase classes, and lowercase module names. Keep new command-line options consistent with the `argparse` patterns in `run.py`. Put reusable network blocks in `layers/`, model-level behavior in `models/`, and training or evaluation workflow changes in `exp/`.

## Testing Guidelines

No formal unit test framework is currently present. Before submitting changes, run the syntax check above and at least one small or representative `DADA.sh` script that exercises the modified path. For metric, scoring, or evaluator changes, compare generated files in `test_results/` against a known baseline and document expected differences.

## Commit & Pull Request Guidelines

This checkout does not include accessible Git history, so no repository-specific commit convention can be inferred. Use short, imperative commit subjects such as `Fix SMAP loader shape handling` or `Add diffusion score aggregation`. Pull requests should state the purpose, affected modules or datasets, commands run, metric changes, and any new data or checkpoint assumptions.

## Security & Configuration Tips

Do not commit raw private datasets, large generated checkpoints, or machine-specific absolute paths. Keep `DETECT_META.csv` aligned with files in `dataset/evaluation_dataset/data/`, especially `file_name` and `train_lens`. Prefer script-local GPU settings over global environment changes.
