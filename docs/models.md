# Model Training

This document describes the model-training code in this repo, the exact input contract it expects, how train/validation/test separation is enforced, what each model writes, and how to run the jobs on Juno.

The current training stack contains three models:

- logistic regression
- vanilla RNN
- LSTM

All three models train on the dedicated exact-width model-window dataset, not the older `sequence_dataset`.

## Purpose

The preprocessing pipeline already produced reusable model windows under:

- `/home/axa230262/scratch/insider/processed/model_windows/window_size=50`

The model code consumes that dataset directly and trains binary classifiers for the 5-minute forward markout signal.

Target:

- `label = 1` if the final trade in the 50-step window has positive user-side markout
- `label = 0` otherwise

Primary selection metric:

- validation `AUC-ROC`

Final reported evaluation:

- test `AUC-ROC` from the checkpoint or parameter state selected on validation only

## Data Contract

### Source dataset

All trainers read:

- `processed/model_windows/window_size=<N>/split=train/*.parquet`
- `processed/model_windows/window_size=<N>/split=validation/*.parquet`
- `processed/model_windows/window_size=<N>/split=test/*.parquet`

For the current run, `N = 50`.

The window dataset is already time-split by the pipeline using:

- `train_ratio = 0.8`
- `validation_ratio = 0.1`
- `test_ratio = 0.1`

Current verified counts from the reference run in [windowing-run-2026-04-06.md](/Users/ani/workspaces/github.com/anishalle/insider/references/windowing-run-2026-04-06.md):

- `train`: `5,985,294`
- `validation`: `747,974`
- `test`: `747,919`

### Per-row schema

Each parquet row contains:

- `window_id`
- `user`
- `market_id`
- `window_start_ts`
- `window_end_ts`
- `window_size`
- `features`
- `label`
- `split`

The training code only requires:

- `window_id`
- `features`
- `label`

### Feature tensor

`features` is a `50 x 7` numeric tensor in this order:

1. `price_yes`
2. `signed_token_amount`
3. `usd_amount`
4. `side`
5. `role_is_maker`
6. `time_delta_seconds`
7. `market_age_seconds`

Meaning:

- `price_yes`: normalized YES-side price
- `signed_token_amount`: positive for buy-like flow, negative for sell-like flow
- `usd_amount`: trade notional
- `side`: `+1` or `-1`
- `role_is_maker`: `1.0` for maker, `0.0` for taker
- `time_delta_seconds`: elapsed time since prior trade in the same `user + market_id` stream
- `market_age_seconds`: elapsed time since the first trade in that stream

### Model input shapes

Logistic regression uses flattened windows:

- input shape: `350`
- computed as `50 * 7`

RNN and LSTM use the sequence directly:

- input shape: `50 x 7`

## Leakage Controls

The training code is intentionally conservative about leakage.

Rules enforced by implementation:

- it never recomputes splits
- it reads only the stored `split=train`, `split=validation`, and `split=test` partitions
- class weights are computed from the train split only
- logistic-regression feature normalization is fit on the train split only
- validation is used only for model selection and early stopping
- test is evaluated only after selecting the best model state on validation

What is not done:

- no reshuffling across split boundaries
- no global normalization over all rows
- no oversampling or rebalancing using validation or test statistics

## Model Implementations

## Logistic Regression

Code:

- [src/lr/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/train.py)
- [src/lr/model.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/model.py)
- [src/lr/data.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/data.py)
- [src/lr/artifacts.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/artifacts.py)

Implementation details:

- standalone trainer, does not import the `insider` preprocessing package
- flattens each `50 x 7` window to a `350`-feature vector
- computes a train-only standardization pass
- applies weighted logistic loss with train-only positive-class weighting
- optimizes with a custom Adam update in NumPy
- keeps the best parameter state by validation `AUC-ROC`

Default CLI:

```bash
PYTHONPATH=src python -m lr.train \
  --config configs/pipeline.toml \
  --window-size 50
```

Default hyperparameters:

- `epochs = 20`
- `batch_size = 8192`
- `eval_batch_size = 16384`
- `learning_rate = 0.01`
- `l2 = 1e-4`
- `patience = 5`
- `seed = 7`

Notes:

- this is true logistic regression, not a tree model or MLP
- it uses streaming parquet reads so it does not need the full dataset resident at once
- it writes the learned weights, bias, and standardization parameters as JSON

## Vanilla RNN

Code:

- [src/rnn/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/train.py)
- [src/rnn/model.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/model.py)
- [src/rnn/data.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/data.py)
- [src/rnn/artifacts.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/artifacts.py)

Implementation details:

- standalone PyTorch trainer
- consumes the `50 x 7` tensor directly
- uses `nn.RNN(..., nonlinearity="tanh", batch_first=True)`
- uses the final hidden state for binary classification
- trains with `BCEWithLogitsLoss(pos_weight=...)`
- selects the best checkpoint by validation `AUC-ROC`
- runs final train, validation, and test prediction passes with the selected weights

Default CLI:

```bash
PYTHONPATH=src python -m rnn.train \
  --config configs/pipeline.toml \
  --window-size 50
```

Default hyperparameters:

- `epochs = 20`
- `batch_size = 1024`
- `eval_batch_size = 2048`
- `hidden_size = 64`
- `num_layers = 1`
- `dropout = 0.0`
- `learning_rate = 1e-3`
- `weight_decay = 0.0`
- `patience = 5`
- `seed = 42`
- `device = cuda if available else cpu`

## LSTM

Code:

- [src/lstm/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lstm/train.py)

Implementation details:

- standalone PyTorch trainer
- consumes the same `50 x 7` sequence input as the RNN
- applies train-only per-feature standardization before the LSTM sees the sequence
- uses `nn.LSTM(batch_first=True)`
- takes the final step output and applies a dropout + linear head
- uses train-only positive-class weighting through `BCEWithLogitsLoss`
- selects the best checkpoint by validation `AUC-ROC`

Default CLI:

```bash
PYTHONPATH=src python -m lstm.train \
  --config configs/pipeline.toml \
  --window-size 50 \
  --debug-metrics
```

Default hyperparameters:

- `epochs = 12`
- `batch_size = 1024`
- `eval_batch_size = 2048`
- `hidden_size = 128`
- `num_layers = 1`
- `dropout = 0.1`
- `learning_rate = 1e-3`
- `weight_decay = 1e-4`
- `patience = 3`
- `seed = 42`
- `device = cuda if available else cpu`

## Metrics

All three models report binary classification metrics driven by predicted probabilities.

Core metrics:

- `auc_roc`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `positive_rate`
- `predicted_positive_rate`
- confusion-matrix counts:
  - `true_positive`
  - `true_negative`
  - `false_positive`
  - `false_negative`

Each split payload also includes:

- `row_count`
- `positive_rows`
- `negative_rows`

Model-selection rule:

- choose the epoch or parameter state with the best validation `auc_roc`

## Output Layout

By default, runs are written under the configured processed root:

- logistic regression:
  - `<output.root>/models/logistic_regression/window_size=50/<timestamp>`
- RNN:
  - `<output.root>/models/rnn/window_size=50/<timestamp>`
- LSTM:
  - `<output.root>/models/lstm/window_size=50/<timestamp>`

You can override that with `--output-dir`.

### Common output files

All three trainers write:

- `summary.json`
- `metrics.json`
- `history.csv`
- `predictions_validation.parquet`
- `predictions_test.parquet`

LSTM also optionally writes:

- `diagnostics.json`

Enable that artifact with `--debug-metrics`. It includes split-level probability and logit summaries, PR-AUC, Brier score, and a validation threshold sweep that makes weighted-loss runs easier to interpret.

Prediction parquet schema:

- `window_id`
- `label`
- `probability`
- `prediction`
- `split`

### Model-specific artifacts

Logistic regression also writes:

- `model.json`

This contains:

- learned `weights`
- `bias`
- `feature_mean`
- `feature_scale`
- `feature_order`
- `window_size`
- `class_weights`
- `hyperparameters`

RNN and LSTM also write:

- `checkpoint.pt`

This contains the selected PyTorch model state and training metadata.

### `summary.json`

This is meant to be the quick run overview.

Typical contents:

- model name
- dataset directory
- output root
- window size
- feature order
- split summaries
- class weights
- feature standardization metadata for the LSTM trainer
- hyperparameters or model config
- best epoch
- best validation `AUC-ROC`

The exact key names vary slightly across the three trainers, but the payloads are intentionally parallel.

### `metrics.json`

This is the main evaluation artifact.

Typical contents:

- `train`
- `validation`
- `test`
- `best_epoch`
- `best_validation_auc_roc`

Each split block contains the metric set described above.

### `history.csv`

This stores epoch-level training history.

Typical columns include:

- `epoch`
- `train_loss`
- `validation_auc_roc`
- `validation_accuracy`

The LSTM history also records validation loss.

## Environment and Packaging

There are two different Python environments in this repo workflow.

### Repo-local preprocessing environment

The preprocessing and analysis code uses the repo-local `.venv` and the `insider` package.

### Juno training environment

The model trainers are designed to run in the Juno conda environment:

- `g_retriever`

Reason:

- the repo itself is Python `>= 3.11`
- `g_retriever` is Python `3.9`
- the recurrent models need the CUDA-enabled PyTorch stack that already exists there

To avoid mixing those concerns, the model trainers are standalone and are run with:

- `PYTHONPATH=src`

They do not import the main `insider` package during Juno training.

## Bootstrap Script

Juno model jobs use:

- [bootstrap-model-env.sh](/Users/ani/workspaces/github.com/anishalle/insider/scripts/bootstrap-model-env.sh)

Behavior:

- activates the requested Python environment before it runs
- checks for lightweight missing packages needed by the standalone trainers
- installs:
  - `numpy`
  - `pyarrow`
  - `scikit-learn`
  - `tomli` on Python 3.9/3.10 when needed
- optionally fails fast if `torch` is missing for GPU jobs

It does not install the repo itself into `g_retriever`.

## Juno Batch Workflow

Job scripts:

- logistic regression:
  - [jobs/lr/run-normal.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/lr/run-normal.sbatch)
- RNN:
  - [jobs/rnn/run-gpu.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/rnn/run-gpu.sbatch)
- LSTM:
  - [jobs/lstm/run-gpu.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/lstm/run-gpu.sbatch)

### Submission

From Juno:

```bash
cd "/home/axa230262/work/001 research/insider"
sbatch jobs/lr/run-normal.sbatch
sbatch jobs/rnn/run-gpu.sbatch
sbatch jobs/lstm/run-gpu.sbatch
```

### Script behavior

Each script:

1. changes into the Juno repo checkout
2. creates `logs/`
3. initializes `conda`
4. activates `g_retriever`
5. runs `scripts/bootstrap-model-env.sh`
6. exports `PYTHONPATH=src`
7. launches the trainer module

### Current resource choices

Logistic regression:

- partition: `normal`
- CPUs: `32`
- memory: `192G`

RNN:

- partition: `a30`
- GPUs: `1`
- CPUs: `32`
- memory: `192G`

LSTM:

- default script partition: `a30`
- GPUs: `1`
- CPUs: `32`
- memory: `192G`

If `a30` is saturated but `h100` is free, you can override at submission time:

```bash
sbatch -p h100 jobs/lstm/run-gpu.sbatch
```

or:

```bash
sbatch -p h100 jobs/rnn/run-gpu.sbatch
```

## Local Testing

Current test coverage includes:

- shared dataset/metrics utility tests
- logistic-regression end-to-end smoke training on a tiny synthetic dataset
- Python 3.9 AST parse checks for all trainer packages
- optional recurrent import smoke checks when local `torch` is installed

Relevant tests:

- [tests/test_lr_training.py](/Users/ani/workspaces/github.com/anishalle/insider/tests/test_lr_training.py)
- [tests/test_model_trainers.py](/Users/ani/workspaces/github.com/anishalle/insider/tests/test_model_trainers.py)
- [tests/test_modeling_common.py](/Users/ani/workspaces/github.com/anishalle/insider/tests/test_modeling_common.py)

## Operational Notes

- The trainers stream parquet batches rather than loading the entire dataset at once.
- Validation and test prediction files can still be large because they contain one row per example.
- Slurm log files may remain empty for a while due to buffering.
- The model-output directories should stay out of Git and remain under scratch-backed storage.
- If you want comparable experiments across models, keep `window_size`, feature order, and the underlying processed dataset fixed.

## Recommended Reading Order

For the full training context, read:

1. [pipeline.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/pipeline.md)
2. [output_dataset.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/output_dataset.md)
3. [models.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/models.md)
4. [windowing-run-2026-04-06.md](/Users/ani/workspaces/github.com/anishalle/insider/references/windowing-run-2026-04-06.md)

That sequence moves from preprocessing, to final data contract, to model implementation, to the verified reference run that produced the current `window_size=50` training dataset.
