# Model Training

This document describes the managed training stack in this repo, the current live Juno dataset it expects, the artifacts each trainer writes, and the tracked runs that are currently relevant.

## Managed Trainers

The repo currently contains four managed trainers:

- logistic regression
- XGBoost
- vanilla RNN
- LSTM

All four train on `model_windows/window_size=<N>`. They do not consume `sequence_dataset` directly.

## Current Live Dataset Contract

Canonical live dataset on Juno:

- `/home/axa230262/scratch/insider/processed/model_windows/window_size=50`

Verified live split counts on April 21, 2026:

| Split | Rows | Positive Rate |
|---|---:|---:|
| `train` | `3,018,859` | `0.5520396944673468` |
| `validation` | `365,112` | `0.5003368829290739` |
| `test` | `367,029` | `0.5001730108520035` |

Verified live shape:

- `window_size = 50`
- `length(features) = 50`
- `length(features[1]) = 16`

Current feature order:

1. `price_yes`
2. `signed_token_amount`
3. `usd_amount`
4. `side`
5. `role_is_maker`
6. `time_delta_seconds`
7. `market_age_seconds`
8. `market_trade_count_1h`
9. `market_volume_1h`
10. `market_price_mean_1h`
11. `market_price_std_1h`
12. `market_price_return_1h`
13. `user_trade_count_1h`
14. `user_market_trade_count_1h`
15. `user_signed_flow_1h`
16. `user_usd_volume_1h`

Per-row schema used by the trainers:

- `window_id`
- `features`
- `label`

## Historical Comparability Warning

`model_windows/window_size=50` has been rebuilt in place. Older April 7, 2026 runs used a previous `50 x 7` dataset with `7,481,187` rows at the same path.

That means historical runs are only comparable after checking the run’s own `summary.json`:

- older 7-feature runs: logistic regression, RNN, and the first XGBoost/LSTM experiments
- current 16-feature runs: the later XGBoost and LSTM experiments

Do not assume the live manifest matches a historical model artifact.

## Leakage Controls

The training code is intentionally conservative:

- it never recomputes splits
- it reads only the stored `split=train`, `split=validation`, and `split=test` partitions
- class weights are computed from the train split only
- feature scaling is fit on the train split only
- validation is used only for model selection and early stopping
- test is evaluated only after selecting the best model state on validation

What it does not do:

- reshuffle across split boundaries
- fit global normalization on all rows
- oversample using validation or test statistics

## Trainer Summaries

### Logistic Regression

Code:

- [src/lr/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/train.py)
- [src/lr/model.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/model.py)
- [src/lr/data.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/data.py)
- [src/lr/artifacts.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/artifacts.py)

Behavior:

- flattens each `50 x feature_width` window into a tabular vector
- computes train-only standardization
- optimizes weighted logistic loss in NumPy
- selects the best parameter state by validation `AUC-ROC`

Default hyperparameters:

- `epochs = 20`
- `batch_size = 8192`
- `eval_batch_size = 16384`
- `learning_rate = 0.01`
- `l2 = 1e-4`
- `patience = 5`
- `seed = 7`

### XGBoost

Code:

- [src/xgb_model/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/xgb_model/train.py)
- [src/xgb_model/data.py](/Users/ani/workspaces/github.com/anishalle/insider/src/xgb_model/data.py)
- [src/xgb_model/artifacts.py](/Users/ani/workspaces/github.com/anishalle/insider/src/xgb_model/artifacts.py)

Behavior:

- flattens the raw window
- appends derived window-summary features by default
- uses train-only class weighting through `scale_pos_weight`
- selects the best boosting round by validation `AUC-ROC`
- writes threshold and calibration diagnostics

Default hyperparameters:

- `num_round = 300`
- `early_stopping_rounds = 30`
- `learning_rate = 0.05`
- `max_depth = 8`
- `subsample = 0.8`
- `colsample_bytree = 0.8`
- `max_bin = 256`
- `tree_method = hist`
- `seed = 7`

### Vanilla RNN

Code:

- [src/rnn/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/train.py)
- [src/rnn/model.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/model.py)
- [src/rnn/data.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/data.py)
- [src/rnn/artifacts.py](/Users/ani/workspaces/github.com/anishalle/insider/src/rnn/artifacts.py)

Behavior:

- consumes the raw `50 x feature_width` tensor directly
- uses `nn.RNN(..., nonlinearity="tanh", batch_first=True)`
- uses the final hidden state for binary classification
- selects the best checkpoint by validation `AUC-ROC`

Default hyperparameters:

- `epochs = 20`
- `batch_size = 1024`
- `eval_batch_size = 2048`
- `hidden_size = 64`
- `num_layers = 1`
- `dropout = 0.0`
- `learning_rate = 1e-3`

### LSTM

Code:

- [src/lstm/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lstm/train.py)

Behavior:

- consumes the raw `50 x feature_width` tensor directly
- supports `last`, `mean`, `max`, `mean_last`, and `attention` pooling
- applies train-only feature transforms, clipping, and standardization
- optionally adds a summary-feature head
- selects the best checkpoint by validation `AUC-ROC`
- writes diagnostics when `--debug-metrics` is enabled

Common defaults:

- `epochs = 12`
- `hidden_size = 128`
- `num_layers = 1`
- `dropout = 0.1`
- `learning_rate = 1e-3`
- `weight_decay = 1e-4`

## Current Tracked Run Summary

### Current 16-feature dataset

These are the strongest tracked managed runs on the live `50 x 16` dataset:

| Model | Run | Validation AUC | Test AUC | Notes |
|---|---|---:|---:|---|
| XGBoost | `20260409T050523Z` | `0.673342` | `0.668963` | summary features enabled |
| LSTM | `20260414T044435Z` | `0.673177` | `0.670204` | attention pooling, no summary head |
| LSTM | `20260409T042628Z` | `0.672597` | `0.670263` | `mean_last` pooling, summary head |

### Older 7-feature dataset

These runs are still useful as historical reference, but they are not apples-to-apples with the current live dataset:

| Model | Run | Validation AUC | Test AUC |
|---|---|---:|---:|
| XGBoost | `20260407T194550Z` | `0.669553` | `0.690606` |
| LSTM | `20260407T171926Z` | `0.663738` | `0.686173` |
| LSTM | `20260407T070523Z` | `0.653211` | `0.673302` |
| RNN | `20260407T035704Z` | `0.582607` | `0.577572` |
| Logistic regression | `20260407T101004Z` | `0.537038` | `0.529491` |

## Output Layout

Trainer outputs are written under:

- `<output.root>/models/logistic_regression/window_size=<N>/<timestamp>`
- `<output.root>/models/xgboost/window_size=<N>/<timestamp>`
- `<output.root>/models/rnn/window_size=<N>/<timestamp>`
- `<output.root>/models/lstm/window_size=<N>/<timestamp>`

Common artifacts:

- `summary.json`
- `metrics.json`
- `history.csv`
- `predictions_validation.parquet`
- `predictions_test.parquet`

Model-specific artifacts:

- logistic regression: `model.json`
- RNN and LSTM: `checkpoint.pt`
- XGBoost and debug-enabled LSTM: `diagnostics.json`

Aggregate reports default to:

- `<output.root>/reports/model_leaderboard.csv`
- `<output.root>/reports/model_audit.csv`

## Juno Batch Workflow

Job scripts:

- logistic regression: [jobs/lr/run-normal.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/lr/run-normal.sbatch)
- XGBoost: [jobs/xgboost/run-normal.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/xgboost/run-normal.sbatch)
- RNN: [jobs/rnn/run-gpu.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/rnn/run-gpu.sbatch)
- LSTM: [jobs/lstm/run-gpu.sbatch](/Users/ani/workspaces/github.com/anishalle/insider/jobs/lstm/run-gpu.sbatch)

Each script:

1. changes into the Juno repo checkout
2. creates `logs/`
3. refuses to run from a dirty checkout unless you set `ALLOW_DIRTY_REPO=1`
4. boots the required environment
5. runs the standalone trainer from `PYTHONPATH=src`

The dedicated attention-LSTM script also refreshes the leaderboard after training, but it now writes that CSV under `<output.root>/reports/` instead of the repo checkout.

## Environment Notes

The preprocessing path uses the repo-local `.venv`.

The heavy model jobs are designed for the Juno `g_retriever` conda environment plus the bootstrap helper in [scripts/bootstrap-model-env.sh](/Users/ani/workspaces/github.com/anishalle/insider/scripts/bootstrap-model-env.sh).
