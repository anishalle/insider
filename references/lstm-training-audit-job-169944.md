# LSTM Training Audit And Relaunch — Job 169944

This note records what was checked to diagnose the weak LSTM result from the completed run and why the trainer was changed before relaunching the model as Slurm job `169944`.

## Context

Completed LSTM run that was audited:

- run dir: `/home/axa230262/scratch/insider/processed/models/lstm/window_size=50/20260407T030719Z`
- originating Slurm job: `169901`
- reported result:
  - `best_epoch = 4`
  - `best_validation_auc_roc = 0.5740150642084294`

Relaunched LSTM run:

- Slurm job: `169944`
- submitted from: `/home/axa230262/work/001 research/insider`
- partition: `h100`

## What Was Audited

The audit used the actual model artifacts on Juno, not just the source code.

Files inspected in the completed run directory:

- `summary.json`
- `metrics.json`
- `history.csv`
- `checkpoint.pt`
- `predictions_validation.parquet`
- `predictions_test.parquet`

Repo code inspected locally:

- [src/lstm/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lstm/train.py)
- [src/lr/train.py](/Users/ani/workspaces/github.com/anishalle/insider/src/lr/train.py)
- [docs/models.md](/Users/ani/workspaces/github.com/anishalle/insider/docs/models.md)

## Findings

### 1. No leakage signal

The LSTM was weak, but it did not show the pattern expected from train/validation leakage.

Observed metrics from the completed run:

- validation `auc_roc = 0.5740150642084294`
- test `auc_roc = 0.5684089503638292`

Interpretation:

- validation and test were both only slightly above chance
- there was no explosive validation behavior
- this is consistent with a weak model, not a contaminated split

### 2. The model’s scores were badly compressed

The validation and test prediction parquet outputs showed that the model produced a very narrow probability range.

Validation probability summary:

- min: about `0.349`
- 1st percentile: about `0.503`
- median: about `0.525`
- 99th percentile: about `0.558`
- max: about `0.559`

Interpretation:

- the model had only weak ranking power
- it was barely separating classes in score space
- the issue was not just thresholding; the raw score spread itself was narrow

### 3. The “predicts almost everything positive” behavior was real, but partly a threshold artifact

The completed run reported:

- validation predicted positive rate: `0.9926026840505151`
- test predicted positive rate: `0.9829700809847055`

However, the trainer used:

- `BCEWithLogitsLoss(pos_weight=1.7106364497252586)`

That means a fixed `0.5` threshold is not the right reference point for interpreting weighted-loss outputs as calibrated class probabilities.

Interpretation:

- the thresholded metrics looked pathological
- but part of that was a reporting issue, not proof that the optimizer had completely failed

### 4. The most specific training issue found was missing feature standardization in the LSTM path

This was the clearest concrete defect found during the audit.

Comparison:

- logistic regression already computes train-only standardization
- the LSTM consumed raw sequence features with no normalization

Sampled real feature scales from the training window dataset on Juno showed severe mismatches:

- `price_yes`: bounded near `0` to `1`
- `side`: `-1` or `1`
- `role_is_maker`: `0` or `1`
- `time_delta_seconds`: median around `8`, 99th percentile around `92,184`, max above `20,634,010`
- `market_age_seconds`: median around `32,204`, 99th percentile around `18,543,742`, max above `25,771,786`

Interpretation:

- the recurrent model was training on features with radically different magnitudes
- this is a plausible reason for weak optimization and compressed outputs
- this was the main issue identified and acted on

### 5. There is temporal label drift across splits

Observed positive-label rates:

- train: `0.36891704902048256`
- validation: `0.4366809541508127`
- test: `0.4431816814387654`

Interpretation:

- the target distribution shifts materially over time
- this is not a trainer bug by itself
- it does make the modeling problem harder and complicates naive threshold interpretation

## Conclusion

The audit did not identify a single catastrophic bug such as:

- leakage
- split misuse
- broken early stopping
- corrupted checkpoint selection

The most specific issue that was actually pinpointed was:

- the LSTM trainer lacked train-only feature standardization despite consuming highly scale-mismatched sequence features

That was treated as the highest-confidence fix.

## Changes Made Before Relaunch

The LSTM path was updated to:

- fit train-only per-feature standardization statistics
- apply those statistics to train, validation, and test sequence batches
- add `--debug-metrics`
- write `diagnostics.json` with:
  - split-level probability summaries
  - split-level logit summaries
  - PR-AUC
  - Brier score
  - validation threshold sweep

The Slurm entrypoint was also updated so the relaunched job includes:

```bash
python -m lstm.train \
  --config configs/pipeline.toml \
  --window-size 50 \
  --device cuda \
  --debug-metrics
```

## Intended Outcome Of Job 169944

Job `169944` is intended to answer two questions cleanly:

1. Does train-only standardization improve LSTM ranking quality over the prior `0.5740` validation AUC run?
2. Are the weak thresholded metrics mostly a calibration/threshold issue, or is the underlying score separation still poor after scaling?

If the rerun still shows weak AUC and compressed scores, the next step should be feature/objective ablations rather than more threshold tweaking.
