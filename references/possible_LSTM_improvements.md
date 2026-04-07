# Possible LSTM Improvements

These items are intentionally deferred. They are not the current focus of the LSTM work.

## Optimization

- run a focused learning-rate sweep around the current baseline instead of keeping `1e-3` fixed
- compare `Adam` against `AdamW`
- add a scheduler such as `ReduceLROnPlateau` after validation AUC stalls
- add gradient clipping to stabilize recurrent optimization
- relax the current `patience=3` once a scheduler is present

## Architecture

- test a 2-layer LSTM with actual inter-layer dropout
- compare last-step pooling against mean/max pooling over all timesteps
- try an attention or gated pooling head so the classifier is not forced to rely only on the final hidden state
- consider a slightly richer classifier head on top of the sequence representation

## Decision Threshold And Calibration

- separate ranking metrics from operating-threshold metrics in all summaries
- choose thresholds on validation and freeze them before scoring test
- add explicit calibration checks before reading too much into `0.5` threshold accuracy

## Experiment Hygiene

- compare strong candidates over more than one seed before declaring a winner
- keep reporting AUC-ROC and PR-AUC as primary model-selection metrics
- continue to inspect score distributions so improvements are not mistaken for threshold artifacts
