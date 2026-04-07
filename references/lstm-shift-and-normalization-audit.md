# LSTM Audit: Split Shift And Normalization

This note captures the two LSTM issues that remain in immediate scope after the successful rerun from Slurm job `169944`.

Best current LSTM run:

- run dir: `/home/axa230262/scratch/insider/processed/models/lstm/window_size=50/20260407T070523Z`
- originating Slurm job: `169944`
- best epoch: `5`
- validation `auc_roc = 0.6532109445192142`
- test `auc_roc = 0.6733024494465055`
- validation `pr_auc = 0.5717255664183264`
- test `pr_auc = 0.6028539107474256`

## 1. There Is Material Train-To-Eval Distribution Shift

The LSTM appears to be learning real ranking signal, but the underlying data distribution changes meaningfully across the time-based splits.

Observed label balance:

- train positive rate: `0.36891704902048256`
- validation positive rate: `0.4366809541508127`
- test positive rate: `0.4431816814387654`

Observed feature shift from a read-only sample on a Juno `dev` node:

- `price_yes` rises from about `0.368` in sampled train windows to `0.414` in validation and `0.431` in test
- `time_delta_seconds` falls from about `5631` in sampled train windows to `3133` in validation and `2694` in test
- `market_age_seconds` falls from about `1.20e6` in sampled train windows to `5.28e5` in validation and `4.62e5` in test
- `role_is_maker` falls from about `0.648` in sampled train windows to `0.581` in validation and `0.544` in test

Interpretation:

- the later splits are not just more positive, they also represent younger and denser trading contexts
- validation and test are not simple IID continuations of train
- any accuracy work on the LSTM should be treated as robustness-to-shift work, not only optimizer tuning

Immediate follow-up:

- report split-level feature drift alongside model metrics for every LSTM run
- inspect drift by timestep position within the 50-step window, not only after flattening
- compare fixed-threshold metrics only after noting that the class prior has shifted

## 2. The Current Normalization Strategy Is Better Than Raw Features, But Still Coarse

The trainer now standardizes each feature using train-only statistics aggregated over all timesteps. That was a necessary fix, but it still compresses several feature behaviors into one global mean and scale.

Current standardizer characteristics:

- one mean per feature over the full train split
- one scale per feature over the full train split
- no clipping, no log transform, no robust scaling
- no position-specific normalization inside the 50-step window
- binary features are scaled together with heavy-tailed continuous features

Why this is probably limiting:

- `time_delta_seconds` and `market_age_seconds` are extremely heavy-tailed
- `signed_token_amount` and `usd_amount` are centered near zero but still have occasional large spikes
- the positive and negative class means differ on the same features, but the global z-score can still be dominated by rare large values
- sequence position may matter: an event near the end of the window may deserve a different normalization treatment than the same value early in the window

Sampled train feature ranges from Juno:

- `price_yes`: min `0.001`, max `0.999`
- `signed_token_amount`: min `-0.10`, max `0.205`
- `usd_amount`: min `0.0`, max `0.150`
- `side`: min `-1`, max `1`
- `role_is_maker`: min `0`, max `1`
- `time_delta_seconds`: min `0`, max `20634010`
- `market_age_seconds`: min `0`, max `25771786`

Immediate follow-up:

- test log or signed-log transforms for:
  - `signed_token_amount`
  - `usd_amount`
  - `time_delta_seconds`
  - `market_age_seconds`
- test percentile clipping before standardization for the heavy-tailed continuous features
- test leaving `side` and `role_is_maker` unscaled
- inspect whether per-position normalization inside the window is materially different from global per-feature normalization

## Practical Reading Of The Current Result

The improved LSTM run is worth keeping. The jump from the earlier `0.5740` validation AUC run to `0.6532` validation AUC after standardization shows that feature scaling was not cosmetic.

At the same time, the remaining shift between train and later splits means:

- some of the model’s behavior is still tied to changing priors and changing feature marginals
- further improvement should prioritize data representation and normalization before more speculative architectural work
