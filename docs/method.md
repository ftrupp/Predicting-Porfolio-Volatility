# Temporal method and audit

At forecast origin t, the current close is already available. Define the simple
return `r[t] = P[t] / P[t-1] - 1`. For configurable horizon h, the target is the
sample standard deviation of `r[t+1], ..., r[t+h]` (`ddof=1`). It consumes every
price from t through t+h, inclusive. Values and forecast errors are return
fractions per observation; charts multiply them by 100. They are not annualized
and are not the volatility of an h-day cumulative return.

Forward-return helpers use `P[t+h]/P[t]-1` or `log(P[t+h]/P[t])`. Trend classes are
positive for gains, negative for losses, neutral at both thresholds and missing
when the target is unknown. These helpers are tested; volatility is the active
model objective.

## Availability and preprocessing

The feature dictionary is specified before fitting: past returns and rolling
volatility for configured windows, lagged squared returns, and relative distance
to the trailing 21-observation moving average. Selection by explicit feature name
is supported. No column depends on a future price. Prefix-invariance tests check
this directly. `AvailableAt <= Date` is checked at ingestion and exported for
every eligible forecast. Availability must also be nondecreasing, so every
earlier price used by a feature was known by the current origin.

Targets and their intervals are calculated before excluding incomplete samples.
The last h origins have no known targets. A target containing missing prices is
not imputed. The baseline's trailing-volatility window must also be available,
so every method is evaluated on identical dates. Remaining feature NaNs are
handled by the fold's fitted median imputer, with zero for an entirely missing
training feature. No backward price fill or learned full-sample cleaning occurs.

Each fold fits its own pipeline:

```text
train-only median imputation
  -> train-only robust scaling
  -> train-only mutual-information feature selection
  -> Ridge or XGBoost regression
```

Mutual information and the model use the configured seed. XGBoost is optional;
its default grid is four candidates (depth 1/2 and six/all features), with 60
trees, one worker, and the same squared-error objective throughout CV and refit.
The default Ridge grid uses alpha 0.1/1 and six/all features. Forecasts are floored
at zero by a predeclared rule used in both CV scoring and final reporting.

## Partitions and strict label-end purging

The train's last target price must be strictly earlier than the next partition's
first forecast origin. Equality is purged because the two labels would share a
price. On a complete fixed-horizon series this removes h origins before a cut.
After gaps or filtered rows, the algorithm uses original label dates, not a
surviving-row count. Optional `purge_extra_days` adds 0..30 calendar days to this
conservative separation; purging cannot be disabled.

Internal expanding folds are constructed from the outer train only. Both their
train tails and consecutive validation tails are purged by label endpoints.
Outer rolling windows are confined to development data whose labels finish
before the final test. Consecutive rolling tests have disjoint target-price
intervals. Past, matured labels can legitimately become training observations
in a later rolling window.

For each external split:

1. Tune hyperparameters with purged inner CV on train only.
2. Report validation without changing the policy or selecting parameters with it.
3. Refit the complete selected pipeline using labels already known strictly before
   test. Earlier purged observations may join only once their labels have matured.
4. Evaluate test once for that run and retain both reference forecasts.

The references are trailing h-observation volatility and the mean target of the
actual training/refit sample. They receive the same dates, target and units as
the model. Exported tables include label intervals, external boundaries, inner
fold endpoints, refit boundaries, availability, predictions and all errors.

The tests enumerate prices feeding each target, spy on actual transformer fits,
and alter final-test answers to verify that hyperparameters and fitted model
predictions do not change. The planning API does not instantiate an estimator.

## Interpretation

Errors remain dependent because adjacent targets overlap within a partition.
Purging does not create iid observations or validate ordinary confidence
intervals. Market regimes, survivorship, revised prices, corporate actions,
selection of assets and data quality still limit inference. The code does not
implement trading, transaction costs, portfolio allocation or execution. An
improvement in forecast error is not evidence of alpha or profitable decisions.

The archived AAPL test is already observed. Its 24.53% RMSE improvement is one
result among mixed periods, not a selection target for this release. New
hypotheses require predeclared new test dates and an honest research log.

References consulted October 30, 2026:

- [scikit-learn: TimeSeriesSplit and gap](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).
- [scikit-learn: data leakage and pipelines](https://scikit-learn.org/stable/common_pitfalls.html).
- [FRED/ALFRED: observation revisions and real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).
