"""Local, bounded and point-in-time volatility experiments (no network or writes).

Forecast at the end of observation t; target = std(r[t+1:t+h+1], ddof=1).
Returns and volatility are fractions per observation, NOT percentages or annualized.
All label boundaries refer to the original price index, before filtering samples.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def _window(window, minimum=1):
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < minimum:
        raise ValueError(f"window must be an integer >= {minimum}")
    return int(window)


def _index(index):
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("A sorted, unique DatetimeIndex without NaT is required")


def _close(data, column="Close"):
    close = data if isinstance(data, pd.Series) else data[column]
    if not isinstance(close, pd.Series):
        raise ValueError("Select exactly one asset before constructing targets")
    _index(close.index)
    close = close.astype(float).replace([np.inf, -np.inf], np.nan)
    if (close.dropna() <= 0).any():
        raise ValueError("Prices must be strictly positive or missing")
    return close


def future_returns(df, column="Close", window=10, log_returns=False):
    """P[t+h]/P[t]-1, or log(P[t+h]/P[t]); trailing h rows remain NaN."""
    window = _window(window)
    close = _close(df, column)
    ratio = close.shift(-window) / close
    value = np.log(ratio) if log_returns else ratio - 1
    name = f"future_returns_{'log_' if log_returns else ''}{window}obs"
    return value.to_frame(name)


def classify_trend(df, return_col="Close", window=10, threshold=0.01):
    """+1 bullish, -1 bearish, 0 neutral, inclusive +/- threshold; NaN stays NaN."""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite nonnegative return fraction")
    ret = future_returns(df, return_col, window).iloc[:, 0]
    trend = pd.Series(np.nan, index=ret.index, name=f"future_return_cat_{window}obs")
    trend.loc[ret.notna()] = 0.0
    trend.loc[ret > threshold] = 1.0
    trend.loc[ret < -threshold] = -1.0
    return trend


def future_volatility(df, return_col="Close", window=10):
    """Std of the NEXT h single-observation simple returns; no annualization."""
    window = _window(window, 2)
    close = _close(df, return_col)
    ret = close.pct_change(fill_method=None)
    return ret.rolling(window, min_periods=window).std(ddof=1).shift(-window).to_frame(f"future_volatility_{window}obs")


def label_intervals(index, horizon=10):
    """Every target uses prices from price_start=t through label_end=t+h."""
    horizon = _window(horizon, 2)
    _index(index)
    dates = pd.Series(index, index=index)
    return pd.DataFrame({
        "price_start": dates,
        "return_start": dates.shift(-1),
        "label_end": dates.shift(-horizon),
        "source_start_pos": np.arange(len(index)),
        "source_end_pos": pd.Series(np.arange(len(index)), index=index).shift(-horizon),
    }, index=index)


def causal_price_features(close, horizon=10, windows=(5, 10, 21), lags=(1, 2, 5)):
    """Fixed feature definitions available AFTER close t. No fit, bfill or macro.

    Missing prices remain missing. Computing on a prefix yields exactly the same
    features on that prefix as computing on the full series.
    """
    close = _close(close)
    horizon = _window(horizon, 2)
    ret = close.pct_change(fill_method=None)
    features = pd.DataFrame({"return_1obs": ret}, index=close.index)
    for window in sorted({_window(w, 2) for w in windows} | {horizon}):
        features[f"return_{window}obs"] = close / close.shift(window) - 1
        features[f"volatility_{window}obs"] = ret.rolling(window, min_periods=window).std(ddof=1)
    for lag in sorted({_window(lag) for lag in lags}):
        features[f"squared_return_lag_{lag}"] = ret.shift(lag).pow(2)
    features["distance_to_ma21"] = close / close.rolling(21, min_periods=21).mean() - 1
    return features.replace([np.inf, -np.inf], np.nan)


@dataclass
class TemporalDataset:
    X: pd.DataFrame
    y: pd.Series
    intervals: pd.DataFrame
    baseline: pd.Series

    def __post_init__(self):
        _index(self.X.index)
        if not all(self.X.index.equals(obj.index) for obj in (self.y, self.intervals, self.baseline)):
            raise ValueError("X, y, intervals and baseline must have identical ordered indexes")
        if self.X.columns.has_duplicates or not all(pd.api.types.is_numeric_dtype(t) for t in self.X.dtypes):
            raise ValueError("Features must be numeric with unique names")
        if not np.isfinite(self.y.to_numpy()).all() or not np.isfinite(self.baseline.to_numpy()).all():
            raise ValueError("Unknown targets/baselines must be excluded, never imputed")
        if (self.y < 0).any() or (self.baseline < 0).any():
            raise ValueError("Volatility cannot be negative")
        _validate_label_end(self.X.index, self.intervals["label_end"])

    def take(self, positions):
        return TemporalDataset(self.X.iloc[positions], self.y.iloc[positions], self.intervals.iloc[positions], self.baseline.iloc[positions])


def prepare_dataset(close, horizon=10, windows=(5, 10, 21), lags=(1, 2, 5), features=None):
    """Build targets BEFORE sample filtering; only X may be imputed within a fold."""
    close = _close(close)
    horizon = _window(horizon, 2)
    X = causal_price_features(close, horizon, windows, lags)
    y = future_volatility(close, window=horizon).iloc[:, 0]
    intervals = label_intervals(close.index, horizon)
    baseline = X[f"volatility_{horizon}obs"].rename("trailing_volatility")
    if features is not None:
        if not features or len(set(features)) != len(features) or not set(features).issubset(X.columns):
            raise ValueError("features must be a nonempty list of unique causal feature names")
        X = X.loc[:, list(features)]
    # Fixed warmup, not a data-driven choice. Never compress the label calendar.
    keep = (np.arange(len(close)) >= max(21, horizon, *windows)) & y.notna() & baseline.notna()
    return TemporalDataset(X.loc[keep], y.loc[keep], intervals.loc[keep], baseline.loc[keep])


def _validate_label_end(index, label_end):
    if not index.equals(label_end.index) or label_end.isna().any():
        raise ValueError("Label end dates must align exactly with X and be known")
    if not (label_end > index).all():
        raise ValueError("Every label must end strictly after its prediction date")


def purge_before(index, label_end, candidates, boundary, extra_days=0):
    """Strict boundary: no training target may use even the first held-out price."""
    _index(index)
    _validate_label_end(index, label_end)
    candidates = np.asarray(candidates, dtype=int)
    if isinstance(extra_days, bool) or not isinstance(extra_days, int) or not 0 <= extra_days <= 30:
        raise ValueError("extra_days must be an integer between 0 and 30 calendar days")
    boundary = boundary - pd.Timedelta(days=extra_days)
    keep = (index[candidates] < boundary) & (label_end.iloc[candidates] < boundary).to_numpy()
    return candidates[keep]


def assert_separated(index, label_end, train, held_out):
    if len(train) == 0 or len(held_out) == 0:
        raise ValueError("Not enough observations after purging")
    if np.intersect1d(train, held_out).size or label_end.iloc[train].max() >= index[held_out].min():
        raise ValueError("Target price intervals overlap a held-out partition")


def purged_cv_splits(X, label_end, n_splits=3, purge_extra_days=0):
    """TimeSeriesSplit candidates plus exact-date purging (works after missing rows).

    On an unfiltered fixed-horizon series this is equivalent to gap=horizon.
    Explicit endpoint purging is safer than counting surviving rows as days.
    """
    _index(X.index)
    _validate_label_end(X.index, label_end)
    if not 2 <= n_splits <= 5:
        raise ValueError("Use 2 to 5 inner folds")
    splits = []
    candidates = list(TimeSeriesSplit(n_splits=n_splits).split(X))
    for number, (train, val) in enumerate(candidates):
        train = purge_before(X.index, label_end, train, X.index[val[0]], purge_extra_days)
        if number + 1 < len(candidates):
            # Do not share target prices between consecutive validation blocks.
            next_validation_start = X.index[candidates[number + 1][1][0]]
            val = purge_before(X.index, label_end, val, next_validation_start, purge_extra_days)
        assert_separated(X.index, label_end, train, val)
        if len(train) < 8:
            raise ValueError("At least 8 training samples per purged fold are required")
        splits.append((train, val))
    return splits


@dataclass
class TemporalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp | None = None
    purge_extra_days: int = 0


def split_time_series(data, validation_start, test_start, train_start=None, test_end=None, purge_extra_days=0):
    index, ends = data.X.index, data.intervals["label_end"]
    train_start = index[0] if train_start is None else pd.Timestamp(train_start)
    validation_start, test_start = pd.Timestamp(validation_start), pd.Timestamp(test_start)
    test_end = None if test_end is None else pd.Timestamp(test_end)
    if not train_start < validation_start < test_start or (test_end is not None and test_start >= test_end):
        raise ValueError("Require train_start < validation_start < test_start < test_end")
    train = np.flatnonzero((index >= train_start) & (index < validation_start))
    train = purge_before(index, ends, train, validation_start, purge_extra_days)
    val = np.flatnonzero((index >= validation_start) & (index < test_start))
    val = purge_before(index, ends, val, test_start, purge_extra_days)
    test_mask = index >= test_start
    if test_end is not None:
        test_mask &= (index < test_end) & (ends < test_end - pd.Timedelta(days=purge_extra_days)).to_numpy()
    test = np.flatnonzero(test_mask)
    assert_separated(index, ends, train, val)
    assert_separated(index, ends, val, test)
    return TemporalSplit(train, val, test, train_start, validation_start, test_start, test_end, purge_extra_days)


def rolling_time_splits(data, train_size=756, validation_size=126, test_size=126, max_windows=3, purge_extra_days=0):
    """At most three rolling windows; caller passes DEVELOPMENT data only.

    Sizes count eligible forecast origins, not calendar days. Label purging still
    uses the original price dates. Adjacent test windows use disjoint target prices.
    """
    for size in (train_size, validation_size, test_size, max_windows):
        _window(size)
    if max_windows > 3:
        raise ValueError("At most 3 outer windows in this bounded experiment")
    total = train_size + validation_size + test_size
    starts = list(range(0, len(data.X) - total + 1, test_size))[-max_windows:]
    if not starts:
        raise ValueError("Insufficient development history for the requested rolling sizes")
    result = []
    for start in starts:
        stop = start + total
        boundary = data.X.index[stop] if stop < len(data.X) else data.intervals["label_end"].max() + pd.Timedelta(nanoseconds=1)
        result.append(split_time_series(data, data.X.index[start + train_size], data.X.index[start + train_size + validation_size], data.X.index[start], boundary, purge_extra_days))
    return result


def split_audit(data, split):
    rows = []
    for name, positions, cutoff in (
        ("train", split.train, split.validation_start),
        ("validation", split.validation, split.test_start),
        ("test", split.test, split.test_end),
    ):
        cutoff = None if cutoff is None else cutoff - pd.Timedelta(days=split.purge_extra_days)
        end = data.intervals["label_end"].iloc[positions].max()
        rows.append({"partition": name, "n": len(positions), "first_prediction": data.X.index[positions[0]], "last_prediction": data.X.index[positions[-1]], "last_label_price": end, "exclusive_cutoff": cutoff, "separated": cutoff is None or end < cutoff})
    return pd.DataFrame(rows)


def make_pipeline(model="xgboost", random_state=42):
    if model == "xgboost":
        from xgboost import XGBRegressor  # optional until explicitly selected
        regressor = XGBRegressor(objective="reg:squarederror", n_estimators=60, max_depth=2, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=random_state)
    elif model == "ridge":
        regressor = Ridge(alpha=1.0)
    else:
        raise ValueError("Supported bounded models: xgboost, ridge")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", RobustScaler()),
        ("select", SelectKBest(partial(mutual_info_regression, random_state=random_state), k="all")),
        ("model", regressor),
    ])


def volatility_rmse(y, prediction):
    # Nonnegative forecast rule is fixed BEFORE model selection and used in CV too.
    return float(np.sqrt(mean_squared_error(y, np.maximum(0.0, prediction))))


def tune_model(X, y, label_end, model="xgboost", param_grid=None, n_splits=3, random_state=42, purge_extra_days=0):
    if not X.index.equals(y.index) or not np.isfinite(y.to_numpy()).all():
        raise ValueError("Targets must be finite and aligned with X")
    if param_grid is None:
        param_grid = {"select__k": [min(6, X.shape[1]), "all"]}
        param_grid["model__max_depth" if model == "xgboost" else "model__alpha"] = [1, 2] if model == "xgboost" else [0.1, 1.0]
    if len(ParameterGrid(param_grid)) > 8:
        raise ValueError("Grid limited to 8 candidates; default is 4")
    if model == "xgboost":
        for params in ParameterGrid(param_grid):
            trees = _window(params.get("model__n_estimators", 60))
            depth = _window(params.get("model__max_depth", 2))
            if trees > 200 or depth > 4 or params.get("model__n_jobs", 1) != 1:
                raise ValueError("Bounded XGBoost requires 1..200 trees, depth 1..4 and one worker")
            if params.get("model__objective", "reg:squarederror") != "reg:squarederror":
                raise ValueError("Keep the same squared-error objective during CV and refit")
    splits = purged_cv_splits(X, label_end, n_splits, purge_extra_days)
    search = GridSearchCV(make_pipeline(model, random_state), param_grid, scoring=make_scorer(volatility_rmse, greater_is_better=False), cv=splits, n_jobs=1, error_score="raise", refit=True)
    search.fit(X, y)
    search.temporal_audit_ = pd.DataFrame([
        {"fold": i, "train_n": len(tr), "validation_n": len(va), "last_train_prediction": X.index[tr[-1]], "last_train_label_price": label_end.iloc[tr].max(), "first_validation_prediction": X.index[va[0]], "separated": bool(label_end.iloc[tr].max() < X.index[va[0]])}
        for i, (tr, va) in enumerate(splits)
    ])
    return search


def prediction_table(data, positions, model, train_mean):
    X = data.X.iloc[positions]
    return pd.DataFrame({"observed": data.y.iloc[positions], "model": np.maximum(0.0, model.predict(X)), "trailing_volatility": data.baseline.iloc[positions], "train_mean": float(train_mean)}, index=X.index)


def score_predictions(predictions):
    rows = []
    for column in ("model", "trailing_volatility", "train_mean"):
        rows.append({"model": column, "n": len(predictions), "rmse": volatility_rmse(predictions["observed"], predictions[column]), "mae": float(mean_absolute_error(predictions["observed"], predictions[column]))})
    return pd.DataFrame(rows).set_index("model")


def evaluate_split(data, split, model="xgboost", param_grid=None, n_splits=3, random_state=42):
    """Tune on train only; report validation; refit before test; evaluate test once.

    Validation scores do not select parameters/iterations. Refit admits previously
    purged samples only once their labels are known strictly before test_start.
    """
    ends = data.intervals["label_end"]
    assert_separated(data.X.index, ends, split.train, split.validation)
    assert_separated(data.X.index, ends, split.validation, split.test)
    search = tune_model(data.X.iloc[split.train], data.y.iloc[split.train], ends.iloc[split.train], model, param_grid, n_splits, random_state, split.purge_extra_days)
    validation = prediction_table(data, split.validation, search.best_estimator_, data.y.iloc[split.train].mean())
    candidates = np.flatnonzero((data.X.index >= split.train_start) & (data.X.index < split.test_start))
    refit_positions = purge_before(data.X.index, ends, candidates, split.test_start, split.purge_extra_days)
    assert_separated(data.X.index, ends, refit_positions, split.test)
    fitted = clone(search.best_estimator_).fit(data.X.iloc[refit_positions], data.y.iloc[refit_positions])
    test = prediction_table(data, split.test, fitted, data.y.iloc[refit_positions].mean())
    return {"search": search, "model": fitted, "refit_positions": refit_positions, "validation_predictions": validation, "test_predictions": test, "metrics": pd.concat({"validation": score_predictions(validation), "test": score_predictions(test)}, names=["partition"]), "audit": split_audit(data, split)}


def align_release_events(prediction_times, events, date_only=True):
    """Optional macro contract, NOT enabled in the notebook experiment.

    Each event is a certified point-in-time snapshot of one feature with columns
    available_at, observation_date, value. A revision becomes a NEW event at its
    real availability time. Events must represent the desired latest observation
    (an old-period revision must not accidentally replace a newer observation).
    With date_only=True (default), releases are conservatively usable from the
    NEXT calendar day. Otherwise available_at must be a verified timestamp.
    This join cannot certify that an input vintage is historically authentic.
    """
    _index(prediction_times)
    required = {"available_at", "observation_date", "value"}
    if not required.issubset(events.columns):
        raise ValueError("Macro data require available_at, observation_date and vintage value")
    right = events[["available_at", "observation_date", "value"]].copy()
    right["available_at"] = pd.to_datetime(right["available_at"])
    right["observation_date"] = pd.to_datetime(right["observation_date"])
    if right.empty or right.isna().any().any() or right["available_at"].duplicated().any():
        raise ValueError("Provide nonempty, unambiguous release snapshots without missing data")
    if (right["available_at"] < right["observation_date"]).any():
        raise ValueError("Release availability cannot precede its observation period")
    if not pd.api.types.is_numeric_dtype(right["value"]) or not np.isfinite(right["value"]).all():
        raise ValueError("Release values must be finite numbers")
    right = right.sort_values("available_at")
    if not right["observation_date"].is_monotonic_increasing:
        raise ValueError("Events must snapshot the latest observation, not overwrite it with an old revision")
    right["usable_at"] = right["available_at"].dt.normalize() + pd.Timedelta(days=1) if date_only else right["available_at"]
    if right["usable_at"].duplicated().any():
        raise ValueError("Provide one release snapshot per usable timestamp")
    left = pd.DataFrame({"prediction_at": prediction_times})
    joined = pd.merge_asof(left, right, left_on="prediction_at", right_on="usable_at", direction="backward", allow_exact_matches=date_only)
    return joined.set_index("prediction_at").rename_axis(prediction_times.name)
