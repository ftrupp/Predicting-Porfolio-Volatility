"""A small reproducible experiment API with explicit, immutable run artifacts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from predicting_volatility_data import PriceInput, utc_timestamp
from predicting_volatility_temporal import (
    evaluate_split, prepare_dataset, purge_before, rolling_time_splits,
    split_audit, split_time_series,
)

VERSION = "0.2.0"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str = "synthetic-demo"
    purpose: str = "synthetic_demo"
    selection_note: str = "Fixed demonstration settings; no market performance claim."
    horizon: int = 10
    seed: int = 42
    model: str = "ridge"
    inner_folds: int = 3
    feature_windows: tuple = (5, 10, 21)
    feature_lags: tuple = (1, 2, 5)
    features: tuple | None = None
    baselines: tuple = ("trailing_volatility", "train_mean")
    purge_policy: str = "label_end_strict"
    purge_extra_days: int = 0
    final_validation_size: int = 126
    final_test_size: int = 126
    validation_start: str | None = None
    test_start: str | None = None
    previously_observed_through: str | None = None
    rolling_train_size: int = 252
    rolling_validation_size: int = 63
    rolling_test_size: int = 63
    rolling_windows: int = 3

    def __post_init__(self):
        import re
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", self.experiment_id):
            raise ValueError("experiment_id must contain 1..64 lowercase letters, digits or hyphens")
        for name in ("horizon", "seed", "inner_folds", "purge_extra_days", "final_validation_size", "final_test_size", "rolling_train_size", "rolling_validation_size", "rolling_test_size", "rolling_windows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 2 <= self.horizon <= 252 or not 0 <= self.seed < 2**32:
            raise ValueError("Use horizon 2..252 and seed in [0, 2**32)")
        if self.model not in ("ridge", "xgboost") or not 2 <= self.inner_folds <= 5:
            raise ValueError("Use ridge/xgboost and 2..5 inner folds")
        if self.purge_policy != "label_end_strict" or not 0 <= self.purge_extra_days <= 30:
            raise ValueError("Strict label-end purging is required; extra calendar days: 0..30")
        for name in ("feature_windows", "feature_lags", "baselines"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.feature_windows or any(isinstance(v, bool) or not isinstance(v, int) or not 2 <= v <= 252 for v in self.feature_windows):
            raise ValueError("feature_windows must contain integers in 2..252")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 252 for v in self.feature_lags):
            raise ValueError("feature_lags must contain integers in 1..252")
        if self.features is not None:
            object.__setattr__(self, "features", tuple(self.features))
        if self.baselines != ("trailing_volatility", "train_mean"):
            raise ValueError("Keep both predeclared reference forecasts visible")
        if not 0 <= self.rolling_windows <= 3:
            raise ValueError("Use 0..3 rolling windows")
        if min(self.final_validation_size, self.final_test_size, self.rolling_train_size, self.rolling_validation_size, self.rolling_test_size) <= self.horizon:
            raise ValueError("Partition sizes must exceed the forecast horizon")
        if bool(self.validation_start) != bool(self.test_start):
            raise ValueError("Specify both validation_start and test_start, or neither")
        if self.purpose not in ("synthetic_demo", "historical_reproduction", "new_hypothesis"):
            raise ValueError("Declare synthetic_demo, historical_reproduction or new_hypothesis")
        if not isinstance(self.selection_note, str) or not 10 <= len(self.selection_note) <= 2000:
            raise ValueError("A 10..2000 character selection_note is required")
        if self.purpose == "new_hypothesis" and not (self.test_start and self.previously_observed_through):
            raise ValueError("New hypotheses need explicit split dates and previously_observed_through")

    @classmethod
    def from_json(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def source_hashes():
    root = Path(__file__).resolve().parent
    names = ("predicting_volatility_temporal.py", "predicting_volatility_data.py", "predicting_volatility_experiment.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def prepare_experiment(prices: PriceInput, config: ExperimentConfig):
    if (prices.metadata["data_kind"] == "synthetic") != (config.purpose == "synthetic_demo"):
        raise ValueError("Synthetic and observed experiments must be explicitly separated")
    data = prepare_dataset(prices.close, config.horizon, config.feature_windows, config.feature_lags, config.features)
    if len(data.X) <= config.final_test_size + config.final_validation_size + 40:
        raise ValueError("Insufficient history for the configured final partitions")
    val_start = utc_timestamp(config.validation_start) if config.validation_start else data.X.index[-config.final_test_size-config.final_validation_size]
    test_start = utc_timestamp(config.test_start) if config.test_start else data.X.index[-config.final_test_size]
    if config.purpose == "new_hypothesis":
        seen = utc_timestamp(config.previously_observed_through)
        if prices.metadata["symbol"].upper() == "AAPL":
            seen = max(seen, pd.Timestamp("2025-03-07 23:59:59Z"))
        if test_start <= seen:
            raise ValueError("The new holdout must start after all previously observed history")
    split = split_time_series(data, val_start, test_start, purge_extra_days=config.purge_extra_days)
    development_positions = purge_before(data.X.index, data.intervals.label_end, np.arange(len(data.X)), split.test_start, config.purge_extra_days)
    development = data.take(development_positions)
    rolling = []
    if config.rolling_windows:
        rolling = rolling_time_splits(development, config.rolling_train_size, config.rolling_validation_size, config.rolling_test_size, config.rolling_windows, config.purge_extra_days)
    return data, split, development, rolling


def make_plan(prices: PriceInput, config: ExperimentConfig):
    """Resolve dates and input/code hashes BEFORE fitting; no estimator is created."""
    data, split, development, rolling = prepare_experiment(prices, config)
    holdout = "synthetic" if config.purpose == "synthetic_demo" else "already_seen" if config.purpose == "historical_reproduction" else "declared_new_by_researcher"
    protocol = {
        "schema_version": 1, "project_version": VERSION, "config": asdict(config),
        "source_code_sha256": source_hashes(), "input_sha256": prices.fingerprint,
        "data_metadata": prices.metadata, "price_rows": len(prices.close), "eligible_samples": len(data.X),
        "feature_names": list(data.X.columns), "feature_availability_rule": "all input prices available at or before forecast timestamp; causal price features only",
        "prediction_convention": "after_close", "target_unit": "simple-return fraction per observation; sample std ddof=1",
        "holdout_status": holdout, "validation_start": split.validation_start.isoformat(),
        "test_start": data.X.index[split.test[0]].isoformat(), "test_last_prediction": data.X.index[split.test[-1]].isoformat(),
        "test_last_label_price": data.intervals.label_end.iloc[split.test].max().isoformat(),
        "rolling_windows": len(rolling), "all_baselines_reported": list(config.baselines),
    }
    # Canonical JSON normalization ensures tuples round-trip to arrays identically.
    protocol = json.loads(canonical(protocol))
    return {"protocol": protocol, "plan_id": hashlib.sha256(canonical(protocol).encode()).hexdigest(), "planned_at_utc": datetime.now(timezone.utc).isoformat()}


def validate_plan(plan, prices, config):
    current = make_plan(prices, config)
    if plan.get("plan_id") != current["plan_id"] or plan.get("protocol") != current["protocol"]:
        raise ValueError("Plan does not match data, source code or configuration; prepare a new plan")
    return current


def run_experiment(prices: PriceInput, config: ExperimentConfig, plan=None):
    """Run frozen bounded searches and return derived tables without writing files."""
    if prices.metadata["data_kind"] == "observed" and plan is None:
        raise ValueError("Observed data require a saved plan created before running")
    plan = make_plan(prices, config) if plan is None else plan
    validate_plan(plan, prices, config)
    data, split, development, rolling = prepare_experiment(prices, config)
    runs = []
    for number, outer in enumerate(rolling, 1):
        runs.append((f"rolling_{number}", development, outer))
    runs.append(("final", data, split))
    evaluations, metrics, predictions, outer_audits, inner_audits, refit_audits = {}, [], [], [], [], []
    for name, samples, outer in runs:
        result = evaluate_split(samples, outer, model=config.model, n_splits=config.inner_folds, random_state=config.seed)
        evaluations[name] = result
        metrics.append(result["metrics"].reset_index().assign(experiment=name))
        outer_audits.append(result["audit"].assign(experiment=name))
        inner_audits.append(result["search"].temporal_audit_.assign(experiment=name))
        for partition in ("validation", "test"):
            table = result[f"{partition}_predictions"].rename_axis("prediction_date").reset_index()
            predictions.append(table.assign(experiment=name, partition=partition))
        end = samples.intervals.label_end.iloc[result["refit_positions"]].max()
        cutoff = outer.test_start - pd.Timedelta(days=config.purge_extra_days)
        refit_audits.append({"experiment": name, "n": len(result["refit_positions"]), "last_label_price": end, "exclusive_label_cutoff": cutoff, "separated": bool(end < cutoff)})
    availability = pd.DataFrame({"latest_input_price_available_at": prices.available_at.loc[data.X.index]}, index=data.X.index)
    availability["forecast_at"] = availability.index
    availability["available"] = availability.latest_input_price_available_at <= availability.forecast_at
    tables = {"metrics": pd.concat(metrics, ignore_index=True), "predictions": pd.concat(predictions, ignore_index=True), "outer_audit": pd.concat(outer_audits, ignore_index=True), "inner_audit": pd.concat(inner_audits, ignore_index=True), "refit_audit": pd.DataFrame(refit_audits), "label_intervals": data.intervals.rename_axis("prediction_date").reset_index(), "feature_availability": availability.reset_index(drop=True)}
    for name in ("outer_audit", "inner_audit", "refit_audit"):
        if not tables[name]["separated"].all():
            raise RuntimeError("Temporal separation audit failed")
    if not availability["available"].all():
        raise RuntimeError("Feature availability audit failed")
    return {"plan": plan, "tables": tables, "evaluations": evaluations, "dataset": data}


def plot_final(result, path=None):
    """Return a figure for explicit display/save, preserving the caller's backend."""
    import matplotlib.pyplot as plt
    frame = result["evaluations"]["final"]["test_predictions"]
    ax = (100 * frame[["observed", "model", "trailing_volatility", "train_mean"]]).plot(figsize=(11, 4), linewidth=1.3)
    kind = result["plan"]["protocol"]["data_metadata"]["data_kind"].upper()
    ax.set(title=f"{kind} — final holdout volatility", ylabel="Volatility per observation (%)", xlabel="Forecast timestamp")
    ax.legend(["Realized target", "Model", "Trailing volatility", "Training mean"], ncol=2)
    plt.tight_layout()
    figure = ax.figure
    if path is not None:
        figure.savefig(path, dpi=140)
    # The returned Figure remains displayable; avoid a second inline auto-display.
    plt.close(figure)
    return figure


def write_run(result, output_root, include_plot=False):
    """Create a content-addressed run directory; never overwrite previous results."""
    plan = result["plan"]
    destination = Path(output_root) / f"{plan['protocol']['config']['experiment_id']}--{plan['plan_id'][:12]}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    for name, frame in result["tables"].items():
        frame.to_csv(destination / f"{name}.csv", index=False, float_format="%.12g", lineterminator="\n")
    if include_plot:
        plot_final(result, destination / "final_holdout.png")
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        "schema_version": 1, "project_version": VERSION, "plan_id": plan["plan_id"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "versions": versions,
        "data_kind": plan["protocol"]["data_metadata"]["data_kind"], "result_kind": "derived_forecast_errors",
        "holdout_status": plan["protocol"]["holdout_status"],
        "best_parameters": {name: r["search"].best_params_ for name, r in result["evaluations"].items()},
        "fit_count": sum(len(r["search"].cv_results_["params"]) * plan["protocol"]["config"]["inner_folds"] + 2 for r in result["evaluations"].values()),
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.iterdir()) if p.is_file()},
        "limitations": ["Forecast errors are not returns, alpha or profitability.", "Overlapping labels within blocks create dependent errors.", "Availability/provenance of external data is declared by its supplier, not independently certified.", "After inspection, a holdout is already seen for subsequent hypotheses."],
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return destination
