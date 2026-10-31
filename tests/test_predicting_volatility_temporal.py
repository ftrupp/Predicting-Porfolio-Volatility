"""Deterministic regressions for the functions actually imported by the notebook.

Run from the repository root: python -m unittest discover -s tests -p
test_predicting_volatility_*.py -v. No downloads, brokers or external credentials.
"""
import json
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest

import predicting_volatility_temporal as vt


def prices(n=400):
    t = np.arange(n, dtype=float)
    returns = 0.001 + 0.013 * np.sin(t / 3.1) + 0.006 * np.cos(t / 8.7)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=pd.bdate_range("2019-01-02", periods=n), name="Close")


class TargetTests(unittest.TestCase):
    def test_forward_return_sign_units_and_log(self):
        s = pd.Series([100, 110, 121, 90, 81], index=pd.bdate_range("2020-01-01", periods=5))
        simple = vt.future_returns(s, window=2).iloc[:, 0]
        logarithmic = vt.future_returns(s, window=2, log_returns=True).iloc[:, 0]
        self.assertAlmostEqual(simple.iloc[0], 0.21)
        self.assertAlmostEqual(simple.iloc[2], 81 / 121 - 1)
        np.testing.assert_allclose(logarithmic.iloc[:3], np.log1p(simple.iloc[:3]))
        self.assertTrue(simple.iloc[-2:].isna().all())
        self.assertTrue(logarithmic.iloc[-2:].isna().all())

    def test_trend_horizon_sign_and_inclusive_thresholds(self):
        s = pd.Series([100, 125, 100, 75, 75, 100, 100, 100], index=pd.bdate_range("2020-01-01", periods=8))
        one = vt.classify_trend(s, window=1, threshold=0.25)
        self.assertEqual(one.iloc[0], 0)  # equality is neutral
        self.assertEqual(one.iloc[2], 0)
        self.assertEqual(one.iloc[4], 1)
        two = vt.classify_trend(s, window=2, threshold=0.10)
        self.assertEqual(two.iloc[0], 0)
        self.assertEqual(two.iloc[1], -1)
        self.assertEqual(two.iloc[3], 1)
        self.assertTrue(two.iloc[-2:].isna().all())

    def test_volatility_uses_exact_future_returns_and_selected_column(self):
        s = prices(35)
        data = pd.DataFrame({"Close": 100.0, "chosen": s})
        for h in (2, 5, 10):
            got = vt.future_volatility(data, return_col="chosen", window=h).iloc[:, 0]
            for t in range(len(s) - h):
                chunk = s.iloc[t:t+h+1].to_numpy()
                expected = np.std(chunk[1:] / chunk[:-1] - 1, ddof=1)
                self.assertAlmostEqual(got.iloc[t], expected, places=13)
            self.assertTrue(got.iloc[-h:].isna().all())

    def test_missing_prices_do_not_get_filled_or_compress_horizon(self):
        s = prices(70)
        s.iloc[35] = np.nan
        target = vt.future_volatility(s, window=5).iloc[:, 0]
        self.assertTrue(target.iloc[30:36].isna().all())
        self.assertFalse(pd.isna(target.iloc[29]))
        dataset = vt.prepare_dataset(s, horizon=5)
        self.assertNotIn(s.index[35], dataset.X.index)
        for date, row in dataset.intervals.iterrows():
            self.assertEqual(row["label_end"], s.index[s.index.get_loc(date) + 5])
        self.assertTrue(dataset.X.index.equals(dataset.y.index))

    def test_features_are_available_on_each_price_prefix(self):
        s = prices(120)
        whole = vt.causal_price_features(s, 10)
        assert_frame_equal(whole.iloc[:77], vt.causal_price_features(s.iloc[:77], 10))
        changed = s.copy()
        changed.iloc[77:] *= 2
        assert_frame_equal(whole.iloc[:77], vt.causal_price_features(changed, 10).iloc[:77])

    def test_invalid_prices_indexes_windows(self):
        s = prices(30)
        for h in (0, 1, -5, 2.5, True):
            with self.assertRaises(ValueError):
                vt.future_volatility(s, window=h)
        with self.assertRaises(ValueError):
            vt.prepare_dataset(s.iloc[::-1])
        duplicated = pd.concat([s.iloc[:3], s.iloc[2:]])
        with self.assertRaises(ValueError):
            vt.prepare_dataset(duplicated)
        s.iloc[2] = 0
        with self.assertRaises(ValueError):
            vt.future_returns(s)


class SplitTests(unittest.TestCase):
    def assert_price_separation(self, data, left, right):
        # Every price feeding ANY left target precedes the first right prediction.
        self.assertLess(data.intervals.iloc[left]["label_end"].max(), data.X.index[right[0]])
        left_prices, right_prices = set(), set()
        for dest, positions in ((left_prices, left), (right_prices, right)):
            for _, r in data.intervals.iloc[positions].iterrows():
                dest.update(range(int(r.source_start_pos), int(r.source_end_pos) + 1))
        self.assertFalse(left_prices & right_prices)

    def test_outer_and_inner_splits_separate_every_target_price(self):
        for h in (2, 5, 10, 21):
            data = vt.prepare_dataset(prices(), h)
            split = vt.split_time_series(data, data.X.index[180], data.X.index[260])
            self.assert_price_separation(data, split.train, split.validation)
            self.assert_price_separation(data, split.validation, split.test)
            self.assertEqual(len(split.train), 180 - h)
            train = data.take(split.train)
            inner_splits = vt.purged_cv_splits(train.X, train.intervals.label_end, n_splits=3)
            for tr, va in inner_splits:
                self.assert_price_separation(train, tr, va)
            for (_, left), (_, right) in zip(inner_splits, inner_splits[1:]):
                self.assert_price_separation(train, left, right)
            self.assertTrue(vt.split_audit(data, split)["separated"].all())

    def test_irregular_calendar_and_filtered_samples_keep_original_end_dates(self):
        s = prices(360).drop(prices(360).index[[9, 10, 42, 130]])
        s.iloc[150] = np.nan
        data = vt.prepare_dataset(s, 10)
        data = data.take(np.flatnonzero(np.arange(len(data.X)) % 7 != 0))
        for tr, va in vt.purged_cv_splits(data.X, data.intervals.label_end):
            self.assert_price_separation(data, tr, va)
        for date in data.X.index:
            self.assertEqual(data.intervals.loc[date, "label_end"], s.index[s.index.get_loc(date)+10])

    def test_rolling_windows_have_disjoint_test_target_prices(self):
        data = vt.prepare_dataset(prices(500), 10)
        splits = vt.rolling_time_splits(data, train_size=150, validation_size=60, test_size=60)
        self.assertEqual(len(splits), 3)
        for split in splits:
            self.assert_price_separation(data, split.train, split.validation)
            self.assert_price_separation(data, split.validation, split.test)
        for left, right in zip(splits, splits[1:]):
            self.assert_price_separation(data, left.test, right.test)

    def test_alignment_and_insufficient_history_fail_explicitly(self):
        data = vt.prepare_dataset(prices(100))
        with self.assertRaises(ValueError):
            vt.purged_cv_splits(data.X, data.intervals.label_end.iloc[::-1])
        with self.assertRaises(ValueError):
            vt.rolling_time_splits(data)
        with self.assertRaises(ValueError):
            vt.split_time_series(data, data.X.index[5], data.X.index[6])
        with self.assertRaises(ValueError):
            vt.TemporalDataset(data.X, data.y.iloc[::-1], data.intervals, data.baseline)


class LearningTests(unittest.TestCase):
    def test_imputation_scaling_and_mi_fit_separately_in_each_fold(self):
        data = vt.prepare_dataset(prices(180), 5)
        X = data.X.copy()
        X.iloc[::9, 0] = np.nan
        X.iloc[100:, 1] += 1000
        X["empty_training_feature"] = np.nan
        X.loc[X.index[100:], "empty_training_feature"] = 1e6
        records = {"imputer": [], "scaler": [], "selector": []}
        imputer_fit, scaler_fit, selector_fit = SimpleImputer.fit, RobustScaler.fit, SelectKBest.fit

        def imputer(self, values, y=None, **kwargs):
            records["imputer"].append(values.index.copy())
            return imputer_fit(self, values, y, **kwargs)

        def scaler(self, values, y=None, **kwargs):
            result = scaler_fit(self, values, y, **kwargs)
            records["scaler"].append(self.center_.copy())
            return result

        def selector(self, values, y=None, **kwargs):
            records["selector"].append(y.index.copy())
            return selector_fit(self, values, y, **kwargs)

        grid = {"select__k": ["all"], "model__alpha": [1.0]}
        with patch.object(SimpleImputer, "fit", imputer), patch.object(RobustScaler, "fit", scaler), patch.object(SelectKBest, "fit", selector):
            search = vt.tune_model(X, data.y, data.intervals.label_end, "ridge", grid, n_splits=2)
        expected = [tr for tr, _ in vt.purged_cv_splits(X, data.intervals.label_end, n_splits=2)] + [np.arange(len(X))]
        self.assertEqual(len(records["imputer"]), len(expected))
        for i, positions in enumerate(expected):
            self.assertTrue(records["imputer"][i].equals(X.iloc[positions].index))
            self.assertTrue(records["selector"][i].equals(X.iloc[positions].index))
            expected_centers = X.iloc[positions].median().fillna(0).to_numpy()
            np.testing.assert_allclose(records["scaler"][i], expected_centers)
        self.assertTrue(search.temporal_audit_["separated"].all())

    def test_final_test_labels_cannot_change_parameters_or_fitted_predictions(self):
        data = vt.prepare_dataset(prices(300), 5)
        split = vt.split_time_series(data, data.X.index[140], data.X.index[210])
        kwargs = {"model": "ridge", "param_grid": {"select__k": ["all"], "model__alpha": [0.1, 1.0]}, "n_splits": 2}
        original = vt.evaluate_split(data, split, **kwargs)
        tampered = vt.TemporalDataset(data.X.copy(), data.y.copy(), data.intervals.copy(), data.baseline.copy())
        tampered.y.iloc[split.test] = 123.0
        changed = vt.evaluate_split(tampered, split, **kwargs)
        self.assertEqual(original["search"].best_params_, changed["search"].best_params_)
        assert_series_equal(original["test_predictions"]["model"], changed["test_predictions"]["model"])
        self.assertLess(data.intervals.label_end.iloc[original["refit_positions"]].max(), split.test_start)
        self.assertEqual(original["test_predictions"].shape[0], len(split.test))
        self.assertGreater(changed["metrics"].loc[("test", "model"), "rmse"], 100)

    def test_baselines_same_observations_and_units(self):
        data = vt.prepare_dataset(prices(220), 5)
        split = vt.split_time_series(data, data.X.index[100], data.X.index[150])
        result = vt.evaluate_split(data, split, model="ridge", param_grid={"select__k": ["all"], "model__alpha": [1]}, n_splits=2)
        pred = result["test_predictions"]
        assert_series_equal(pred["trailing_volatility"], data.baseline.iloc[split.test], check_names=False)
        np.testing.assert_allclose(pred["train_mean"], data.y.iloc[result["refit_positions"]].mean())
        self.assertTrue((result["metrics"].loc["test", "n"] == len(split.test)).all())
        direct_rmse = np.sqrt(np.mean((pred.observed - pred.trailing_volatility) ** 2))
        self.assertAlmostEqual(result["metrics"].loc[("test", "trailing_volatility"), "rmse"], direct_rmse)

    def test_search_budget_and_objective_are_guarded(self):
        data = vt.prepare_dataset(prices(100), 5)
        for grid in ({"model__max_depth": list(range(9))}, {"model__max_depth": [0]}, {"model__n_estimators": [10000]}, {"model__n_estimators": [0]}, {"model__n_jobs": [-1]}, {"model__objective": ["reg:absoluteerror"]}):
            with self.assertRaises(ValueError):
                vt.tune_model(data.X, data.y, data.intervals.label_end, param_grid=grid)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "optional xgboost extra is not installed")
    def test_small_xgboost_search_refits_the_same_objective(self):
        data = vt.prepare_dataset(prices(150), 5)
        search = vt.tune_model(data.X, data.y, data.intervals.label_end, param_grid={"model__n_estimators": [5], "model__max_depth": [1], "select__k": ["all"]}, n_splits=2)
        self.assertEqual(search.best_estimator_.named_steps["model"].get_params()["objective"], "reg:squarederror")
        self.assertTrue(np.isfinite(search.predict(data.X.tail(3))).all())


class MacroAndNotebookTests(unittest.TestCase):
    def test_release_dates_and_revisions_are_not_backfilled(self):
        dates = pd.date_range("2020-01-01", periods=12)
        events = pd.DataFrame({"observation_date": ["2019-12-01", "2019-12-01"], "available_at": ["2020-01-04", "2020-01-09"], "value": [1.0, 9.0]})
        result = vt.align_release_events(dates, events)
        self.assertTrue(result.loc[:"2020-01-04", "value"].isna().all())
        self.assertTrue((result.loc["2020-01-05":"2020-01-09", "value"] == 1).all())
        self.assertTrue((result.loc["2020-01-10":, "value"] == 9).all())
        with self.assertRaises(ValueError):
            vt.align_release_events(dates, events.drop(columns="available_at"))
        with self.assertRaises(ValueError):
            vt.align_release_events(dates, events.assign(observation_date=["2019-12-01", "2019-11-01"]))

    def test_release_timestamp_must_precede_forecast_timestamp(self):
        predictions = pd.to_datetime(["2020-02-01 08:00Z", "2020-02-01 08:30Z", "2020-02-01 09:00Z"])
        events = pd.DataFrame({"available_at": pd.to_datetime(["2020-02-01 08:30Z"]), "observation_date": pd.to_datetime(["2020-01-01 00:00Z"]), "value": [4]})
        got = vt.align_release_events(predictions, events, date_only=False)
        self.assertTrue(got["value"].iloc[:2].isna().all())
        self.assertEqual(got["value"].iloc[2], 4)

    def test_notebook_uses_synthetic_core_and_marks_regenerated_outputs(self):
        path = Path(__file__).resolve().parents[1] / "volatility_prediction.ipynb"
        nb = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
        self.assertIn("from predicting_volatility_experiment import", source)
        self.assertIn("run_experiment(", source)
        self.assertEqual(nb["metadata"]["volatility_lab"]["data_kind"], "synthetic")
        plan = json.loads((path.parent / "docs/results/synthetic_plan.json").read_text())
        self.assertEqual(nb["metadata"]["volatility_lab"]["plan_id"], plan["plan_id"])
        for disallowed in ("yf.download", "fred.get_series", "data_fetcher", "financial_instruments", "pct_change(-", "pip install", "api_key=", "scale_data("):
            self.assertNotIn(disallowed, source)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")
                self.assertGreater(cell.get("execution_count", 0), 0)
                self.assertTrue(cell.get("outputs"))
                self.assertFalse(any(output.get("output_type") == "error" for output in cell["outputs"]))


if __name__ == "__main__":
    unittest.main()
