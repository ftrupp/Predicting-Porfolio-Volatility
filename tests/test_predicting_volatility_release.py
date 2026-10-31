"""Public API, provenance, planning and artifact regressions, all offline."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from predicting_volatility_cli import main
from predicting_volatility_data import load_csv, synthetic_prices, validate_prices
from predicting_volatility_experiment import ExperimentConfig, make_plan, run_experiment, validate_plan, write_run
import predicting_volatility_temporal as vt


def small_config(**changes):
    return ExperimentConfig(rolling_windows=0, final_validation_size=50, final_test_size=50, inner_folds=2, **changes)


@contextlib.contextmanager
def scratch_directory():
    # Keep test writes in the checkout, including on restricted Windows hosts.
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix=".release-test-", dir=root) as folder:
        if not Path(folder).resolve().is_relative_to(root):
            raise RuntimeError("Test temporary directory escaped its intended root")
        yield folder


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_prices(300, 17)
        self.frame = pd.DataFrame({"Date": self.data.close.index, "Close": self.data.close.to_numpy(), "AvailableAt": self.data.available_at.to_numpy()})

    def test_synthetic_generator_is_deterministic_and_labelled(self):
        same = synthetic_prices(300, 17)
        assert_series_equal(self.data.close, same.close)
        self.assertEqual(self.data.fingerprint, same.fingerprint)
        self.assertNotEqual(self.data.fingerprint, synthetic_prices(300, 18).fingerprint)
        self.assertEqual(same.metadata["data_kind"], "synthetic")
        self.assertEqual(same.metadata["currency"], "SYNTHETIC_UNITS")

    def test_local_csv_round_trip_requires_metadata_and_never_unpickles(self):
        with scratch_directory() as temp:
            path = Path(temp)
            self.frame.to_csv(path / "prices.csv", index=False)
            (path / "source.json").write_text(json.dumps(self.data.metadata))
            got = load_csv(path / "prices.csv", path / "source.json")
            np.testing.assert_allclose(got.close, self.data.close)
            self.assertEqual(got.close.index.tolist(), self.data.close.index.tolist())
            (path / "source.json").write_text('{}')
            with self.assertRaises(ValueError):
                load_csv(path / "prices.csv", path / "source.json")

    def test_future_availability_and_naive_times_are_rejected(self):
        late = self.frame.copy()
        late.loc[2, "AvailableAt"] = late.loc[2, "Date"] + pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError):
            validate_prices(late, self.data.metadata, "test")
        naive = self.frame.copy()
        naive["Date"] = naive.Date.dt.tz_localize(None)
        with self.assertRaises(ValueError):
            validate_prices(naive, self.data.metadata, "test")

    def test_bad_rows_fail_and_missing_prices_are_not_filled(self):
        duplicate = self.frame.copy()
        duplicate.loc[2, "Date"] = duplicate.loc[1, "Date"]
        for frame in (duplicate, self.frame.iloc[::-1], self.frame.assign(Close=np.inf)):
            with self.assertRaises(ValueError):
                validate_prices(frame, self.data.metadata, "test")
        missing = self.frame.copy()
        missing.loc[30, "Close"] = np.nan
        got = validate_prices(missing, self.data.metadata, "test")
        self.assertTrue(pd.isna(got.close.iloc[30]))


class ProtocolTests(unittest.TestCase):
    def test_plan_does_not_fit_and_changes_in_input_or_config_invalidate_it(self):
        data, cfg = synthetic_prices(300), small_config()
        with patch("predicting_volatility_experiment.evaluate_split", side_effect=AssertionError("fit called")):
            plan = make_plan(data, cfg)
        with self.assertRaises(ValueError):
            validate_plan(plan, synthetic_prices(300, 43), cfg)
        with self.assertRaises(ValueError):
            validate_plan(plan, data, small_config(horizon=5))
        changed = json.loads(json.dumps(plan))
        changed["protocol"]["config"]["seed"] = 99
        with self.assertRaises(ValueError):
            validate_plan(changed, data, cfg)

    def test_observed_input_requires_plan_and_explicit_selection(self):
        data = synthetic_prices(300)
        data.metadata = dict(data.metadata, data_kind="observed", symbol="EXAMPLE")
        cfg = small_config(purpose="historical_reproduction", selection_note="Already inspected historical example; no new holdout claim.")
        with self.assertRaises(ValueError):
            run_experiment(data, cfg)
        plan = make_plan(data, cfg)
        self.assertEqual(plan["protocol"]["holdout_status"], "already_seen")
        with self.assertRaises(ValueError):
            small_config(purpose="new_hypothesis")

    def test_seen_aapl_period_is_rejected_as_new_holdout(self):
        data = synthetic_prices(300)
        data.metadata = dict(data.metadata, data_kind="observed", symbol="AAPL")
        cfg = small_config(purpose="new_hypothesis", selection_note="A new test is being declared for a regression test.", validation_start=data.close.index[150].isoformat(), test_start=data.close.index[220].isoformat(), previously_observed_through=data.close.index[100].isoformat())
        with self.assertRaisesRegex(ValueError, "previously observed"):
            make_plan(data, cfg)

    def test_new_period_must_follow_researchers_seen_boundary(self):
        data = synthetic_prices(300)
        data.metadata = dict(data.metadata, data_kind="observed", symbol="EXAMPLE")
        cfg = small_config(purpose="new_hypothesis", selection_note="Fixed protocol reserved after previously inspected history.", validation_start=data.close.index[150].isoformat(), test_start=data.close.index[220].isoformat(), previously_observed_through=data.close.index[219].isoformat())
        plan = make_plan(data, cfg)
        self.assertEqual(plan["protocol"]["holdout_status"], "declared_new_by_researcher")

    def test_features_seed_and_extra_purge_are_effective(self):
        data = synthetic_prices(350)
        cfg = small_config(seed=17, features=("volatility_10obs",), purge_extra_days=3)
        result = run_experiment(data, cfg)
        self.assertEqual(list(result["dataset"].X.columns), ["volatility_10obs"])
        model = result["evaluations"]["final"]["model"]
        self.assertEqual(model.named_steps["select"].score_func.keywords["random_state"], 17)
        outer = result["tables"]["outer_audit"]
        for row in outer.iloc[:2].itertuples():
            self.assertLess(row.last_label_price, row.exclusive_cutoff)
        self.assertTrue(result["tables"]["feature_availability"]["available"].all())


class ArtifactAndCLITests(unittest.TestCase):
    def test_public_modules_import_without_network_or_file_writes(self):
        code = '''
import builtins, socket
from pathlib import Path
from unittest.mock import patch
original_open = builtins.open
def guarded_open(file, mode="r", *args, **kwargs):
    if any(flag in mode for flag in ("w", "a", "+", "x")):
        raise RuntimeError("write on import")
    return original_open(file, mode, *args, **kwargs)
def blocked(*args, **kwargs):
    raise RuntimeError("side effect on import")
with patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked), patch.object(builtins, "open", guarded_open), patch.object(Path, "mkdir", blocked):
    import predicting_volatility_temporal
    import predicting_volatility_data
    import predicting_volatility_experiment
    import predicting_volatility_cli
'''
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=45)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_archived_history_retains_every_mixed_partition(self):
        path = Path(__file__).resolve().parents[1] / "docs/results/aapl_historical_metrics.csv"
        history = pd.read_csv(path)
        self.assertEqual(len(history), 24)
        pivot = history.pivot(index=["experiment", "partition"], columns="model", values="rmse")
        self.assertGreater(pivot.loc[("rolling_3", "test"), "model"], pivot.loc[("rolling_3", "test"), "trailing_volatility"])
        self.assertGreater(pivot.loc[("final", "validation"), "model"], pivot.loc[("final", "validation"), "trailing_volatility"])
        ratio = pivot.loc[("final", "test"), "model"] / pivot.loc[("final", "test"), "trailing_volatility"] - 1
        self.assertAlmostEqual(100 * ratio, -24.52982308641276)

    def test_versioned_outputs_do_not_overwrite_and_preserve_all_references(self):
        result = run_experiment(synthetic_prices(300), small_config())
        with scratch_directory() as temp:
            output = write_run(result, temp)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["data_kind"], "synthetic")
            self.assertEqual(manifest["result_kind"], "derived_forecast_errors")
            self.assertIn("metrics.csv", manifest["files_sha256"])
            with self.assertRaises(FileExistsError):
                write_run(result, temp)
            metrics = pd.read_csv(output / "metrics.csv")
            self.assertEqual(set(metrics["model"]), {"model", "train_mean", "trailing_volatility"})
            self.assertEqual(set(metrics["partition"]), {"validation", "test"})

    def test_repeated_runs_are_numerically_identical(self):
        data, cfg = synthetic_prices(300), small_config()
        first, second = run_experiment(data, cfg), run_experiment(data, cfg)
        assert_frame_equal(first["tables"]["metrics"], second["tables"]["metrics"])
        assert_frame_equal(first["tables"]["predictions"], second["tables"]["predictions"])

    def test_cli_plan_and_run_use_saved_configuration(self):
        from dataclasses import asdict
        with scratch_directory() as temp, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temp)
            config = folder / "config.json"
            config.write_text(json.dumps(asdict(small_config())))
            plan = folder / "plan.json"
            common = ["--config", str(config), "--rows", "300", "--plan", str(plan)]
            self.assertEqual(main(["plan", *common]), 0)
            self.assertEqual(main(["run", *common, "--output", str(folder / "runs")]), 0)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(["run", *common, "--output", str(folder / "runs")])
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
