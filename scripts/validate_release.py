"""Offline regression suite plus execution of the committed educational notebook."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib_cache")
    import nbformat
    import pandas as pd
    from pandas.testing import assert_frame_equal
    import IPython.display
    import matplotlib.pyplot as plt

    notebook = root / "volatility_prediction.ipynb"
    digest = hashlib.sha256(notebook.read_bytes()).hexdigest()
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    namespace = {"__name__": "__main__"}
    log = io.StringIO()
    executed = []

    def blocked(*args, **kwargs):
        raise RuntimeError("Network is disabled during validation")

    def display_local(*objects, **kwargs):
        for obj in objects:
            print(obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else type(obj).__name__)

    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), patch.object(socket, "create_connection", blocked), patch.object(IPython.display, "display", display_local), redirect_stdout(log), redirect_stderr(log):
        suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_predicting_volatility_*.py")
        test_result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        if not test_result.wasSuccessful():
            (output / "validation_log.txt").write_text(log.getvalue(), encoding="utf-8")
            raise RuntimeError("Tests failed; inspect validation_log.txt")
        for number, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                exec(compile(cell.source, f"notebook_cell_{number}", "exec"), namespace)
                executed.append(number)
    plt.close("all")
    result = namespace["result"]
    if result["plan"]["plan_id"] != nb.metadata.volatility_lab.plan_id:
        raise RuntimeError("Notebook outputs are stale relative to code/config/data hashes")
    expected = pd.read_csv(root / "docs/results/synthetic_metrics.csv")
    assert_frame_equal(expected, result["tables"]["metrics"], check_exact=False, rtol=1e-9, atol=1e-12)
    if hashlib.sha256(notebook.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Validator must not mutate the notebook")
    (output / "validation_log.txt").write_text(log.getvalue(), encoding="utf-8")
    summary = {"status": "passed", "tests": test_result.testsRun, "skipped": len(test_result.skipped), "notebook_code_cells": len(executed), "network_blocked": True, "notebook_sha256": digest, "plan_id": result["plan"]["plan_id"], "synthetic_tables_match": True, "historical_model_retuned": False}
    (output / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
