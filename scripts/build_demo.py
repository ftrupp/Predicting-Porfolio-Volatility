"""Regenerate the public notebook, synthetic tables and both research figures.

Executes only the built-in synthetic presentation below, with network blocked.
Historical AAPL aggregate errors are plotted, never re-tuned.
"""
from contextlib import redirect_stdout, redirect_stderr
import base64
import io
import json
import os
from pathlib import Path
import socket
import sys
from textwrap import dedent
from unittest.mock import patch


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = str(root / ".matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import nbformat as nbf
    import pandas as pd
    import IPython.display
    from predicting_volatility_experiment import plot_final

    cells = []

    def add(kind, source):
        make = nbf.v4.new_markdown_cell if kind == "markdown" else nbf.v4.new_code_cell
        cells.append(make(dedent(source).strip(), id=f"volatility-{len(cells):02d}"))

    add("markdown", """
    # Temporal volatility forecasting

    This notebook presents the same tested core used by `volatility-lab`.
    **All newly executed data are synthetic.** The archived AAPL results are
    shown separately with their losing periods. Run from the repository root
    after installing `.[notebook]`. No key, account or download is needed.
    """)
    add("code", """
    import pandas as pd
    from IPython.display import display
    from predicting_volatility_data import synthetic_prices
    from predicting_volatility_experiment import ExperimentConfig, make_plan, run_experiment, plot_final
    config = ExperimentConfig.from_json('examples/synthetic.json')
    prices = synthetic_prices(n=900, seed=config.seed)
    print('Input:', prices.metadata['data_kind'], '| Units:', prices.metadata['currency'])
    """)
    add("markdown", r"""
    ## Define the target and availability

    After the close at t is known, predict
    $y_t=\mathrm{std}(r_{t+1},\ldots,r_{t+h})$, with
    $r_t=P_t/P_{t-1}-1$ and `ddof=1`. Each target uses prices t through t+h.
    The horizon counts source observations, not calendar days. Values remain
    return fractions per observation: 0.01 means 1%, without annualization.

    The fixture has predeclared variance regimes and artificial weekday dates.
    It is not an observed market dataset or a calibrated investment model.
    """)
    add("code", """
    display(pd.Series({
        'first_origin': prices.close.index.min(), 'last_origin': prices.close.index.max(),
        'observations': len(prices.close), 'horizon': config.horizon, 'seed': config.seed,
        'prices_available_by_forecast': bool((prices.available_at <= prices.close.index).all())
    }))
    """)
    add("markdown", """
    ## Fix the protocol before fitting

    A plan fixes configuration, dates, features, references and data/code hashes
    without fitting. Training labels must finish strictly before the next
    validation origin. The same rule separates validation/test and internal folds.
    Dates are retained from the original price calendar before incomplete samples
    are removed. Targets are never imputed.
    """)
    add("code", """
    plan = make_plan(prices, config)
    display(pd.Series({key: plan['protocol'][key] for key in (
        'holdout_status', 'validation_start', 'test_start',
        'test_last_prediction', 'test_last_label_price', 'target_unit'
    )}))
    """)
    add("markdown", """
    ## Train within each fold and audit the boundaries

    Each fold fits its own `median imputation → robust scaling → MI selection →
    model` pipeline. The bounded Ridge grid has four candidates and three inner
    folds. XGBoost can use the same workflow through an optional dependency.
    Parameters are selected only with inner CV. After reporting validation,
    refit uses labels already matured before test; test labels cannot enter fit.
    """)
    add("code", """
    result = run_experiment(prices, config, plan)
    display(result['tables']['outer_audit'])
    display(result['tables']['inner_audit'])
    display(result['tables']['refit_audit'])
    assert result['tables']['feature_availability']['available'].all()
    """)
    add("markdown", """
    ## Report every partition and both references

    Trailing-horizon volatility and mean training volatility receive exactly
    the same dates and target units. The table reports errors as fractions;
    the chart multiplies volatility by 100. Synthetic metrics verify the workflow
    and do not establish market performance.
    """)
    add("code", """
    display(result['tables']['metrics'])
    display(plot_final(result))
    """)
    add("markdown", """
    ## Keep the historical evidence mixed

    This table redisplays frozen AAPL errors, without prices or retraining.
    The 24.53% improvement in final-test RMSE was one outcome. The third rolling
    test and final validation were worse than trailing volatility. Adjustment
    provenance and redistribution rights for that source cache were not verified.
    """)
    add("code", """
    historical = pd.read_csv('docs/results/aapl_historical_metrics.csv')
    display(historical)
    """)
    add("markdown", """
    ## Interpretation and new research

    Adjacent labels overlap inside each block; errors remain dependent. No iid
    confidence claim, return, cost or alpha is inferred. The AAPL period is already
    seen. New hypotheses need new predeclared test dates and selection rules.
    The program cannot authenticate a supplier's timestamps or know everything
    the researcher inspected elsewhere.

    The CLI writes immutable versioned runs with plans, hashes, predictions and
    audit tables. See `docs/method.md`, `docs/data-contract.md`, and the complete
    `docs/historical-results.md`. No external data are needed to rebuild this demo.
    """)
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "volatility_lab": {"data_kind": "synthetic", "project_version": "0.2.0", "historical_results": "archived_not_retuned"}})
    current, namespace = None, {"__name__": "__main__"}

    def display_local(*objects, **kwargs):
        for obj in objects:
            if isinstance(obj, Figure):
                buffer = io.BytesIO()
                obj.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
                payload = {"image/png": base64.b64encode(buffer.getvalue()).decode(), "text/plain": "Synthetic final holdout volatility"}
                plt.close(obj)
            elif isinstance(obj, (pd.DataFrame, pd.Series)):
                payload = {"text/plain": obj.to_string()}
                if isinstance(obj, pd.DataFrame):
                    payload["text/html"] = obj.to_html(index=False)
            else:
                payload = {"text/plain": str(obj)}
            current.outputs.append(nbf.v4.new_output("display_data", data=payload))

    def blocked(*args, **kwargs):
        raise RuntimeError("Network is disabled during public demo generation")

    count = 0
    with patch.object(socket.socket, "connect", blocked), patch.object(socket.socket, "connect_ex", blocked), patch.object(socket, "create_connection", blocked), patch.object(IPython.display, "display", display_local):
        for current in nb.cells:
            if current.cell_type == "code":
                count += 1
                current.execution_count = count
                capture = io.StringIO()
                with redirect_stdout(capture), redirect_stderr(capture):
                    exec(compile(current.source, f"synthetic_cell_{count}", "exec"), namespace)
                if capture.getvalue():
                    current.outputs.append(nbf.v4.new_output("stream", name="stdout", text=capture.getvalue()))
    result = namespace["result"]
    nb.metadata.volatility_lab["plan_id"] = result["plan"]["plan_id"]
    nbf.validate(nb)
    (root / "volatility_prediction.ipynb").write_text(nbf.writes(nb) + "\n", encoding="utf-8", newline="\n")
    output = root / "docs" / "results"
    output.mkdir(parents=True, exist_ok=True)
    result["tables"]["metrics"].to_csv(output / "synthetic_metrics.csv", index=False, float_format="%.12g", lineterminator="\n")
    result["tables"]["outer_audit"].to_csv(output / "synthetic_outer_audit.csv", index=False, lineterminator="\n")
    (output / "synthetic_plan.json").write_text(json.dumps(result["plan"], indent=2) + "\n", encoding="utf-8", newline="\n")
    plot_final(result, output / "synthetic_final_holdout.png")
    historical = namespace["historical"].copy()
    historical["partition_label"] = historical["experiment"] + " / " + historical["partition"]
    pivot = historical.pivot(index="partition_label", columns="model", values="rmse") * 100
    ax = pivot[["model", "trailing_volatility", "train_mean"]].plot.barh(figsize=(10, 6), width=0.75)
    ax.set(xlabel="RMSE (percentage points of per-observation volatility)", ylabel="", title="AAPL historical study — all reported partitions")
    ax.legend(["XGBoost", "Trailing volatility", "Training mean"])
    plt.tight_layout()
    ax.figure.savefig(output / "aapl_historical_errors.png", dpi=140)
    plt.close(ax.figure)
    print(f"Regenerated {count} synthetic code cells, public tables and figures. No historical model was refit.")


if __name__ == "__main__":
    main()
