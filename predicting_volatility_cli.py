"""CLI: python -m predicting_volatility_cli --help (no work on import)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from predicting_volatility_data import load_csv, synthetic_prices
from predicting_volatility_experiment import ExperimentConfig, make_plan, run_experiment, validate_plan, write_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Purged temporal volatility experiments, offline by design.")
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--config", type=Path, help="JSON configuration; defaults to a fixed synthetic demo")
    parser.add_argument("--data", type=Path, help="Local CSV obeying docs/data-contract.md")
    parser.add_argument("--metadata", type=Path, help="Required provenance JSON for --data")
    parser.add_argument("--rows", type=int, default=900, help="Synthetic price observations only")
    parser.add_argument("--plan", type=Path, help="plan: new plan file; run: previously saved matching plan")
    parser.add_argument("--output", type=Path, default=Path("runs"), help="Parent folder of immutable versioned run directories")
    parser.add_argument("--plot", action="store_true", help="Also render a PNG; requires the plots extra")
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(args.config) if args.config else ExperimentConfig()
        if bool(args.data) != bool(args.metadata):
            raise ValueError("Supply both --data and --metadata, or neither for synthetic data")
        prices = load_csv(args.data, args.metadata) if args.data else synthetic_prices(args.rows, config.seed)
        if args.command == "plan":
            if args.plan is None:
                raise ValueError("plan requires --plan PATH")
            plan = make_plan(prices, config)
            args.plan.parent.mkdir(parents=True, exist_ok=True)
            with args.plan.open("x", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2)
                handle.write("\n")
            print(f"Prepared {plan['plan_id']} without fitting a model.")
            return 0
        plan = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else None
        if plan is None and args.data:
            raise ValueError("Local observed input requires a saved --plan before run")
        plan = make_plan(prices, config) if plan is None else plan
        validate_plan(plan, prices, config)
        destination = args.output / f"{config.experiment_id}--{plan['plan_id'][:12]}"
        if destination.exists():
            raise FileExistsError("This exact run already exists; choose another output parent to reproduce it")
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")  # CLI file rendering must not open a GUI.
        result = run_experiment(prices, config, plan)
        destination = write_run(result, args.output, include_plot=args.plot)
        print(result["tables"]["metrics"].to_string(index=False))
        print(f"Saved {destination.name}; {prices.metadata['data_kind']} input; target units are return fractions.")
        return 0
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
