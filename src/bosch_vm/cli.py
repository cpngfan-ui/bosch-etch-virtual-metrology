"""Command-line interface for preparing and running nested-LOLO experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .experiment import ExperimentConfig, prepare, run
from .sensitivity import run_sensitivity_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bosch-vm",
        description=(
            "Prepare wafer-level features and run lot-grouped nested "
            "leave-one-lot-out virtual-metrology experiments."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Build the canonical one-row-per-wafer feature table."
    )
    _add_shared_config_argument(prepare_parser)
    prepare_parser.add_argument("--raw-dir", type=Path)
    prepare_parser.add_argument("--prepared-dir", type=Path)
    prepare_parser.add_argument("--experiment-dir", type=Path)
    prepare_parser.add_argument("--force", action="store_true", help="Overwrite known outputs.")

    run_parser = subparsers.add_parser(
        "run", help="Run outer and inner leave-one-lot-out model evaluation."
    )
    _add_shared_config_argument(run_parser)
    run_parser.add_argument("--prepared-dir", type=Path)
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--experiment-dir", type=Path)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--n-jobs", type=int)
    run_parser.add_argument("--bootstrap-replicates", type=int)
    run_parser.add_argument("--permutation-repetitions", type=int)
    run_parser.add_argument("--force", action="store_true", help="Overwrite known outputs.")

    sensitivity_parser = subparsers.add_parser(
        "sensitivity",
        help="Run the Ridge input and measurement robustness suite.",
    )
    _add_shared_config_argument(sensitivity_parser)
    sensitivity_parser.add_argument("--raw-dir", type=Path)
    sensitivity_parser.add_argument("--prepared-dir", type=Path)
    sensitivity_parser.add_argument("--output-dir", type=Path)
    sensitivity_parser.add_argument("--experiment-dir", type=Path)
    sensitivity_parser.add_argument("--seed", type=int)
    sensitivity_parser.add_argument("--n-jobs", type=int)
    sensitivity_parser.add_argument("--bootstrap-replicates", type=int)
    sensitivity_parser.add_argument(
        "--force", action="store_true", help="Overwrite known sensitivity outputs."
    )

    for subparser in (prepare_parser, run_parser, sensitivity_parser):
        subparser.add_argument("--id-column")
        subparser.add_argument("--lot-column")
        subparser.add_argument("--wafer-column")
        subparser.add_argument("--target-column")
    run_parser.add_argument("--primary-model-id")
    return parser


def _add_shared_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Optional YAML/JSON configuration. Keys may be top-level or under "
            "the selected command name. CLI flags take precedence."
        ),
    )


def _read_config(path: Path | None, command: str) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration file must contain a mapping.")
    common = payload.get("experiment", {})
    command_values = payload.get(command, {})
    if common and not isinstance(common, dict):
        raise ValueError("The 'experiment' configuration section must be a mapping.")
    if command_values and not isinstance(command_values, dict):
        raise ValueError(f"The {command!r} configuration section must be a mapping.")
    if "experiment" in payload or command in payload:
        return {**common, **command_values}
    return payload


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_mapping(_read_config(args.config, args.command))
    overrides = {
        "raw_dir": getattr(args, "raw_dir", None),
        "prepared_dir": getattr(args, "prepared_dir", None),
        "output_dir": getattr(args, "output_dir", None),
        "experiment_dir": getattr(args, "experiment_dir", None),
        "seed": getattr(args, "seed", None),
        "n_jobs": getattr(args, "n_jobs", None),
        "bootstrap_replicates": getattr(args, "bootstrap_replicates", None),
        "permutation_repetitions": getattr(args, "permutation_repetitions", None),
        "id_column": getattr(args, "id_column", None),
        "lot_column": getattr(args, "lot_column", None),
        "wafer_column": getattr(args, "wafer_column", None),
        "target_column": getattr(args, "target_column", None),
        "primary_model_id": getattr(args, "primary_model_id", None),
    }
    resolved_overrides = {
        key: value for key, value in overrides.items() if value is not None
    }
    if args.force:
        resolved_overrides["overwrite"] = True
    return replace(config, **resolved_overrides)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "prepare":
            artifacts = prepare(config)
            print(f"Prepared wafer-level feature table: {artifacts.feature_table_path}")
        elif args.command == "run":
            artifacts = run(config)
            print(f"Completed nested LOLO comparison: {artifacts.comparison_path}")
        elif args.command == "sensitivity":
            summary_path = run_sensitivity_suite(config, overwrite=args.force)
            print(f"Completed sensitivity suite: {summary_path}")
        else:  # pragma: no cover - argparse enforces the subcommand choices
            parser.error(f"Unknown command: {args.command}")
        return 0
    except Exception as exc:  # CLI boundary: preserve the original exception in library use.
        parser.exit(status=1, message=f"bosch-vm: error: {exc}\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
