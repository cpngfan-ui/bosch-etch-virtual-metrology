"""Tests for command-line configuration precedence."""

from __future__ import annotations

from pathlib import Path

from bosch_vm.cli import _config_from_args, build_parser


def test_cli_paths_override_nested_project_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
seed: 17
paths:
  raw_dir: yaml/raw
  processed_dir: yaml/processed
  results_dir: yaml/results
  experiment_dir: yaml/experiment
task:
  target_name: mean_si_etch_um
  primary_model: cycle_ridge
validation:
  selection_metric: neg_mean_absolute_error
""".strip(),
        encoding="utf-8",
    )
    cli_prepared = tmp_path / "cli-prepared"
    cli_results = tmp_path / "cli-results"
    cli_experiment = tmp_path / "cli-experiment"

    args = build_parser().parse_args(
        [
            "run",
            "--config",
            str(config_path),
            "--prepared-dir",
            str(cli_prepared),
            "--output-dir",
            str(cli_results),
            "--experiment-dir",
            str(cli_experiment),
            "--seed",
            "29",
            "--n-jobs",
            "3",
            "--force",
        ]
    )

    config = _config_from_args(args)

    assert config.raw_dir == Path("yaml/raw")
    assert config.prepared_dir == cli_prepared
    assert config.output_dir == cli_results
    assert config.experiment_dir == cli_experiment
    assert config.seed == 29
    assert config.n_jobs == 3
    assert config.overwrite is True


def test_prepare_cli_raw_path_overrides_nested_project_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  raw_dir: yaml/raw
  processed_dir: yaml/processed
  experiment_dir: yaml/experiment
""".strip(),
        encoding="utf-8",
    )
    cli_raw = tmp_path / "cli-raw"

    args = build_parser().parse_args(
        ["prepare", "--config", str(config_path), "--raw-dir", str(cli_raw)]
    )

    config = _config_from_args(args)

    assert config.raw_dir == cli_raw
    assert config.prepared_dir == Path("yaml/processed")
    assert config.experiment_dir == Path("yaml/experiment")
