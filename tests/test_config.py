from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from context_engine.config import (
    ExperimentConfig,
    add_config_args,
    config_from_args,
    load_experiment_config,
    resolved_artifact_path,
)
from context_engine.evaluation import ScoringWeights


def _full_payload() -> dict:
    return {
        "experiment_name": "test_exp",
        "model_name": "gpt-test",
        "selector_strategy": "gold_only",
        "evaluator_version": "eval_v1_test",
        "dataset_dir": "data/processed",
        "artifact_version": "v1",
        "token_budget": 1500,
        "scoring_weights": {"correctness": 0.6, "support": 0.3, "efficiency": 0.1},
        "seed": 1337,
    }


def test_from_dict_builds_full_config() -> None:
    config = ExperimentConfig.from_dict(_full_payload())
    assert config.experiment_name == "test_exp"
    assert config.dataset_dir == Path("data/processed")
    assert config.artifact_version == "v1"
    assert config.token_budget == 1500
    assert config.scoring_weights == ScoringWeights(correctness=0.6, support=0.3, efficiency=0.1)
    assert config.seed == 1337


def test_from_dict_applies_defaults_for_optional_keys() -> None:
    payload = {
        "experiment_name": "exp",
        "model_name": "m",
        "selector_strategy": "s",
        "evaluator_version": "e",
    }
    config = ExperimentConfig.from_dict(payload)
    assert config.dataset_dir == Path("data/processed")
    assert config.artifact_version == "v1"
    assert config.token_budget == 1500
    assert config.seed == 1337
    assert config.scoring_weights == ScoringWeights()


def test_from_dict_raises_on_missing_required_keys() -> None:
    payload = {"experiment_name": "x"}
    with pytest.raises(ValueError) as excinfo:
        ExperimentConfig.from_dict(payload)
    assert "missing required keys" in str(excinfo.value)


def test_from_dict_preserves_extra_fields() -> None:
    payload = _full_payload()
    payload["learned_model_path"] = "models/foo.json"
    config = ExperimentConfig.from_dict(payload)
    assert config.extra["learned_model_path"] == "models/foo.json"


def test_load_experiment_config_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "exp.json"
    target.write_text(json.dumps(_full_payload()))
    config = load_experiment_config(target)
    assert config.experiment_name == "test_exp"


def test_load_experiment_config_rejects_non_object(tmp_path: Path) -> None:
    target = tmp_path / "exp.json"
    target.write_text("[]")
    with pytest.raises(ValueError):
        load_experiment_config(target)


def test_resolved_artifact_path_uses_dataset_dir_and_version() -> None:
    config = ExperimentConfig.from_dict(_full_payload())
    path = resolved_artifact_path(config, "corpus_chunks")
    assert path == Path("data/processed/corpus_chunks_v1.jsonl")


def test_resolved_artifact_path_overrides() -> None:
    config = ExperimentConfig.from_dict(_full_payload())
    path = resolved_artifact_path(
        config,
        "outcomes",
        dataset_dir_override=Path("/tmp/exp"),
        artifact_version_override="v2",
    )
    assert path == Path("/tmp/exp/outcomes_v2.jsonl")


def test_config_from_args_loads_file_when_provided(tmp_path: Path) -> None:
    payload = _full_payload()
    payload["experiment_name"] = "from_file"
    config_path = tmp_path / "exp.json"
    config_path.write_text(json.dumps(payload))

    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(["--config", str(config_path)])

    config = config_from_args(args)
    assert config.experiment_name == "from_file"


def test_config_from_args_applies_overrides(tmp_path: Path) -> None:
    payload = _full_payload()
    config_path = tmp_path / "exp.json"
    config_path.write_text(json.dumps(payload))

    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args(
        ["--config", str(config_path), "--dataset-dir", "/alt/dir", "--artifact-version", "v2"]
    )

    config = config_from_args(args)
    assert config.dataset_dir == Path("/alt/dir")
    assert config.artifact_version == "v2"


def test_config_from_args_falls_back_to_defaults_when_no_config() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    args = parser.parse_args([])
    config = config_from_args(args)
    assert config.dataset_dir == Path("data/processed")
    assert config.artifact_version == "v1"
