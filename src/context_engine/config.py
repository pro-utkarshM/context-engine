"""Experiment config loading for reproducible benchmark runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .evaluation import ScoringWeights


REQUIRED_KEYS: tuple[str, ...] = (
    "experiment_name",
    "model_name",
    "selector_strategy",
    "evaluator_version",
)


DEFAULT_DATASET_DIR = Path("data/processed")
DEFAULT_ARTIFACT_VERSION = "v1"
DEFAULT_TOKEN_BUDGET = 1500
DEFAULT_SEED = 1337


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_name: str
    model_name: str
    selector_strategy: str
    evaluator_version: str
    dataset_dir: Path = DEFAULT_DATASET_DIR
    artifact_version: str = DEFAULT_ARTIFACT_VERSION
    token_budget: int = DEFAULT_TOKEN_BUDGET
    scoring_weights: ScoringWeights = field(default_factory=ScoringWeights)
    seed: int = DEFAULT_SEED
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        missing = [key for key in REQUIRED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"experiment config missing required keys: {sorted(missing)}")

        weights_payload = payload.get("scoring_weights") or {}
        defaults = ScoringWeights()
        weights = ScoringWeights(
            correctness=float(weights_payload.get("correctness", defaults.correctness)),
            support=float(weights_payload.get("support", defaults.support)),
            efficiency=float(weights_payload.get("efficiency", defaults.efficiency)),
        )

        known = {
            "experiment_name",
            "model_name",
            "selector_strategy",
            "evaluator_version",
            "dataset_dir",
            "artifact_version",
            "token_budget",
            "scoring_weights",
            "seed",
        }
        extra = {key: value for key, value in payload.items() if key not in known}

        return cls(
            experiment_name=str(payload["experiment_name"]),
            model_name=str(payload["model_name"]),
            selector_strategy=str(payload["selector_strategy"]),
            evaluator_version=str(payload["evaluator_version"]),
            dataset_dir=Path(payload.get("dataset_dir", DEFAULT_DATASET_DIR)),
            artifact_version=str(payload.get("artifact_version", DEFAULT_ARTIFACT_VERSION)),
            token_budget=int(payload.get("token_budget", DEFAULT_TOKEN_BUDGET)),
            scoring_weights=weights,
            seed=int(payload.get("seed", DEFAULT_SEED)),
            extra=extra,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "selector_strategy": self.selector_strategy,
            "evaluator_version": self.evaluator_version,
            "dataset_dir": str(self.dataset_dir),
            "artifact_version": self.artifact_version,
            "token_budget": self.token_budget,
            "scoring_weights": {
                "correctness": self.scoring_weights.correctness,
                "support": self.scoring_weights.support,
                "efficiency": self.scoring_weights.efficiency,
            },
            "seed": self.seed,
            **self.extra,
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment config at {path} must be a JSON object")
    return ExperimentConfig.from_dict(payload)


def resolved_artifact_path(
    config: ExperimentConfig,
    artifact_name: str,
    *,
    dataset_dir_override: Path | None = None,
    artifact_version_override: str | None = None,
) -> Path:
    dataset_dir = dataset_dir_override if dataset_dir_override is not None else config.dataset_dir
    version = artifact_version_override if artifact_version_override is not None else config.artifact_version
    return Path(dataset_dir) / f"{artifact_name}_{version}.jsonl"


def build_default_config(
    *,
    experiment_name: str = "ad_hoc",
    model_name: str = "stub",
    selector_strategy: str = "default",
    evaluator_version: str = "eval_v1",
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    artifact_version: str = DEFAULT_ARTIFACT_VERSION,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=experiment_name,
        model_name=model_name,
        selector_strategy=selector_strategy,
        evaluator_version=evaluator_version,
        dataset_dir=Path(dataset_dir),
        artifact_version=artifact_version,
    )


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Attach shared --config / --dataset-dir / --artifact-version flags."""
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON experiment config file.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Directory containing versioned JSONL artifacts. Overrides config.dataset_dir.",
    )
    parser.add_argument(
        "--artifact-version",
        default=None,
        help="Artifact major version suffix (e.g. 'v1'). Overrides config.artifact_version.",
    )


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Resolve an ExperimentConfig from CLI args, applying overrides last."""
    if getattr(args, "config", None):
        config = load_experiment_config(args.config)
    else:
        config = build_default_config()

    dataset_dir = getattr(args, "dataset_dir", None)
    artifact_version = getattr(args, "artifact_version", None)
    if dataset_dir is None and artifact_version is None:
        return config

    return ExperimentConfig(
        experiment_name=config.experiment_name,
        model_name=config.model_name,
        selector_strategy=config.selector_strategy,
        evaluator_version=config.evaluator_version,
        dataset_dir=Path(dataset_dir) if dataset_dir is not None else config.dataset_dir,
        artifact_version=artifact_version if artifact_version is not None else config.artifact_version,
        token_budget=config.token_budget,
        scoring_weights=config.scoring_weights,
        seed=config.seed,
        extra=dict(config.extra),
    )
