"""Tests for scripts/analyze_results.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from context_engine.replications import (
    ReplicationPairedDelta,
    ReplicationStrategySummary,
    ReplicationSummary,
    summarize_replications,
)
from context_engine.stats import DistributionSummary, PairedDeltaSummary


def _context_set(set_id, query_id, strategy):
    from context_engine.artifacts import ContextSet
    return ContextSet.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "candidate_pool_id": f"pool_{query_id}",
            "strategy": strategy,
            "selected_ids": ["c1"],
            "ordering_type": "best_first",
            "token_count": 100,
            "metadata": {
                "contains_all_gold": True,
                "missing_gold_count": 0,
                "distractor_types": [],
            },
        }
    )


def _outcome(set_id, query_id, overall):
    from context_engine.artifacts import Outcome
    return Outcome.from_dict(
        {
            "set_id": set_id,
            "query_id": query_id,
            "answer": "ans",
            "scores": {"correctness": overall, "support": overall, "overall": overall},
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "latency_ms": 0,
            "evaluator_version": "eval_v1",
        }
    )


def _write_summary(path: Path, summary: ReplicationSummary) -> None:
    path.write_text(json.dumps(summary.to_dict()) + "\n", encoding="utf-8")


def _build_summary():
    """3 runs of 2 strategies × 4 queries, with a paired delta."""
    runs = [
        (
            [_context_set(f"q{i}_{strategy}", f"q{i}", strategy) for strategy in ("gold_only", "shuffled_order") for i in range(4)],
            [_outcome(f"q{i}_{strategy}", f"q{i}", 0.6 + 0.05 * i + 0.02 * r) for strategy in ("gold_only", "shuffled_order") for i in range(4)],
        )
        for r in range(3)
    ]
    summary = summarize_replications(
        runs,
        experiment_name="exp",
        runner="minimax",
        model_name="MiniMax-M3",
        artifact_version="v1",
        seed=0,
    )
    # Add a paired section.
    left = (
        [_context_set(f"q{i}_gold_only", f"q{i}", "gold_only") for i in range(4)],
        [_outcome(f"q{i}_gold_only", f"q{i}", 0.7) for i in range(4)],
    )
    right = (
        [_context_set(f"q{i}_gold_only", f"q{i}", "gold_only") for i in range(4)],
        [_outcome(f"q{i}_gold_only", f"q{i}", 0.5) for i in range(4)],
    )
    from context_engine.replications import paired_summary
    pairs = paired_summary([left], [right], left_pool_source="auto", right_pool_source="canonical", seed=0)
    return summary.with_paired(pairs)


def _write_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write minimal context_sets_v1.jsonl and outcomes_v1.jsonl to tmp_path."""
    context_sets = [
        _context_set(f"q{i}_{strategy}", f"q{i}", strategy)
        for strategy in ("gold_only", "shuffled_order")
        for i in range(4)
    ]
    outcomes = [
        _outcome(f"q{i}_{strategy}", f"q{i}", 0.6 + 0.05 * i)
        for strategy in ("gold_only", "shuffled_order")
        for i in range(4)
    ]
    (tmp_path / "context_sets_v1.jsonl").write_text(
        "\n".join(json.dumps(cs.to_dict()) for cs in context_sets) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "outcomes_v1.jsonl").write_text(
        "\n".join(json.dumps(o.to_dict()) for o in outcomes) + "\n",
        encoding="utf-8",
    )
    summary = _build_summary()
    summary_path = tmp_path / "replications_summary_v1.jsonl"
    _write_summary(summary_path, summary)
    return tmp_path, summary_path, tmp_path / "context_sets_v1.jsonl"


def _run_analyze(monkeypatch, tmp_path: Path, *args):
    """Run scripts/analyze_results.py against tmp_path."""
    from context_engine.config import ExperimentConfig, build_default_config
    monkeypatch.setattr(sys, "argv", ["analyze_results", "--dataset-dir", str(tmp_path), *args])
    # Import the script's main()
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyze_results", "scripts/analyze_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main()


def test_analyze_json_includes_replication_summary(tmp_path: Path, monkeypatch, capsys):
    _write_artifacts(tmp_path)
    rc = _run_analyze(monkeypatch, tmp_path, "--format", "json", "--replications-summary", "replications_summary_v1.jsonl")
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert "replication_summary" in payload
    assert payload["replication_summary"]["n_runs"] == 3
    assert len(payload["replication_summary"]["strategies"]) == 2
    assert len(payload["replication_summary"]["paired"]) == 1


def test_analyze_md_includes_replication_section(tmp_path: Path, monkeypatch, capsys):
    _write_artifacts(tmp_path)
    rc = _run_analyze(monkeypatch, tmp_path, "--format", "md", "--replications-summary", "replications_summary_v1.jsonl")
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Replication Confidence Intervals" in out
    assert "## Pool Comparison" in out
    assert "reliability" in out
    assert "auto - canonical" in out


def test_analyze_without_replication_summary_runs_clean(tmp_path: Path, monkeypatch, capsys):
    _write_artifacts(tmp_path)
    rc = _run_analyze(monkeypatch, tmp_path, "--format", "md")
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Replication Confidence Intervals" not in out
    assert "## Pool Comparison" not in out


def test_analyze_rejects_empty_replication_summary(tmp_path: Path, monkeypatch):
    _write_artifacts(tmp_path)
    empty_path = tmp_path / "empty_summary.jsonl"
    empty_path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _run_analyze(monkeypatch, tmp_path, "--format", "json", "--replications-summary", "empty_summary.jsonl")


def test_analyze_replication_summary_csv_mode_unaffected(tmp_path: Path, monkeypatch, capsys):
    """CSV format should not include the replication section (it's not a per-set view)."""
    _write_artifacts(tmp_path)
    rc = _run_analyze(monkeypatch, tmp_path, "--format", "csv", "--replications-summary", "replications_summary_v1.jsonl")
    assert rc == 0
    out = capsys.readouterr().out
    # CSV must keep its header even with replication summary passed.
    assert out.startswith("set_id,query_id,strategy")
    assert "replication_summary" not in out
