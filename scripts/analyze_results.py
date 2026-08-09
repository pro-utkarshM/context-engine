"""Aggregate and render benchmark results.

Reads context-set and outcome artifacts (and optionally a marginal-impact
artifact for the predicted-vs-measured view) and emits a report in one of
several formats. The default format is plain text on stdout for
backward compatibility with prior versions of this script.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from context_engine.analysis import (
    render_csv_per_query,
    render_json_report,
    render_markdown_report,
    render_text_report,
)
from context_engine.artifacts import ContextSet, MarginalImpact, Outcome, Query
from context_engine.config import add_config_args, config_from_args, resolved_artifact_path
from context_engine.io import load_jsonl
from context_engine.replications import ReplicationSummary


FORMATS = ("text", "json", "csv", "md")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze_results")
    add_config_args(parser)
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="text",
        help="Output format. Default: text (stdout).",
    )
    parser.add_argument(
        "--outcomes",
        default=None,
        help="Outcome artifact filename under dataset_dir. Defaults to outcomes_<version>.jsonl.",
    )
    parser.add_argument(
        "--context-sets",
        default=None,
        help="Context set artifact filename under dataset_dir. Defaults to context_sets_<version>.jsonl.",
    )
    parser.add_argument(
        "--marginal-impacts",
        default=None,
        help="Optional marginal-impact artifact filename under dataset_dir. "
        "When provided, enables the predicted-vs-measured view in JSON / Markdown reports.",
    )
    parser.add_argument(
        "--queries",
        default=None,
        help="Optional queries artifact filename under dataset_dir. "
        "Required for the predicted-vs-measured view to label chunks as gold vs distractor.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path. Default: stdout.",
    )
    parser.add_argument(
        "--replications-summary",
        default=None,
        help="Optional replication-summary JSONL filename under dataset_dir. "
        "When provided, the JSON / Markdown reports include a per-strategy "
        "bootstrap CI section and a paired pool-comparison section "
        "(if the summary carries paired deltas).",
    )
    return parser


def _load_artifacts(
    config,
    *,
    outcomes_filename: str | None,
    context_sets_filename: str | None,
    marginal_impacts_filename: str | None,
    queries_filename: str | None,
) -> tuple[list[ContextSet], list[Outcome], list[MarginalImpact], dict[str, Query]]:
    context_sets_path = (
        config.dataset_dir / context_sets_filename
        if context_sets_filename
        else resolved_artifact_path(config, "context_sets")
    )
    outcomes_path = (
        config.dataset_dir / outcomes_filename
        if outcomes_filename
        else resolved_artifact_path(config, "outcomes")
    )

    context_sets = [ContextSet.from_dict(row) for row in load_jsonl(context_sets_path)]
    outcomes = [Outcome.from_dict(row) for row in load_jsonl(outcomes_path)]

    marginal_impacts: list[MarginalImpact] = []
    if marginal_impacts_filename:
        mi_path = config.dataset_dir / marginal_impacts_filename
        marginal_impacts = [MarginalImpact.from_dict(row) for row in load_jsonl(mi_path)]

    queries_by_id: dict[str, Query] = {}
    if queries_filename:
        q_path = config.dataset_dir / queries_filename
    else:
        q_path = resolved_artifact_path(config, "queries")
    if marginal_impacts and q_path.exists():
        queries_by_id = {
            query.query_id: query
            for query in (Query.from_dict(row) for row in load_jsonl(q_path))
        }

    return context_sets, outcomes, marginal_impacts, queries_by_id


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)

    context_sets, outcomes, marginal_impacts, queries_by_id = _load_artifacts(
        config,
        outcomes_filename=args.outcomes,
        context_sets_filename=args.context_sets,
        marginal_impacts_filename=args.marginal_impacts,
        queries_filename=args.queries,
    )

    replication_summary: ReplicationSummary | None = None
    if args.replications_summary:
        summary_path = config.dataset_dir / args.replications_summary
        rows = load_jsonl(summary_path)
        if not rows:
            raise ValueError(f"replications summary at {summary_path} is empty")
        # Round-trip through the dataclass to normalize types.
        from context_engine.stats import DistributionSummary, PairedDeltaSummary
        from context_engine.replications import (
            ReplicationPairedDelta,
            ReplicationStrategySummary,
        )

        summary_row = rows[0]
        strategies: list[ReplicationStrategySummary] = []
        for entry in summary_row.get("strategies", []):
            rms = DistributionSummary(**entry["run_mean_summary"])
            wqs = DistributionSummary(**entry["within_run_query_summary"])
            strategies.append(
                ReplicationStrategySummary(
                    strategy=entry["strategy"],
                    n_runs=entry["n_runs"],
                    n_queries_per_run=entry["n_queries_per_run"],
                    run_means=entry["run_means"],
                    run_mean_summary=rms,
                    within_run_query_means=entry["within_run_query_means"],
                    within_run_query_summary=wqs,
                )
            )
        paired: list[ReplicationPairedDelta] = []
        for entry in summary_row.get("paired", []):
            ds_kwargs = dict(entry["delta_summary"])
            # Backward compatibility: legacy summary files only had
            # ``p_value_two_sided`` (which was actually a one-sided tail
            # probability mislabeled). Map the legacy value to
            # ``p_value_one_sided`` and leave ``p_value_two_sided`` at
            # None so callers can detect the legacy form.
            if "p_value_one_sided" not in ds_kwargs and "p_value_two_sided" in ds_kwargs:
                ds_kwargs["p_value_one_sided"] = ds_kwargs.pop("p_value_two_sided")
                ds_kwargs["p_value_two_sided"] = 0.0
            ds = PairedDeltaSummary(**ds_kwargs)
            paired.append(
                ReplicationPairedDelta(
                    strategy=entry["strategy"],
                    n_queries=entry["n_queries"],
                    left_pool_source=entry["left_pool_source"],
                    right_pool_source=entry["right_pool_source"],
                    delta_summary=ds,
                )
            )
        replication_summary = ReplicationSummary(
            experiment_name=summary_row["experiment_name"],
            runner=summary_row["runner"],
            model_name=summary_row["model_name"],
            artifact_version=summary_row["artifact_version"],
            n_runs=summary_row["n_runs"],
            n_resamples=summary_row["n_resamples"],
            ci_level=summary_row["ci_level"],
            seed=summary_row["seed"],
            strategies=strategies,
            paired=paired,
        )

    if args.format == "text":
        report = render_text_report(context_sets, outcomes)
    elif args.format == "json":
        report = render_json_report(
            context_sets,
            outcomes,
            marginal_impacts=marginal_impacts or None,
            queries_by_id=queries_by_id or None,
            replication_summary=replication_summary,
        )
    elif args.format == "csv":
        report = render_csv_per_query(context_sets, outcomes)
    else:  # md
        report = render_markdown_report(
            context_sets,
            outcomes,
            marginal_impacts=marginal_impacts or None,
            queries_by_id=queries_by_id or None,
            replication_summary=replication_summary,
        )

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(Path(args.out))
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())