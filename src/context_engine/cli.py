"""Minimal CLI for artifact validation."""

from __future__ import annotations

import argparse

from .validation import validate_jsonl_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="context-engine-validate")
    parser.add_argument("path", help="Path to a versioned JSONL artifact file.")
    parser.add_argument(
        "--artifact",
        dest="artifact_name",
        default=None,
        help="Optional explicit artifact type override.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = validate_jsonl_file(args.path, args.artifact_name)
    print(f"validated {summary.row_count} rows from {summary.path} as {summary.artifact_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
