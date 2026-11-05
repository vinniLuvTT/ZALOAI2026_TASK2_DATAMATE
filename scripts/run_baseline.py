#!/usr/bin/env python3
"""CLI entrypoint for DataMate Baseline Runner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner import DataMateRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DataMate baseline end-to-end.")
    parser.add_argument(
        "--split",
        type=str,
        default="public_test",
        help="Dataset split directory (e.g. 'train' or 'public_test').",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional override path to questions.jsonl (defaults to <split>/questions.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.jsonl",
        help="Output submission JSONL file path.",
    )
    parser.add_argument(
        "--trace",
        type=str,
        default=None,
        help="Optional diagnostic trace JSONL output path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to baseline YAML config.",
    )
    args = parser.parse_args()

    split_dir = Path(args.split)
    questions_path = Path(args.input) if args.input else (split_dir / "questions.jsonl")

    runner = DataMateRunner(split_dir=split_dir, config_path=args.config)
    stats = runner.run_file(
        questions_path=questions_path,
        output_path=args.output,
        trace_path=args.trace,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
