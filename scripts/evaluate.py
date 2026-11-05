#!/usr/bin/env python3
"""CLI entrypoint for evaluating a DataMate submission JSONL against answers JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DataMate predictions.")
    parser.add_argument("--gold", default="train/answers.jsonl", help="Path to ground-truth answers.jsonl")
    parser.add_argument("--pred", required=True, help="Path to predicted submission.jsonl")
    parser.add_argument("--questions", default="train/questions.jsonl", help="Path to questions.jsonl")
    parser.add_argument("--output", default=None, help="Optional path to save detailed JSON report")
    args = parser.parse_args()

    res = evaluate_predictions(args.gold, args.pred, args.questions)
    summary = {k: v for k, v in res.items() if k not in ("errors", "by_business")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
