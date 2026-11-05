"""Strict JSONL Serializer for DataMate Submissions.

Guarantees compliance with `rulebook.md` §9:
- Exact keys: `id`, `type`, `answer`
- `allow_nan=False`, `ensure_ascii=False`
- `unanswerable` and `abstain` always serialize `answer: null`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.executor import ExecutionResult
from src.ir import SubmissionRecord
from src.verifier import verify_and_gate


def result_to_record(
    question_id: str,
    res: ExecutionResult,
    confidence_threshold: float = 0.65,
) -> SubmissionRecord:
    """Convert an `ExecutionResult` into a verified `SubmissionRecord`."""
    verified = verify_and_gate(res, confidence_threshold=confidence_threshold)
    return SubmissionRecord(
        id=str(question_id),
        type=verified.answer_type,
        answer=verified.value if verified.answer_type not in ("unanswerable", "abstain") else None,
    )


def serialize_record_line(record: SubmissionRecord) -> str:
    """Serialize a single `SubmissionRecord` to a strict JSONL line."""
    payload = {
        "id": record.id,
        "type": record.type,
        "answer": record.answer,
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


def serialize_prediction_line(question_id: str, res: ExecutionResult) -> str:
    """Helper to convert ExecutionResult and serialize directly."""
    return serialize_record_line(result_to_record(question_id, res))


def write_submission_jsonl(
    records: Iterable[SubmissionRecord],
    output_path: str | Path,
) -> int:
    """Write `SubmissionRecord` items to `output_path` and return total line count."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(serialize_record_line(rec) + "\n")
            count += 1
    return count
