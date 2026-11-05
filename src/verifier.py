"""Verification & Abstention Gating Layer for DataMate.

Validates `ExecutionResult` from `src/executor.py`, sanitizes numpy/pandas scalars,
checks type/value invariants per `rulebook.md` §9, and converts invalid or
low-confidence results into `abstain` (0 points instead of -1 penalty).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Any, List

import numpy as np

from src.executor import ExecutionResult

VALID_TYPES = {
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
}


def sanitize_scalar(val: Any) -> Any:
    """Convert numpy/pandas scalar types to pure Python types and check finiteness."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if not math.isfinite(f):
            return None
        if abs(f - round(f)) < 1e-9:
            return int(round(f))
        return round(f, 2)
    if isinstance(val, (dt.datetime, dt.date)):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str):
        return val.strip()
    return val


def verify_and_gate(
    res: ExecutionResult,
    confidence_threshold: float = 0.65,
) -> ExecutionResult:
    """Verify `ExecutionResult` and gate low-confidence or malformed answers to `abstain`."""
    if res is None:
        return ExecutionResult(answer_type="abstain", value=None, reason="Null result", confidence=0.0)

    t = res.answer_type
    if t not in VALID_TYPES:
        return ExecutionResult(answer_type="abstain", value=None, reason=f"Invalid type: {t}", confidence=0.0)

    if t == "abstain":
        return ExecutionResult(answer_type="abstain", value=None, reason=res.reason, confidence=0.0)

    if res.confidence < confidence_threshold:
        return ExecutionResult(
            answer_type="abstain",
            value=None,
            reason=f"Confidence {res.confidence:.2f} < {confidence_threshold:.2f}",
            confidence=res.confidence,
        )

    if t == "unanswerable":
        return ExecutionResult(answer_type="unanswerable", value=None, reason=res.reason, confidence=res.confidence)

    val = res.value

    if t in ("number", "percent"):
        s_val = sanitize_scalar(val)
        if s_val is None or isinstance(s_val, bool) or not isinstance(s_val, (int, float)):
            return ExecutionResult(answer_type="abstain", value=None, reason="Invalid numeric value", confidence=0.0)
        return ExecutionResult(answer_type=t, value=s_val, reason=res.reason, confidence=res.confidence)

    if t == "boolean":
        if not isinstance(val, (bool, np.bool_)):
            return ExecutionResult(answer_type="abstain", value=None, reason="Invalid boolean value", confidence=0.0)
        return ExecutionResult(answer_type="boolean", value=bool(val), reason=res.reason, confidence=res.confidence)

    if t == "string":
        if val is None:
            return ExecutionResult(answer_type="abstain", value=None, reason="Null string", confidence=0.0)
        s = str(val).strip()
        if not s:
            return ExecutionResult(answer_type="abstain", value=None, reason="Empty string", confidence=0.0)
        return ExecutionResult(answer_type="string", value=s, reason=res.reason, confidence=res.confidence)

    if t == "date":
        s_val = sanitize_scalar(val)
        if not isinstance(s_val, str) or not re.match(r"^\d{4}-\d{2}(?:-\d{2})?$", s_val):
            return ExecutionResult(answer_type="abstain", value=None, reason=f"Invalid date: {s_val}", confidence=0.0)
        return ExecutionResult(answer_type="date", value=s_val, reason=res.reason, confidence=res.confidence)

    if t == "list":
        if not isinstance(val, (list, tuple)):
            return ExecutionResult(answer_type="abstain", value=None, reason="Invalid list value", confidence=0.0)
        clean_list: List[Any] = []
        for item in val:
            si = sanitize_scalar(item)
            if si is None or (isinstance(si, str) and not si):
                continue
            clean_list.append(si)
        if not clean_list:
            return ExecutionResult(answer_type="abstain", value=None, reason="Empty list", confidence=0.0)
        return ExecutionResult(answer_type="list", value=clean_list, reason=res.reason, confidence=res.confidence)

    if t == "table":
        if not isinstance(val, dict) or "columns" not in val or "rows" not in val:
            return ExecutionResult(answer_type="abstain", value=None, reason="Invalid table dict", confidence=0.0)
        cols = val.get("columns")
        rows = val.get("rows")
        if not isinstance(cols, list) or not cols or not isinstance(rows, list) or not rows:
            return ExecutionResult(answer_type="abstain", value=None, reason="Empty table", confidence=0.0)
        n_cols = len(cols)
        clean_rows: List[List[Any]] = []
        for r in rows:
            if not isinstance(r, (list, tuple)) or len(r) != n_cols:
                return ExecutionResult(answer_type="abstain", value=None, reason="Row length mismatch", confidence=0.0)
            cr = [sanitize_scalar(c) for c in r]
            if any(c is None for c in cr):
                return ExecutionResult(answer_type="abstain", value=None, reason="Null cell in table", confidence=0.0)
            clean_rows.append(cr)
        return ExecutionResult(
            answer_type="table",
            value={"columns": [str(c) for c in cols], "rows": clean_rows},
            reason=res.reason,
            confidence=res.confidence,
        )

    return ExecutionResult(answer_type="abstain", value=None, reason="Fallback abstain", confidence=0.0)
