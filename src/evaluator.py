"""Official-compliant evaluator for DataMate (Zalo AI Challenge 2026).

Implements Section 9 of rulebook.md:
- Correct answer: +1
- Wrong answer (or invalid format): -1
- Abstain ({"type": "abstain"} or missing line, or unanswerable with non-string reason): 0
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


VALID_TYPES = {
    "number",
    "percent",
    "date",
    "boolean",
    "string",
    "list",
    "table",
    "unanswerable",
    "abstain",
}


def normalize_vn_string(val: str) -> str:
    """Normalize string per Rulebook §9:
    Case-insensitive, ignore whitespace and punctuation, preserve Vietnamese diacritics.
    """
    s = unicodedata.normalize("NFC", val).casefold()
    # Remove all punctuation and whitespace (keep letters and numbers, including Vietnamese)
    cleaned = "".join(
        ch for ch in s if unicodedata.category(ch)[0] in ("L", "N")
    )
    return cleaned


def is_json_number(val: Any) -> bool:
    """Check if val is a valid JSON number (int or float, excluding bool and NaN/Inf)."""
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return math.isfinite(float(val))
    return False


def numbers_match(pred: Any, gold: Any, tol: float = 1e-5) -> bool:
    if not is_json_number(pred) or not is_json_number(gold):
        return False
    return abs(float(pred) - float(gold)) <= tol


def strings_match(pred: Any, gold: str, aliases: Optional[List[str]] = None) -> bool:
    if not isinstance(pred, str) or not pred.strip():
        return False
    norm_pred = normalize_vn_string(pred)
    if not norm_pred:
        return False
    candidates = [gold] + (aliases or [])
    for cand in candidates:
        if isinstance(cand, str) and normalize_vn_string(cand) == norm_pred:
            return True
    return False


def lists_match(
    pred: Any, gold_list: List[Any], gold_aliases: Optional[List[Any]] = None
) -> bool:
    if not isinstance(pred, list):
        return False
    # Check element format first: elements must be numbers or strings
    for item in pred:
        if not (is_json_number(item) or isinstance(item, str)):
            return False
    if len(pred) != len(gold_list):
        return False
    for idx, (p_item, g_item) in enumerate(zip(pred, gold_list)):
        item_aliases: Optional[List[str]] = None
        if gold_aliases and idx < len(gold_aliases):
            if isinstance(gold_aliases[idx], list):
                item_aliases = gold_aliases[idx]
            elif isinstance(gold_aliases[idx], str):
                item_aliases = [gold_aliases[idx]]
        if is_json_number(g_item):
            if not numbers_match(p_item, g_item):
                return False
        elif isinstance(g_item, str):
            if not strings_match(p_item, g_item, item_aliases):
                return False
        else:
            if p_item != g_item:
                return False
    return True


def tables_match(
    pred: Any, gold_table: Dict[str, Any], gold_aliases: Optional[List[List[str]]] = None
) -> bool:
    """Compare table per Rulebook §9:
    - {"columns": [...], "rows": [[...], ...]}
    - Numeric cells must be JSON numbers.
    - First column is object name/code (matched with aliases); subsequent columns are measures.
    - Match columns by name if possible, otherwise by position.
    - Row order does NOT affect grading.
    """
    if not isinstance(pred, dict):
        return False
    p_cols = pred.get("columns")
    p_rows = pred.get("rows")
    if not isinstance(p_cols, list) or not isinstance(p_rows, list):
        return False
    if not all(isinstance(c, str) for c in p_cols):
        return False

    g_cols = gold_table.get("columns", [])
    g_rows = gold_table.get("rows", [])
    if len(p_cols) != len(g_cols):
        return False

    for r in p_rows:
        if not isinstance(r, list) or len(r) != len(p_cols):
            return False
        # First column must be string, subsequent columns must match expected type (numbers)
        if len(r) > 0 and not isinstance(r[0], str):
            return False
        for cell in r[1:]:
            if not is_json_number(cell) and not isinstance(cell, str):
                return False

    if len(p_rows) != len(g_rows):
        return False

    # Column alignment: if all column names match as a set (after normalization), permute; else positional
    norm_g_cols = [normalize_vn_string(c) for c in g_cols]
    norm_p_cols = [normalize_vn_string(c) for c in p_cols]
    col_map = list(range(len(g_cols)))
    if len(set(norm_g_cols)) == len(g_cols) and set(norm_g_cols) == set(norm_p_cols):
        col_map = [norm_p_cols.index(nc) for nc in norm_g_cols]

    # Row matching (order-insensitive bipartite matching)
    used_pred = [False] * len(p_rows)
    for g_idx, g_row in enumerate(g_rows):
        row_aliases = (
            gold_aliases[g_idx]
            if (gold_aliases and g_idx < len(gold_aliases))
            else None
        )
        matched = False
        for p_idx, p_row in enumerate(p_rows):
            if used_pred[p_idx]:
                continue
            aligned_p = [p_row[col_map[c]] for c in range(len(g_cols))]
            # Compare col 0 (object name)
            if not strings_match(aligned_p[0], g_row[0], row_aliases):
                continue
            # Compare remaining cols
            cells_ok = True
            for c_idx in range(1, len(g_cols)):
                g_val = g_row[c_idx]
                p_val = aligned_p[c_idx]
                if is_json_number(g_val):
                    if not numbers_match(p_val, g_val):
                        cells_ok = False
                        break
                else:
                    if not strings_match(p_val, str(g_val)):
                        cells_ok = False
                        break
            if cells_ok:
                used_pred[p_idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


def score_single_record(
    pred: Optional[Dict[str, Any]], gold: Dict[str, Any]
) -> Tuple[int, str]:
    """Score a single prediction against ground truth.

    Returns:
        (score, status_tag) where score in {+1, 0, -1} and status_tag in:
        - "ABSTAIN" (0)
        - "CORRECT" (+1)
        - "WRONG_TYPE" (-1)
        - "FORMAT_ERROR" (-1)
        - "WRONG_VALUE" (-1)
    """
    if pred is None:
        return 0, "ABSTAIN"
    if not isinstance(pred, dict):
        return -1, "FORMAT_ERROR"

    p_type = pred.get("type")
    if p_type == "abstain" or p_type is None:
        return 0, "ABSTAIN"

    if p_type not in VALID_TYPES:
        return -1, "FORMAT_ERROR"

    # Rulebook §7: If unanswerable has a non-string reason, line is unreadable and treated as abstain (0)
    if p_type == "unanswerable":
        if "reason" in pred and pred["reason"] is not None and not isinstance(pred["reason"], str):
            return 0, "ABSTAIN"
        if gold["type"] == "unanswerable":
            return 1, "CORRECT"
        return -1, "WRONG_TYPE"

    # Check format of pred["answer"] based on p_type BEFORE checking against gold["type"]
    p_ans = pred.get("answer")
    if p_type in ("number", "percent"):
        if not is_json_number(p_ans):
            return -1, "FORMAT_ERROR"
    elif p_type == "date":
        if not isinstance(p_ans, str) or not re.match(r"^\d{4}-\d{2}(-\d{2})?$", p_ans.strip()):
            return -1, "FORMAT_ERROR"
    elif p_type == "boolean":
        if not isinstance(p_ans, bool):
            return -1, "FORMAT_ERROR"
    elif p_type == "string":
        if not isinstance(p_ans, str) or not p_ans.strip():
            return -1, "FORMAT_ERROR"
    elif p_type == "list":
        if not isinstance(p_ans, list):
            return -1, "FORMAT_ERROR"
    elif p_type == "table":
        if not isinstance(p_ans, dict) or "columns" not in p_ans or "rows" not in p_ans:
            return -1, "FORMAT_ERROR"

    g_type = gold["type"]
    if p_type != g_type:
        return -1, "WRONG_TYPE"

    g_ans = gold.get("answer")
    g_alias = gold.get("alias", [])

    if g_type in ("number", "percent"):
        ok = numbers_match(p_ans, g_ans)
    elif g_type == "date":
        ok = isinstance(p_ans, str) and p_ans.strip() == str(g_ans).strip()
    elif g_type == "boolean":
        ok = isinstance(p_ans, bool) and p_ans == g_ans
    elif g_type == "string":
        ok = strings_match(p_ans, g_ans, g_alias)
    elif g_type == "list":
        ok = lists_match(p_ans, g_ans, g_alias)
    elif g_type == "table":
        ok = tables_match(p_ans, g_ans, g_alias)
    else:
        ok = False

    return (1, "CORRECT") if ok else (-1, "WRONG_VALUE")


def evaluate_predictions(
    gold_path: Path | str,
    pred_path: Path | str,
    questions_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Evaluate a submission JSONL file against ground-truth answers JSONL."""
    gold_records: Dict[str, Dict[str, Any]] = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                gold_records[obj["id"]] = obj

    pred_records: Dict[str, Optional[Dict[str, Any]]] = {}
    malformed_lines = 0
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "id" in obj:
                    pred_records[obj["id"]] = obj
                else:
                    malformed_lines += 1
            except json.JSONDecodeError:
                malformed_lines += 1

    questions_map: Dict[str, Dict[str, Any]] = {}
    if questions_path and Path(questions_path).exists():
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    questions_map[q["id"]] = q

    total = len(gold_records)
    total_score = 0
    counts = {
        "CORRECT": 0,
        "ABSTAIN": 0,
        "WRONG_VALUE": 0,
        "WRONG_TYPE": 0,
        "FORMAT_ERROR": 0,
    }
    by_type: Dict[str, Dict[str, int]] = {}
    by_business: Dict[str, Dict[str, int]] = {}
    errors: List[Dict[str, Any]] = []

    for qid, gold in gold_records.items():
        pred = pred_records.get(qid)
        sc, status = score_single_record(pred, gold)
        total_score += sc
        counts[status] = counts.get(status, 0) + 1

        gtype = gold["type"]
        if gtype not in by_type:
            by_type[gtype] = {"total": 0, "score": 0, "correct": 0, "wrong": 0, "abstain": 0}
        by_type[gtype]["total"] += 1
        by_type[gtype]["score"] += sc
        if sc == 1:
            by_type[gtype]["correct"] += 1
        elif sc == -1:
            by_type[gtype]["wrong"] += 1
        else:
            by_type[gtype]["abstain"] += 1

        biz = questions_map.get(qid, {}).get("business_id", "unknown")
        if biz not in by_business:
            by_business[biz] = {"total": 0, "score": 0, "correct": 0, "wrong": 0, "abstain": 0}
        by_business[biz]["total"] += 1
        by_business[biz]["score"] += sc
        if sc == 1:
            by_business[biz]["correct"] += 1
        elif sc == -1:
            by_business[biz]["wrong"] += 1
        else:
            by_business[biz]["abstain"] += 1

        if sc == -1:
            q_info = questions_map.get(qid, {})
            errors.append(
                {
                    "id": qid,
                    "business_id": q_info.get("business_id"),
                    "as_of_date": q_info.get("as_of_date"),
                    "question": q_info.get("question"),
                    "status": status,
                    "gold_type": gold["type"],
                    "gold_answer": gold.get("answer"),
                    "gold_alias": gold.get("alias"),
                    "pred_type": pred.get("type") if isinstance(pred, dict) else None,
                    "pred_answer": pred.get("answer") if isinstance(pred, dict) else None,
                }
            )

    answered = counts["CORRECT"] + counts["WRONG_VALUE"] + counts["WRONG_TYPE"] + counts["FORMAT_ERROR"]
    coverage = answered / total if total else 0.0
    precision_on_answered = counts["CORRECT"] / answered if answered else 0.0

    return {
        "total_questions": total,
        "total_score": total_score,
        "normalized_score": total_score / total if total else 0.0,
        "answered": answered,
        "coverage": round(coverage, 4),
        "precision_on_answered": round(precision_on_answered, 4),
        "counts": counts,
        "malformed_lines": malformed_lines,
        "by_type": by_type,
        "by_business": dict(sorted(by_business.items())),
        "errors": errors,
    }
