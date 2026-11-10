"""Local scorer implementing the matching rules of the task (section 2.1) + PA with a configurable penalty.

The official PA formula is an image in the task statement. Default here:
    PA = (correct - penalty * wrong) / N        (abstain = 0)
Set DM_PENALTY (or --penalty) to match the real formula once you read it.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Optional

from .verify import norm_str, norm_str_compact


def _is_num(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return False


def _num_eq(a: Any, b: Any, tol: float = 1e-2) -> bool:
    if not _is_num(a) or not _is_num(b):
        return False
    return abs(float(a) - float(b)) <= tol or abs(round(float(a), 2) - round(float(b), 2)) < 1e-9


def _str_eq(pred: Any, gold: Any, aliases: Optional[list] = None) -> bool:
    if not isinstance(pred, str) or not pred.strip():
        return False
    np = norm_str_compact(pred)
    if not np:
        return False
    cands = [gold] + (aliases or [])
    for c in cands:
        if isinstance(c, str) and (norm_str_compact(c) == np or norm_str(c) == norm_str(pred)):
            return True
    return False


def _cell_eq(p: Any, r: Any, aliases: Optional[list] = None) -> bool:
    if _is_num(r):
        return _num_eq(p, r)
    return _str_eq(p, str(r), aliases)


def match(pred: Optional[dict], ref: dict) -> str:
    """-> 'correct' | 'wrong' | 'abstain'"""
    if not pred or not isinstance(pred, dict) or pred.get("type") in (None, "abstain"):
        return "abstain"
    rt, pt = ref["type"], pred["type"]
    if pt == "unanswerable":
        if "reason" in pred and pred["reason"] is not None and not isinstance(pred["reason"], str):
            return "abstain"
        return "correct" if rt == "unanswerable" else "wrong"
    if pt != rt:
        return "wrong"
    pa, ra = pred.get("answer"), ref.get("answer")
    aliases = ref.get("alias") or []
    if rt in ("number", "percent"):
        return "correct" if _num_eq(pa, ra) else "wrong"
    if rt == "string":
        return "correct" if _str_eq(pa, ra, aliases if isinstance(aliases, list) else []) else "wrong"
    if rt == "date":
        if not isinstance(pa, str) or not re.match(r"^\d{4}-\d{2}(-\d{2})?$", pa.strip()):
            return "wrong"
        return "correct" if pa.strip() == str(ra).strip() else "wrong"
    if rt == "boolean":
        return "correct" if isinstance(pa, bool) and pa == bool(ra) else "wrong"
    if rt == "list":
        if not isinstance(pa, list) or len(pa) != len(ra):
            return "wrong"
        for idx, (p_item, r_item) in enumerate(zip(pa, ra)):
            item_al = None
            if isinstance(aliases, list) and idx < len(aliases):
                item_al = aliases[idx] if isinstance(aliases[idx], list) else [aliases[idx]]
            if not _cell_eq(p_item, r_item, item_al):
                return "wrong"
        return "correct"
    if rt == "table":
        try:
            if not isinstance(pa, dict):
                return "wrong"
            pc, rc = pa["columns"], ra["columns"]
            pr, rr = pa["rows"], ra["rows"]
            if len(pc) != len(rc) or len(pr) != len(rr):
                return "wrong"
            by_name = [norm_str_compact(c) for c in pc]
            ref_names = [norm_str_compact(c) for c in rc]
            if sorted(by_name) == sorted(ref_names) and len(set(ref_names)) == len(ref_names):
                idx = [by_name.index(n) for n in ref_names]
            else:
                idx = list(range(len(rc)))
            prow = [[row[i] for i in idx] for row in pr]
            used = [False] * len(prow)
            for g_idx, g_row in enumerate(rr):
                row_al = aliases[g_idx] if (isinstance(aliases, list) and g_idx < len(aliases)) else None
                if isinstance(row_al, str):
                    row_al = [row_al]
                matched = False
                for p_idx, p_row in enumerate(prow):
                    if used[p_idx] or len(p_row) != len(g_row):
                        continue
                    if len(g_row) > 0 and not _cell_eq(p_row[0], g_row[0], row_al):
                        continue
                    if all(_cell_eq(p_row[c], g_row[c]) for c in range(1, len(g_row))):
                        used[p_idx] = True
                        matched = True
                        break
                if not matched:
                    return "wrong"
            return "correct"
        except Exception:
            return "wrong"
    return "wrong"


def pa_score(c: int, w: int, n: int, penalty: float = 1.0) -> float:
    return (c - penalty * w) / n if n else 0.0


def summarize(records: list[dict], refs: dict[str, dict], penalty: float = 1.0) -> dict:
    """records: [{id, business_id, final, ...}]. Missing answers count as abstain."""
    tot = Counter()
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_biz: dict[str, Counter] = defaultdict(Counter)
    by_id = {r["id"]: r for r in records}
    for qid, ref in refs.items():
        rec = by_id.get(qid)
        res = match(rec["final"] if rec else None, ref)
        tot[res] += 1
        by_type[ref["type"]][res] += 1
        by_biz[(rec or {}).get("business_id", "?")][res] += 1
    n = sum(tot.values())
    out = {
        "N": n, "correct": tot["correct"], "wrong": tot["wrong"], "abstain": tot["abstain"],
        "accuracy": tot["correct"] / n if n else 0.0,
        "PA": pa_score(tot["correct"], tot["wrong"], n, penalty), "penalty": penalty,
        "by_type": {t: dict(c) for t, c in by_type.items()},
        "by_business": {b: dict(c) for b, c in sorted(by_biz.items())},
    }
    return out


def calibrate(records: list[dict], refs: dict[str, dict], penalty: float = 1.0) -> list[tuple[float, float, int, int]]:
    """Replay saved votes at different agreement thresholds -> [(threshold, PA, correct, wrong)].
    Needs records with `candidate` (top voted answer ignoring abstain gates) and `confidence`."""
    rows = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        c = w = n = 0
        for qid, ref in refs.items():
            rec = next((r for r in records if r["id"] == qid), None)
            n += 1
            if not rec or not rec.get("candidate") or rec.get("confidence", 0) + 1e-9 < t:
                continue
            res = match(rec["candidate"], ref)
            c += res == "correct"
            w += res == "wrong"
        rows.append((t, pa_score(c, w, n, penalty), c, w))
    return rows
