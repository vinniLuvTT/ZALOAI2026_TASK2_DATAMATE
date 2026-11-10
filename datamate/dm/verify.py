"""
Answer normalization, type validation, self-consistency voting and abstention.

Answer dict shapes (submission format):
  {"type": "number", "answer": 48250000}   {"type": "percent", "answer": -12.45}
  {"type": "string", "answer": "..."}      {"type": "date", "answer": "2026-08-15" | "2025-09"}
  {"type": "boolean", "answer": true}      {"type": "list", "answer": [...]}
  {"type": "table", "answer": {"columns": [...], "rows": [[...]]}}
  {"type": "unanswerable", "reason": "..."} {"type": "abstain"}
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"}


def round_half_up(x: float, nd: int = 2) -> float:
    """Round half-up per Rulebook §9 (e.g. 2.345 -> 2.35)."""
    q = Decimal("10") ** (-nd) if nd > 0 else Decimal("1")
    return float(Decimal(str(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def norm_str(s: Any) -> str:
    """Official string normalization: NFC, lowercase, non letter/digit -> space, collapse."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = re.sub(r"[^\w\s]|_", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_str_compact(s: Any) -> str:
    """Rulebook §9 exact string normalization: NFC, casefold, strip all punctuation and whitespace."""
    s = unicodedata.normalize("NFC", str(s)).casefold()
    return "".join(ch for ch in s if unicodedata.category(ch)[0] in ("L", "N"))


def _to_number(x: Any) -> float:
    if isinstance(x, bool):
        raise ValueError("bool is not a number")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace(" ", "")
    s = re.sub(r"(vnđ|vnd|đ|₫|%)$", "", s, flags=re.I)
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    elif re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"-?\d+,\d+", s):
        s = s.replace(",", ".")
    return float(s)


def _num_out(x: float, nd: int = 2):
    r = round_half_up(float(x), nd)
    return int(round(r)) if abs(r - round(r)) < 1e-9 else r


def _date_out(x: Any) -> str:
    s = str(x).strip()
    if re.fullmatch(r"\d{4}-\d{2}(-\d{2})?", s):
        if len(s) == 10:
            datetime.strptime(s, "%Y-%m-%d")
        else:
            m = int(s[5:7])
            if not 1 <= m <= 12:
                raise ValueError("bad month")
        return s
    m = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", s)
    if m:
        return datetime(int(m[3]), int(m[2]), int(m[1])).strftime("%Y-%m-%d")
    m = re.fullmatch(r"(\d{1,2})[/-](\d{4})", s)
    if m:
        return f"{int(m[2]):04d}-{int(m[1]):02d}"
    raise ValueError(f"unrecognized date {s!r} (use yyyy-mm-dd or yyyy-mm)")


def _elem_out(x: Any):
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return _num_out(x)
    return str(x)


def coerce(atype: str, answer: Any = None, reason: str = "") -> dict:
    """Validate + normalize. Raises ValueError with a message the model can act on."""
    atype = str(atype).strip().lower()
    if atype not in TYPES:
        raise ValueError(f"type must be one of {sorted(TYPES)}")
    if atype == "unanswerable":
        return {"type": "unanswerable", "reason": str(reason or answer or "")[:300]}
    if answer is None or (isinstance(answer, str) and not answer.strip()):
        raise ValueError("answer is empty; use type 'unanswerable' if the data cannot support the question")
    if isinstance(answer, str) and atype in ("list", "table", "number", "percent", "boolean"):
        try:
            answer = json.loads(answer)
        except ValueError:
            pass
    if atype == "number":
        return {"type": "number", "answer": _num_out(_to_number(answer))}
    if atype == "percent":
        return {"type": "percent", "answer": round_half_up(_to_number(answer), 2)}
    if atype == "string":
        if isinstance(answer, (list, dict)):
            raise ValueError("string answer must be a single name, not a list/dict")
        return {"type": "string", "answer": str(answer).strip()}
    if atype == "date":
        return {"type": "date", "answer": _date_out(answer)}
    if atype == "boolean":
        if isinstance(answer, bool):
            return {"type": "boolean", "answer": answer}
        t = str(answer).strip().lower()
        if t in ("true", "yes", "có", "co", "đúng", "dung", "1"):
            return {"type": "boolean", "answer": True}
        if t in ("false", "no", "không", "khong", "sai", "0"):
            return {"type": "boolean", "answer": False}
        raise ValueError("boolean must be true or false")
    if atype == "list":
        if not isinstance(answer, list):
            raise ValueError("list answer must be a JSON array")
        return {"type": "list", "answer": [_elem_out(e) for e in answer]}
    if atype == "table":
        if isinstance(answer, str):
            answer = json.loads(answer)
        if not (isinstance(answer, dict) and isinstance(answer.get("columns"), list) and isinstance(answer.get("rows"), list)):
            raise ValueError('table answer must be {"columns": [...], "rows": [[...], ...]}')
        ncol = len(answer["columns"])
        rows = []
        for r in answer["rows"]:
            if not isinstance(r, list) or len(r) != ncol:
                raise ValueError("every table row must be a list with one value per column")
            rows.append([_elem_out(c) for c in r])
        return {"type": "table", "answer": {"columns": [str(c) for c in answer["columns"]], "rows": rows}}
    raise ValueError("unreachable")


def canon(a: dict) -> str:
    """Hashable key used to compare answers across samples."""
    t = a["type"]
    if t == "unanswerable":
        return "unanswerable"
    v = a["answer"]
    if t in ("number", "percent"):
        return f"{t}:{round(float(v), 2)}"
    if t == "string":
        return f"string:{norm_str(v)}"
    if t == "date":
        return f"date:{v}"
    if t == "boolean":
        return f"boolean:{v}"
    if t == "list":
        return "list:" + "|".join(norm_str(e) if isinstance(e, str) else str(round(float(e), 2)) for e in v)
    if t == "table":
        rows = sorted("|".join(norm_str(c) if isinstance(c, str) else str(round(float(c), 2)) for c in r) for r in v["rows"])
        return "table:" + ";".join(rows)
    return json.dumps(a, sort_keys=True)


def expected_types(question: str) -> set[str]:
    """Weak linguistic prior on the answer type. Used only as a soft signal."""
    from .prep import unaccent
    q = unaccent(question).lower()
    exp: set[str] = set()
    if re.search(r"phan tram|% |ty le|ty trong|tang truong", q + " "):
        exp.add("percent")
    if re.search(r"\b(co phai|phai khong|dung khong|co .{1,40} khong)\b", q) and "bao nhieu" not in q:
        exp.add("boolean")
    if re.search(r"ngay nao|hom nao|khi nao|thang nao|ngay may", q):
        exp.update({"date", "string"})
    if re.search(r"\btop\s*\d+|liet ke|danh sach|nhung .{1,30} nao|cac .{1,30} nao|xep hang", q):
        exp.update({"list", "table"})
    if re.search(r"\bnhat\b", q) and not re.search(r"bao nhieu|tong|top\s*\d", q):
        exp.update({"string", "list"})
    if re.search(r"bao nhieu|tong |doanh thu|so luong|trung binh|may |tien", q):
        exp.update({"number", "percent"})
    if "unanswerable" not in exp:
        exp.add("unanswerable")
    return exp if len(exp) > 1 else set()


def vote(results: list[Optional[dict]], n_total: int, threshold: float, question: str = "") -> dict:
    """Self-consistency vote over samples. Returns {'final': submission-dict, 'confidence', 'info'}."""
    valid = [r for r in results if r]
    if not valid:
        return {"final": {"type": "abstain"}, "confidence": 0.0, "info": "no valid sample"}
    counts = Counter(canon(r) for r in valid)
    ranked = counts.most_common()
    top_key, top_n = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == top_n:
        return {"final": {"type": "abstain"}, "confidence": top_n / n_total, "info": f"tie {ranked[:2]}",
                "votes": dict(counts), "candidate": None}
    conf = top_n / max(n_total, 1)
    rep = next(r for r in valid if canon(r) == top_key)  # earliest sample = greedy sample first
    need = threshold
    exp = expected_types(question)
    suspect = bool(exp) and rep["type"] not in exp
    if suspect:
        need = 1.0
    if conf + 1e-9 < need:
        return {"final": {"type": "abstain"}, "confidence": conf,
                "info": f"low agreement {top_n}/{n_total} (need {need:.2f}{', type suspect' if suspect else ''})",
                "votes": dict(counts), "candidate": rep}
    return {"final": rep, "confidence": conf, "info": f"{top_n}/{n_total} agree", "votes": dict(counts),
            "candidate": rep}
