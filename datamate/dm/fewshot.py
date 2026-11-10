"""
Dynamic few-shot bank (train data only).

Entries: {"business_id","question","type","answer","sql"?}
 - built from train questions/answers (question + answer type)
 - enriched by `run.py eval --harvest`: SQL of runs that were scored correct (rejection sampling)
Retrieval: char n-gram TF-IDF on unaccented lowercase question. At eval time pass
exclude_biz so the same business never leaks into its own prompt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .prep import unaccent


def _prep(q: str) -> str:
    return unaccent(q).lower()


class FewShotBank:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.items: list[dict] = []
        self._vec = None
        self._mat = None
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.items.append(json.loads(line))

    def add_train(self, questions_path: Path, answers_path: Path) -> None:
        qs = {j["id"]: j for j in map(json.loads, filter(str.strip, Path(questions_path).read_text(encoding="utf-8").splitlines()))}
        for line in Path(answers_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            a = json.loads(line)
            q = qs.get(a["id"])
            if q:
                self.items.append({"id": a["id"], "business_id": q["business_id"], "question": q["question"],
                                   "type": a["type"], "answer": a.get("answer")})
        self._vec = None

    def add_harvest(self, qid: str, biz: str, question: str, atype: str, answer, sql: list[str]) -> None:
        for it in self.items:
            if it.get("id") == qid:
                it["sql"] = sql[-2:]
                return
        self.items.append({"id": qid, "business_id": biz, "question": question, "type": atype,
                           "answer": answer, "sql": sql[-2:]})
        self._vec = None

    def save(self, path: str | Path | None = None) -> None:
        p = Path(path) if path else self.path
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in self.items), encoding="utf-8")

    def _index(self):
        if self._vec is not None or not self.items:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self._mat = self._vec.fit_transform([_prep(i["question"]) for i in self.items])

    def top(self, question: str, k: int = 3, exclude_biz: Optional[str] = None) -> list[dict]:
        if not self.items or k <= 0:
            return []
        try:
            self._index()
            sims = (self._mat @ self._vec.transform([_prep(question)]).T).toarray().ravel()
        except Exception:
            return []
        order = sims.argsort()[::-1]
        out, seen_types = [], 0
        for i in order:
            it = self.items[i]
            if exclude_biz and it["business_id"] == exclude_biz:
                continue
            if sims[i] < 0.2:
                break
            out.append(it)
            if len(out) >= k:
                break
        # prefer examples that carry SQL
        out.sort(key=lambda it: 0 if it.get("sql") else 1)
        return out

    def render(self, question: str, k: int, exclude_biz: Optional[str] = None) -> str:
        ex = self.top(question, k, exclude_biz)
        if not ex:
            return ""
        lines = ["Similar solved questions from OTHER businesses (column names differ; use for answer type & approach only):"]
        for it in ex:
            lines.append(f"- Q: {it['question']}  -> answer type: {it['type']}")
            for s in it.get("sql", []):
                lines.append(f"    sql: {s[:400]}")
        return "\n".join(lines)
