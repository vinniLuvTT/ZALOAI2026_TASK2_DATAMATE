"""
End-to-end runner for one split (train / public test / private test).

per business:  prepare (DuckDB) -> Cube model -> start Cube -> answer questions (threads) -> stop Cube
per question:  N samples (agent) -> early stop when unanimous -> vote -> verify -> submission line

Crash/timeout safety: every answer is appended + flushed to submission.jsonl immediately; a re-run
resumes. When the time budget is exhausted, remaining questions are written as abstain.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from . import cube_gen
from .agent import Solver, build_user_message, compact_schema
from .config import Config
from .cube_server import CubeServer
from .fewshot import FewShotBank
from .prep import BizContext, RuleOptions, prepare_business
from .timeutil import Calendar
from .tools import SQLRunner, ToolBox
from .verify import canon, vote


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


class Split:
    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root
        self.data_dir = root / "data" if (root / "data").is_dir() else root
        self.questions = read_jsonl(root / "questions.jsonl")
        self.rulebook = (root / "rulebook.md").read_text(encoding="utf-8") if (root / "rulebook.md").exists() else ""
        self.calendar = Calendar.load(root / "calendar.csv")
        self.answers = {a["id"]: a for a in read_jsonl(root / "answers.jsonl")} if (root / "answers.jsonl").exists() else {}


def _role_llm(cfg: Config):
    if not cfg.refine_roles_llm:
        return None
    from openai import OpenAI
    client = OpenAI(base_url=cfg.llm_url_list[0], api_key=cfg.llm_api_key, timeout=cfg.llm_timeout)

    def call(prompt: str) -> str:
        r = client.chat.completions.create(
            model=cfg.llm_model, messages=[{"role": "user", "content": prompt}], temperature=0, seed=cfg.seed,
            max_tokens=2048, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        return r.choices[0].message.content or ""
    return call


def _submission_line(qid: str, final: dict) -> dict:
    line = {"id": qid, "type": final["type"]}
    if final["type"] == "unanswerable":
        line["reason"] = final.get("reason", "")
    elif final["type"] != "abstain":
        line["answer"] = final["answer"]
    return line


class Runner:
    def __init__(self, cfg: Config, split: Split, out_dir: Path, bank: Optional[FewShotBank] = None,
                 exclude_same_business_fewshot: bool = False):
        self.cfg, self.split, self.out_dir = cfg, split, Path(out_dir)
        self.bank = bank
        self.exclude_same = exclude_same_business_fewshot
        self.solver = Solver(cfg)
        self.lock = threading.Lock()
        self.t0 = time.time()
        (self.out_dir / "logs").mkdir(parents=True, exist_ok=True)
        self.sub_path = self.out_dir / "submission.jsonl"
        self.records: list[dict] = []

    # ------------------------------------------------------------------
    def _over_budget(self) -> bool:
        return time.time() - self.t0 > self.cfg.time_budget_s

    def _write(self, line: dict, record: dict) -> None:
        with self.lock:
            with open(self.sub_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                f.flush()
            (self.out_dir / "logs" / f"{record['id']}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
            self.records.append(record)

    # ------------------------------------------------------------------
    def answer_question(self, q: dict, ctx: BizContext, sql: SQLRunner, cube: Optional[CubeServer],
                        cube_catalog: str, shared: dict) -> None:
        cfg = self.cfg
        t0 = time.time()
        if self._over_budget():
            self._write({"id": q["id"], "type": "abstain"}, {
                "id": q["id"], "business_id": q["business_id"], "question": q["question"],
                "final": {"type": "abstain"}, "info": "time budget exhausted", "elapsed": 0})
            return
        fewshot = self.bank.render(q["question"], cfg.n_fewshot,
                                   q["business_id"] if self.exclude_same else None) if self.bank else ""
        user = build_user_message(q, ctx, self.split.rulebook, cfg, fewshot, compact_schema(ctx),
                                  cube_catalog, cube is not None)
        samples, answers = [], []
        for i in range(cfg.n_samples):
            if self._over_budget():
                break
            tb = ToolBox(cfg, ctx, self.split.rulebook, self.split.calendar, cube, cube_catalog, sql, shared)
            r = self.solver.solve_sample(q, tb, user, i)
            samples.append(r)
            answers.append(r.answer)
            valid = [a for a in answers if a]
            if cfg.early_stop and len(valid) >= cfg.early_stop and len({canon(a) for a in valid}) == 1 \
                    and len(valid) == len(answers):
                break
        v = vote(answers, max(len(samples), 1), cfg.agree_threshold, q["question"])
        final = v["final"]
        best_sql = next((s.sql for s in samples if s.answer and v.get("candidate") and canon(s.answer) == canon(v["candidate"])), [])
        record = {
            "id": q["id"], "business_id": q["business_id"], "question": q["question"], "as_of_date": q["as_of_date"],
            "final": final, "candidate": v.get("candidate"), "confidence": v["confidence"], "info": v["info"],
            "votes": v.get("votes"), "elapsed": round(time.time() - t0, 2), "sql": best_sql,
            "samples": [{"answer": s.answer, "steps": s.steps, "error": s.error, "elapsed": s.elapsed,
                         "trace": s.trace[-12:]} for s in samples],
        }
        self._write(_submission_line(q["id"], final), record)

    # ------------------------------------------------------------------
    def run(self, businesses: Optional[list[str]] = None, limit: Optional[int] = None) -> list[dict]:
        cfg = self.cfg
        done = set()
        if self.sub_path.exists():
            done = {json.loads(l)["id"] for l in self.sub_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        by_biz: dict[str, list[dict]] = {}
        for q in self.split.questions:
            if q["id"] in done or (businesses and q["business_id"] not in businesses):
                continue
            by_biz.setdefault(q["business_id"], []).append(q)
        if limit:
            by_biz = {b: qs[:limit] for b, qs in by_biz.items()}

        for biz, qs in sorted(by_biz.items()):
            print(f"[run] {biz}: {len(qs)} questions  (elapsed {time.time() - self.t0:.0f}s)", flush=True)
            if self._over_budget():
                for q in qs:
                    self._write({"id": q["id"], "type": "abstain"}, {"id": q["id"], "business_id": biz,
                                "question": q["question"], "final": {"type": "abstain"}, "info": "time budget exhausted"})
                continue
            self._run_business(biz, qs)
        return self.records

    def _run_business(self, biz: str, qs: list[dict]) -> None:
        cfg = self.cfg
        cube: Optional[CubeServer] = None
        sql: Optional[SQLRunner] = None
        try:
            opts = RuleOptions.from_rulebook(self.split.rulebook)
            ctx = prepare_business(biz, self.split.data_dir / biz, cfg.work_dir, llm=_role_llm(cfg), opts=opts)
            catalog = ""
            if cfg.use_cube and ctx.ok:
                try:
                    catalog = cube_gen.write_model(ctx, ctx.work_dir / "model")
                    srv = CubeServer(cfg, ctx.work_dir, ctx.cube_db)
                    if srv.start():
                        cube = srv
                    else:
                        print(f"[run] {biz}: Cube unavailable ({srv.error}); SQL only", flush=True)
                        srv.stop()
                        catalog = ""
                except Exception as e:
                    print(f"[run] {biz}: Cube setup failed ({e}); SQL only", flush=True)
            sql = SQLRunner(ctx.sql_db, cfg.sql_timeout_s)
            shared: dict = {}
            with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
                futs = [ex.submit(self._safe_answer, q, ctx, sql, cube, catalog, shared) for q in qs]
                for f in futs:
                    f.result()
        except Exception as e:  # business-level failure -> abstain everything rather than guess
            print(f"[run] {biz} FAILED: {type(e).__name__}: {e}", flush=True)
            for q in qs:
                self._write({"id": q["id"], "type": "abstain"}, {
                    "id": q["id"], "business_id": biz, "question": q["question"],
                    "final": {"type": "abstain"}, "info": f"business failed: {e}"})
        finally:
            if cube:
                cube.stop()

    def _safe_answer(self, q, ctx, sql, cube, catalog, shared) -> None:
        try:
            self.answer_question(q, ctx, sql, cube, catalog, shared)
        except Exception as e:
            self._write({"id": q["id"], "type": "abstain"}, {
                "id": q["id"], "business_id": q["business_id"], "question": q["question"],
                "final": {"type": "abstain"}, "info": f"error: {type(e).__name__}: {e}"})
