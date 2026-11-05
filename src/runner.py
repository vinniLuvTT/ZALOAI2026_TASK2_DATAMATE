"""End-to-End Batch Runner for DataMate.

Coordinates:
1. Loading `LunarSolarCalendar` and `CanonicalBusiness` + `EntityLinker` per business
2. Question parsing (`parse_question`)
3. Rulebook-compliant execution (`execute_question`)
4. Verification & serialization (`result_to_record`, `write_submission_jsonl`)
5. Optional diagnostic trace logging (`artifacts/traces_<split>.jsonl`)
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.canonical import CanonicalBusiness, load_canonical_business
from src.entity_linker import EntityLinker
from src.executor import ExecutionResult, execute_question
from src.ir import SubmissionRecord
from src.parser import ParsedQuestion, parse_question
from src.serializer import result_to_record, write_submission_jsonl
from src.temporal import LunarSolarCalendar


class DataMateRunner:
    """Stateful runner with per-business in-memory and optional disk caching."""

    def __init__(
        self,
        split_dir: str | Path,
        config_path: str | Path = "configs/baseline.yaml",
    ) -> None:
        self.split_dir = Path(split_dir)
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        pipe_cfg = self.config.get("pipeline", {})
        self.confidence_threshold: float = float(pipe_cfg.get("confidence_threshold", 0.65))
        self.use_disk_cache: bool = bool(pipe_cfg.get("use_disk_cache", True))
        self.cache_dir: Optional[str] = pipe_cfg.get("cache_dir", "artifacts/cache")

        cal_path = self.split_dir / "calendar.csv"
        if not cal_path.exists():
            # Fallback to train/calendar.csv if split doesn't have one
            cal_path = Path("train/calendar.csv")
        self.calendar = LunarSolarCalendar(cal_path)

        self._biz_cache: Dict[str, Tuple[CanonicalBusiness, EntityLinker]] = {}

    def get_business_context(self, business_id: str) -> Tuple[CanonicalBusiness, EntityLinker]:
        if business_id not in self._biz_cache:
            cb = load_canonical_business(
                self.split_dir,
                business_id,
                cache_dir=self.cache_dir,
                use_cache=self.use_disk_cache,
            )
            linker = EntityLinker(cb)
            self._biz_cache[business_id] = (cb, linker)
        return self._biz_cache[business_id]

    def answer_single(
        self,
        q_record: Dict[str, Any],
    ) -> Tuple[SubmissionRecord, Optional[ParsedQuestion], ExecutionResult]:
        qid = str(q_record.get("id", ""))
        biz_id = str(q_record.get("business_id", ""))
        try:
            cb, linker = self.get_business_context(biz_id)
            pq = parse_question(q_record, cb, linker, self.calendar)
            exec_res = execute_question(pq, cb, self.calendar)
            sub_rec = result_to_record(qid, exec_res, confidence_threshold=self.confidence_threshold)
            return sub_rec, pq, exec_res
        except Exception as exc:
            err_msg = f"Exception ({type(exc).__name__}): {exc}"
            fallback_res = ExecutionResult(answer_type="abstain", value=None, reason=err_msg, confidence=0.0)
            sub_rec = result_to_record(qid, fallback_res, confidence_threshold=self.confidence_threshold)
            return sub_rec, None, fallback_res

    def run_file(
        self,
        questions_path: str | Path,
        output_path: str | Path,
        trace_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        q_path = Path(questions_path)
        questions: List[Dict[str, Any]] = []
        with open(q_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    questions.append(json.loads(s))

        records: List[SubmissionRecord] = []
        traces: List[Dict[str, Any]] = []
        type_counts: Dict[str, int] = {}

        for q in questions:
            sub_rec, pq, exec_res = self.answer_single(q)
            records.append(sub_rec)
            type_counts[sub_rec.type] = type_counts.get(sub_rec.type, 0) + 1

            if trace_path is not None:
                trace_item: Dict[str, Any] = {
                    "id": sub_rec.id,
                    "business_id": q.get("business_id"),
                    "question": q.get("question"),
                    "as_of_date": q.get("as_of_date"),
                    "pred_type": sub_rec.type,
                    "pred_answer": sub_rec.answer,
                    "reason": exec_res.reason,
                    "confidence": exec_res.confidence,
                }
                if pq is not None:
                    trace_item["intent"] = pq.intent
                    trace_item["time_label"] = pq.time_window.label
                    trace_item["start_date"] = pq.time_window.start_date.isoformat()
                    trace_item["end_date"] = pq.time_window.end_date.isoformat()
                    trace_item["product_key"] = pq.entities.product_key
                    trace_item["product_name"] = pq.entities.product_name
                    trace_item["customer_key"] = pq.entities.customer_key
                    trace_item["category_norm"] = pq.entities.category_norm
                    trace_item["target_unit_norm"] = pq.entities.target_unit_norm
                traces.append(trace_item)

        n_written = write_submission_jsonl(records, output_path)

        if trace_path is not None:
            tp = Path(trace_path)
            tp.parent.mkdir(parents=True, exist_ok=True)
            with open(tp, "w", encoding="utf-8") as f:
                for tr in traces:
                    f.write(json.dumps(tr, ensure_ascii=False) + "\n")

        elapsed = time.perf_counter() - t0
        return {
            "total_questions": len(questions),
            "written_records": n_written,
            "elapsed_sec": round(elapsed, 3),
            "qps": round(len(questions) / max(elapsed, 1e-6), 2),
            "type_distribution": type_counts,
        }
