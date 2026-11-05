"""Integration tests for the end-to-end DataMate pipeline (`src/runner.py`)."""

from __future__ import annotations

import json
import unittest

from src.canonical import load_canonical_business
from src.entity_linker import EntityLinker
from src.executor import execute_question
from src.parser import parse_question
from src.serializer import serialize_prediction_line
from src.temporal import LunarSolarCalendar
from src.verifier import verify_and_gate


class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cal = LunarSolarCalendar("train/calendar.csv")
        cls.cb_106 = load_canonical_business("train", "biz_106", use_cache=True)
        cls.linker_106 = EntityLinker(cls.cb_106)

    def test_exact_revenue_and_order_status_filtering(self) -> None:
        q = {
            "id": "test_rev_may",
            "business_id": "biz_106",
            "question": "Tháng 5/2025 doanh thu là bao nhiêu?",
            "as_of_date": "2026-06-30",
        }
        pq = parse_question(q, self.cb_106, self.linker_106, self.cal)
        res = verify_and_gate(execute_question(pq, self.cb_106, self.cal))
        self.assertEqual(res.answer_type, "number")
        self.assertEqual(res.value, 271866000)

    def test_as_of_date_prevents_future_leakage(self) -> None:
        q = {
            "id": "test_future",
            "business_id": "biz_106",
            "question": "Doanh thu tháng 12/2027 là bao nhiêu?",
            "as_of_date": "2026-06-30",
        }
        pq = parse_question(q, self.cb_106, self.linker_106, self.cal)
        res = verify_and_gate(execute_question(pq, self.cb_106, self.cal))
        self.assertEqual(res.answer_type, "unanswerable")
        self.assertIsNone(res.value)

    def test_subjective_question_unanswerable(self) -> None:
        q = {
            "id": "test_subj",
            "business_id": "biz_106",
            "question": "Nhà cung cấp nào uy tín và đáng tin cậy nhất năm 2025?",
            "as_of_date": "2026-06-30",
        }
        pq = parse_question(q, self.cb_106, self.linker_106, self.cal)
        res = verify_and_gate(execute_question(pq, self.cb_106, self.cal))
        self.assertEqual(res.answer_type, "unanswerable")
        line = serialize_prediction_line(q["id"], res)
        parsed = json.loads(line)
        self.assertEqual(parsed, {"id": "test_subj", "type": "unanswerable", "answer": None})


if __name__ == "__main__":
    unittest.main()
