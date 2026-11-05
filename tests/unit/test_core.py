"""Unit tests for DataMate core components:
- Evaluator (`src/evaluator.py`)
- Value parser & rounding (`src/value_parser.py`)
- Temporal & Lunar-Solar calendar resolver (`src/temporal.py`)
- Rulebook status normalization (`src/rule_engine.py`)
- Question Telex/chat normalization (`src/canonical.py`)
- Verification & serializer (`src/verifier.py`, `src/serializer.py`)
"""

from __future__ import annotations

import datetime as dt
import json
import unittest

from src.canonical import is_walkin_customer, normalize_customer_group, normalize_payment_method, normalize_question_text
from src.evaluator import evaluate_predictions, score_single_record
from src.executor import ExecutionResult
from src.rule_engine import normalize_order_status, round_2dec, round_money_vnd
from src.serializer import serialize_prediction_line
from src.temporal import LunarSolarCalendar, parse_flexible_date, parse_flexible_hour, resolve_question_time_window
from src.value_parser import parse_vn_number
from src.verifier import verify_and_gate


class TestValueParserAndRounding(unittest.TestCase):
    def test_parse_vn_numbers(self) -> None:
        self.assertEqual(parse_vn_number("1.250.000đ"), 1250000.0)
        self.assertEqual(parse_vn_number("450k"), 450000.0)
        self.assertEqual(parse_vn_number("1tr5"), 1500000.0)
        self.assertEqual(parse_vn_number("1,2 triệu"), 1200000.0)
        self.assertEqual(parse_vn_number("-35.000"), -35000.0)

    def test_half_up_rounding(self) -> None:
        self.assertEqual(round_money_vnd(1234.5), 1235)
        self.assertEqual(round_money_vnd(1234.4), 1234)
        self.assertEqual(round_2dec(12.345), 12.35)
        self.assertEqual(round_2dec(12.344), 12.34)


class TestRuleEngineAndNormalization(unittest.TestCase):
    def test_order_status_normalization(self) -> None:
        self.assertEqual(normalize_order_status("Đang vận chuyển"), "delivering")
        self.assertEqual(normalize_order_status("shipping"), "delivering")
        self.assertEqual(normalize_order_status("Đã hoàn hàng"), "returned")
        self.assertEqual(normalize_order_status("trả lại"), "returned")
        self.assertEqual(normalize_order_status("Không giao được"), "cancelled")
        self.assertEqual(normalize_order_status("Huỷ đơn"), "cancelled")
        self.assertEqual(normalize_order_status("Hoàn thành", "Thước thủy nhôm 60cm"), "completed")
        self.assertEqual(normalize_order_status("Hoàn tất", "ghi sổ, khách trả hết rồi"), "completed")

    def test_question_telex_normalization(self) -> None:
        raw = "Thangs 5/2025 cuwar hangf bans dduwowcj bnhiu ddown, dthu laf bao nhieeu tieenf?"
        norm = normalize_question_text(raw)
        self.assertIn("tháng 5/2025", norm)
        self.assertIn("cửa hàng", norm)
        self.assertIn("được bao nhiêu đơn", norm)
        self.assertIn("doanh thu", norm)

    def test_walkin_and_payment_normalization(self) -> None:
        self.assertFalse(is_walkin_customer("", "KH0001"))
        self.assertTrue(is_walkin_customer("Khách lẻ", ""))
        self.assertTrue(is_walkin_customer("lẻ", ""))
        self.assertEqual(normalize_payment_method("Thu hộ (COD)"), "cod")
        self.assertEqual(normalize_customer_group("Thân thiết"), "than_thiet")


class TestTemporalResolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cal = LunarSolarCalendar("train/calendar.csv")

    def test_explicit_quarter_vs_month(self) -> None:
        as_of = dt.date(2026, 5, 15)
        tw_q = resolve_question_time_window("Doanh thu quý 4/2025 là bao nhiêu?", as_of, self.cal)
        self.assertEqual(tw_q.start_date, dt.date(2025, 10, 1))
        self.assertEqual(tw_q.end_date, dt.date(2025, 12, 31))

        tw_m = resolve_question_time_window("Doanh thu tháng 4/2025 là bao nhiêu?", as_of, self.cal)
        self.assertEqual(tw_m.start_date, dt.date(2025, 4, 1))
        self.assertEqual(tw_m.end_date, dt.date(2025, 4, 30))

    def test_tet_window_and_comparison(self) -> None:
        as_of = dt.date(2026, 6, 1)
        tw = resolve_question_time_window(
            "Dịp Tết âm lịch 2026 bán ra nhiều hay ít tiền hơn 17 ngày trước Tết, chênh bao nhiêu %?",
            as_of,
            self.cal,
        )
        self.assertEqual(tw.label, "tet_2026")
        self.assertIsNotNone(tw.compare_window)
        self.assertEqual((tw.end_date - tw.start_date).days + 1, 17)
        self.assertEqual((tw.compare_window.end_date - tw.compare_window.start_date).days + 1, 17)

    def test_multi_month_consecutive(self) -> None:
        as_of = dt.date(2026, 6, 1)
        tw = resolve_question_time_window(
            "Cho doanh thu của 3 tháng liền nhau kết thúc ở tháng 10/2025 theo thứ tự thời gian?",
            as_of,
            self.cal,
        )
        self.assertEqual(len(tw.multi_windows), 3)
        self.assertEqual([w.label for w in tw.multi_windows], ["2025-08", "2025-09", "2025-10"])

    def test_flexible_date_and_hour(self) -> None:
        self.assertEqual(parse_flexible_date("15/08/2025"), dt.date(2025, 8, 15))
        self.assertEqual(parse_flexible_date("15/08", default_year=2025), dt.date(2025, 8, 15))
        self.assertEqual(parse_flexible_hour("2025-08-15 14:35:00"), 14)


class TestEvaluatorAndVerifier(unittest.TestCase):
    def test_evaluator_scoring_rules(self) -> None:
        gold = [
            {"id": "q1", "type": "number", "answer": 1500000},
            {"id": "q2", "type": "percent", "answer": 12.34},
            {"id": "q3", "type": "string", "answer": "Nguyễn Văn A"},
            {"id": "q4", "type": "unanswerable", "answer": None},
            {"id": "q5", "type": "list", "answer": [10, 20, 30]},
        ]
        pred = [
            {"id": "q1", "type": "number", "answer": 1500000},
            {"id": "q2", "type": "abstain", "answer": None},
            {"id": "q3", "type": "string", "answer": "  nguyễn   văn A "},
            {"id": "q4", "type": "unanswerable", "answer": None},
            {"id": "q5", "type": "list", "answer": [10, 99, 30]},
        ]
        import tempfile
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f_g, \
             tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as f_p:
            for g in gold:
                f_g.write(json.dumps(g, ensure_ascii=False) + "\n")
            for p in pred:
                f_p.write(json.dumps(p, ensure_ascii=False) + "\n")
            f_g.flush()
            f_p.flush()
            res = evaluate_predictions(f_g.name, f_p.name)
            self.assertEqual(res["counts"]["CORRECT"], 3)
            self.assertEqual(res["counts"]["ABSTAIN"], 1)
            self.assertEqual(res["counts"]["WRONG_VALUE"], 1)
            self.assertEqual(res["total_score"], 2)  # +3 - 1 + 0 = +2

    def test_score_single_record(self) -> None:
        g = {"id": "q1", "type": "number", "answer": 100}
        p_ok = {"id": "q1", "type": "number", "answer": 100}
        p_abs = {"id": "q1", "type": "abstain", "answer": None}
        self.assertEqual(score_single_record(p_ok, g)[1], "CORRECT")
        self.assertEqual(score_single_record(p_abs, g)[1], "ABSTAIN")

    def test_verifier_and_serializer(self) -> None:
        r_ok = verify_and_gate(ExecutionResult("date", "2025-08"))
        self.assertEqual(r_ok.answer_type, "date")
        self.assertEqual(r_ok.value, "2025-08")

        r_nan = verify_and_gate(ExecutionResult("number", float("nan")))
        self.assertEqual(r_nan.answer_type, "abstain")
        self.assertIsNone(r_nan.value)

        line = serialize_prediction_line("q_test", r_ok)
        self.assertEqual(line, '{"id": "q_test", "type": "date", "answer": "2025-08"}')


if __name__ == "__main__":
    unittest.main()
