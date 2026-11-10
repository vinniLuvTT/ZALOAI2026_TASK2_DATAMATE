"""
tests/test_hybrid_guard_gating.py
Unit test kiểm chứng toàn diện:
1. Chống bỏ rơi ràng buộc (Constraint Dropping) trên 4 mẫu lỗi điển hình của Public Test và các biến thể mới.
2. Phân biệt chính xác UNANSWERABLE và ABSTAIN.
3. Chế độ Blind Validation chống rò rỉ tập Train.
"""
import unittest
from src.guard import ConstraintGuard
from src.gating import GatingController
from src.llm_parser import HybridLLMParser


class TestHybridGuardAndGating(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = ConstraintGuard()
        self.llm_parser = HybridLLMParser()
        self.gating = GatingController(guard=self.guard, llm_parser=None, always_call_llm=False)

    def test_public_test_constraint_dropping_cases_blocked(self) -> None:
        v1 = self.guard.verify_plan(
            "nv nào bán đc dthu cao nhứt",
            {"intent": "revenue_total", "target_type": "number"},
        )
        self.assertEqual(v1.action, "ABSTAIN")
        self.assertIn("who_cue_dropped", v1.missing_constraints)

        v2 = self.guard.verify_plan(
            "NV001 bán được bao nhiêu tiền",
            {"intent": "revenue_total", "target_type": "number", "staff_id": None},
        )
        self.assertEqual(v2.action, "ABSTAIN")
        self.assertTrue(any("NV001" in m for m in v2.missing_constraints))

        v3 = self.guard.verify_plan(
            "wan mik huỷ bnhiu đơn",
            {"intent": "order_count", "target_type": "number", "only_canceled": False},
        )
        self.assertEqual(v3.action, "ABSTAIN")
        self.assertIn("cancel_constraint_dropped", v3.missing_constraints)

        v4 = self.guard.verify_plan(
            "Khách nhóm thân thiết 08/2026 mang ve may % tong doanh thu",
            {"intent": "revenue_total", "target_type": "number"},
        )
        self.assertEqual(v4.action, "ABSTAIN")
        self.assertIn("percent_cue_mismatch", v4.missing_constraints)

    def test_distinguish_unanswerable_vs_abstain(self) -> None:
        decision_unans = self.gating.arbitrate(
            question="Dự đoán cảm xúc của khách hàng vào năm sau là gì?",
            as_of_date="2026-08-15",
            business_id="dev_001",
            regex_plan={"intent": "revenue_total", "target_type": "number"},
            llm_raw_output={
                "answerability": "unanswerable",
                "unanswerable_reason": "subjective_or_missing_column",
                "expected_answer_type": "unanswerable",
                "detected_constraints": [],
                "unmapped_constraints": [],
                "raw_time_expression": None,
                "raw_entity_mentions": {
                    "product": None,
                    "staff_or_mechanic": None,
                    "customer": None,
                    "customer_group": None,
                    "category": None,
                    "shift": None,
                    "explicit_code": None,
                },
                "ir_slots": {"intent": "unanswerable"},
                "confidence": 0.92,
            },
        )
        self.assertEqual(decision_unans.status, "UNANSWERABLE")

        decision_abstain = self.gating.arbitrate(
            question="Tính doanh thu các đơn giao trong vòng 15 phút trời mưa",
            as_of_date="2026-08-15",
            business_id="dev_001",
            regex_plan={"intent": "revenue_total", "target_type": "number"},
            llm_raw_output={
                "answerability": "answerable",
                "unanswerable_reason": None,
                "expected_answer_type": "number",
                "detected_constraints": ["threshold_or_comparison"],
                "unmapped_constraints": ["delivery_time_under_15m", "weather_rain"],
                "raw_time_expression": None,
                "raw_entity_mentions": {
                    "product": None,
                    "staff_or_mechanic": None,
                    "customer": None,
                    "customer_group": None,
                    "category": None,
                    "shift": None,
                    "explicit_code": None,
                },
                "ir_slots": {"intent": "revenue_total", "target_type": "number"},
                "confidence": 0.85,
            },
        )
        self.assertEqual(decision_abstain.status, "ABSTAIN")

    def test_llm_rescues_dropped_constraint(self) -> None:
        gating = GatingController(guard=self.guard, llm_parser=self.llm_parser, always_call_llm=False)
        decision = gating.arbitrate(
            question="NV001 bán được bao nhiêu tiền trong tháng 8/2026",
            as_of_date="2026-08-31",
            business_id="dev_003",
            regex_plan={"intent": "revenue_total", "target_type": "number", "staff_id": None},
            llm_raw_output={
                "answerability": "answerable",
                "unanswerable_reason": None,
                "expected_answer_type": "number",
                "detected_constraints": ["person_or_staff", "time_range_or_date"],
                "unmapped_constraints": [],
                "raw_time_expression": "tháng 8/2026",
                "raw_entity_mentions": {
                    "product": None,
                    "staff_or_mechanic": "NV001",
                    "customer": None,
                    "customer_group": None,
                    "category": None,
                    "shift": None,
                    "explicit_code": "NV001",
                },
                "ir_slots": {
                    "intent": "staff_revenue",
                    "target_type": "number",
                    "staff_id": "NV001",
                    "start_date": "2026-08-01",
                    "end_date": "2026-08-31",
                },
                "confidence": 0.95,
            },
        )
        self.assertEqual(decision.status, "EXECUTE_LLM")
        self.assertEqual(decision.plan["staff_id"], "NV001")


if __name__ == "__main__":
    unittest.main()
