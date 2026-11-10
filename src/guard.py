"""
src/guard.py
Bộ phòng thủ kiểm tra tính khả giải (UNANSWERABLE vs ABSTAIN) và kiểm toán độ phủ ràng buộc
(Constraint Coverage Verification) để ngăn chặn hiện tượng bỏ rơi ràng buộc (Constraint Dropping)
và sai kiểu dữ liệu (WRONG_TYPE).
"""
import dataclasses
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple


def _strip_accents(text: str) -> str:
    norm = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in norm if unicodedata.category(ch) != "Mn").replace("đ", "d").replace("Đ", "D")


def _normalize_question(question: str) -> Tuple[str, str]:
    q_low = " ".join(question.lower().strip().split())
    q_ascii = _strip_accents(q_low)
    return q_low, q_ascii


@dataclass
class GuardVerdict:
    action: str  # "ALLOW" | "ABSTAIN" | "UNANSWERABLE"
    reason_code: Optional[str] = None
    missing_constraints: Tuple[str, ...] = ()
    expected_type_hint: Optional[str] = None


class ConstraintGuard:
    """
    Kiểm tra chéo giữa câu hỏi tiếng Việt gốc (kể cả teencode/viết tắt) và QueryPlanIR:
    1. Phân biệt rõ UNANSWERABLE (câu hỏi chủ quan / dữ liệu không tồn tại trong schema)
       với ABSTAIN (kế hoạch truy vấn làm rơi ràng buộc hoặc mâu thuẫn kiểu dữ liệu).
    2. Kiểm tra trên mọi intent (không chỉ giới hạn ở revenue_total hay order_count).
    """

    PERCENT_PAT = re.compile(
        r"(?:\bty\s*le\b|\bti\s*le\b|\bty\s*trong\b|\bti\s*trong\b|\bbao\s*nhieu\s*%\b|\bmay\s*%\b|\bchiem\s*%\b|%)"
    )
    WHO_PAT = re.compile(
        r"(?:\b(?:nv|nhan\s*vien|tho|khach|khach\s*hang|nguoi|ai)\s+nao\b|\bai\s+(?:la|ban|mua|lam|phu\s*trach|no)\b)"
    )
    WHAT_ENTITY_PAT = re.compile(
        r"(?:\b(?:sp|san\s*pham|mat\s*hang|mon|dich\s*vu|thuoc|lo\s*thuoc|khoan)\s+(?:nao|gi)\b|\bla\s+(?:sp|mon|mat\s*hang)\s+gi\b)"
    )
    WHEN_DATE_PAT = re.compile(
        r"(?:\bngay\s+nao\b|\bngay\s+may\b|\bvao\s+ngay\s+nao\b|\bthoi\s+diem\s+nao\b|\bbao\s+gio\s+het\s+han\b)"
    )
    TABLE_OR_LIST_PAT = re.compile(
        r"(?:\bliet\s+ke\b|\bdanh\s+sach\b|\bbang\s+thong\s+ke\b|\btop\s+\d+\b|\b\d+\s+(?:sp|san\s*pham|nv|nhan\s*vien|khach|mon)\s+dau\b)"
    )
    BOOLEAN_PAT = re.compile(
        r"(?:\bco\s+phai\b|\bdung\s+khong\b|\bco\s+hay\s+khong\b|\bphai\s+khong\b|\bco\s+vuot\b|\bco\s+dat\b)"
    )

    CANCEL_RETURN_PAT = re.compile(
        r"(?:\bhuy\b|\bdon\s+huy\b|\bbi\s+huy\b|\bhoan\b|\btra\s+hang\b|\bdoi\s+tra\b|\bboom\s+hang\b)"
    )
    SHIFT_PAT = re.compile(
        r"(?:\bca\s+(?:sang|chieu|toi|dem|1|2|3|hc|hanh\s*chinh)\b|\bbuoi\s+(?:sang|trua|chieu|toi)\b)"
    )
    CUSTOMER_GROUP_PAT = re.compile(
        r"(?:\bnhom\s+(?:khach|than\s*thiet|vip|vang\s*lai|si|le|moi|cu)\b|\bkhach\s+(?:than\s*thiet|vip|vang\s*lai|si|le|quen)\b|\bhang\s+the\b)"
    )
    STAFF_MENTION_PAT = re.compile(
        r"(?:\b(?:nv|nhan\s*vien|tho|ky\s*thuat\s*vien|ktv|bac\s*si|giao\s*vien)\b|\b(?:nv|kh|sp|hd|dh|bn|hs)\s*[-_]?\d{2,6}\b)"
    )
    EXPLICIT_CODE_PAT = re.compile(
        r"\b(?:nv|kh|sp|mh|hd|dh|don|lo|bn|hs|gv|bs|ktv)[-_]?\d{2,6}\b",
        re.IGNORECASE,
    )
    THRESHOLD_PAT = re.compile(
        r"(?:\b(?:tren|duoi|tu|hon|it\s+hon|nhieu\s+hon|cao\s+hon|thap\s+hon|vuot\s+qua|toi\s+thieu|toi\s+da|>=|<=|>|<)\s*\d+)"
    )
    TEMPORAL_PAT = re.compile(
        r"(?:\b(?:hom\s+nay|hom\s+qua|tuan\s+nay|tuan\s+roi|tuan\s+truoc|thang\s+nay|thang\s+roi|thang\s+truoc|quy\s+[1-4]|nam\s+nay|nam\s+ngoai|cung\s+ky)\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b\d{1,2}[/-]\d{4}\b|\bthang\s+\d{1,2}\b|\bram\b|\bmung\s+\d{1,2}\b|\bam\s+lich\b)"
    )
    SUBJECTIVE_UNANSWERABLE_PAT = re.compile(
        r"(?:\btai\s+sao\b|\bvi\s+sao\b|\bnguyen\s+nhan\s+nao\b|\bdu\s+doan\s+nam\s+sau\b|\bnen\s+chon\b|\bcam\s+xuc\b|\bthai\s+do\b|\bdanh\s+gia\s+chu\s+quan\b)"
    )

    GENERIC_INTENTS: Set[str] = {
        "revenue_total",
        "order_count",
        "total_revenue",
        "total_orders",
        "gross_revenue",
        "net_revenue",
    }

    def detect_question_cues(self, question: str) -> Dict[str, Any]:
        """Trích xuất tập tín hiệu ràng buộc và kỳ vọng kiểu dữ liệu từ câu hỏi gốc."""
        q_low, q_ascii = _normalize_question(question)

        explicit_codes = [m.group(0).upper() for m in self.EXPLICIT_CODE_PAT.finditer(q_ascii)]
        cues = {
            "percent_cue": bool(self.PERCENT_PAT.search(q_ascii)),
            "who_cue": bool(self.WHO_PAT.search(q_ascii)),
            "what_entity_cue": bool(self.WHAT_ENTITY_PAT.search(q_ascii)),
            "date_cue": bool(self.WHEN_DATE_PAT.search(q_ascii)),
            "table_cue": bool(self.TABLE_OR_LIST_PAT.search(q_ascii)),
            "boolean_cue": bool(self.BOOLEAN_PAT.search(q_ascii)),
            "cancel_cue": bool(self.CANCEL_RETURN_PAT.search(q_ascii)),
            "shift_cue": bool(self.SHIFT_PAT.search(q_ascii)),
            "customer_group_cue": bool(self.CUSTOMER_GROUP_PAT.search(q_ascii)),
            "staff_cue": bool(self.STAFF_MENTION_PAT.search(q_ascii)),
            "threshold_cue": bool(self.THRESHOLD_PAT.search(q_ascii)),
            "temporal_cue": bool(self.TEMPORAL_PAT.search(q_ascii)),
            "subjective_cue": bool(self.SUBJECTIVE_UNANSWERABLE_PAT.search(q_ascii)),
            "explicit_codes": explicit_codes,
        }
        return cues

    def has_constraint_cues(self, question: str) -> bool:
        cues = self.detect_question_cues(question)
        return any(
            bool(v)
            for k, v in cues.items()
            if k not in {"temporal_cue"}
        )

    @staticmethod
    def _plan_to_dict(plan: Any) -> Dict[str, Any]:
        if plan is None:
            return {}
        if isinstance(plan, dict):
            return dict(plan)
        if dataclasses.is_dataclass(plan):
            return dataclasses.asdict(plan)
        if hasattr(plan, "model_dump"):
            return plan.model_dump()
        if hasattr(plan, "dict"):
            return plan.dict()
        if hasattr(plan, "__dict__"):
            return dict(plan.__dict__)
        return {}

    def _has_non_empty_slot(self, plan_dict: Dict[str, Any], candidate_keys: Tuple[str, ...]) -> bool:
        for k in candidate_keys:
            val = plan_dict.get(k)
            if val not in (None, "", [], {}, False):
                return True
        for sub_key in ("filters", "entities", "linked_entities", "ir_slots", "params"):
            sub = plan_dict.get(sub_key)
            if isinstance(sub, dict):
                for k in candidate_keys:
                    if sub.get(k) not in (None, "", [], {}, False):
                        return True
            elif sub is not None and hasattr(sub, "__dict__"):
                for k in candidate_keys:
                    if getattr(sub, k, None) not in (None, "", [], {}, False):
                        return True
        return False

    def verify_plan(
        self,
        question: str,
        plan: Any,
        llm_meta: Optional[Dict[str, Any]] = None,
        schema_supports_question: Optional[bool] = None,
    ) -> GuardVerdict:
        """
        Kiểm tra tính toàn vẹn của QueryPlanIR đối với câu hỏi:
        - Trả về UNANSWERABLE nếu câu hỏi mang tính chủ quan hoặc schema không có dữ liệu.
        - Trả về ABSTAIN nếu phát hiện bất kỳ ràng buộc nào trong câu hỏi bị rớt khỏi plan
          hoặc kiểu dữ liệu trả về mâu thuẫn với câu hỏi.
        - Trả về ALLOW nếu toàn bộ ràng buộc đều được ánh xạ đầy đủ.
        """
        cues = self.detect_question_cues(question)

        if schema_supports_question is False:
            return GuardVerdict(action="UNANSWERABLE", reason_code="schema_unsupported")

        if llm_meta and llm_meta.get("answerability") == "unanswerable":
            if llm_meta.get("confidence", 0.0) >= 0.80 or cues["subjective_cue"]:
                return GuardVerdict(
                    action="UNANSWERABLE",
                    reason_code=llm_meta.get("unanswerable_reason") or "llm_confirmed_unanswerable",
                )

        plan_dict = self._plan_to_dict(plan)
        if not plan_dict:
            return GuardVerdict(action="ABSTAIN", reason_code="empty_plan")

        intent = str(plan_dict.get("intent") or plan_dict.get("ir_slots", {}).get("intent") or "").lower()
        target_type = str(
            plan_dict.get("target_type")
            or plan_dict.get("answer_type")
            or plan_dict.get("expected_type")
            or plan_dict.get("ir_slots", {}).get("target_type")
            or ""
        ).lower()

        if intent == "unanswerable" or target_type == "unanswerable":
            return GuardVerdict(action="UNANSWERABLE", reason_code="plan_marked_unanswerable")
        if intent in {"abstain", "unknown", "unknown_or_unsupported"} or target_type == "abstain":
            return GuardVerdict(action="ABSTAIN", reason_code="plan_marked_abstain")

        if llm_meta:
            unmapped = llm_meta.get("unmapped_constraints") or []
            if len(unmapped) > 0:
                return GuardVerdict(
                    action="ABSTAIN",
                    reason_code="llm_reported_unmapped_constraints",
                    missing_constraints=tuple(str(x) for x in unmapped),
                )

        missing: List[str] = []

        if cues["percent_cue"]:
            is_pct_intent = ("percent" in intent or "ratio" in intent or "share" in intent or "rate" in intent)
            if target_type and target_type != "percent" and not is_pct_intent:
                missing.append("percent_cue_mismatch")
            elif not target_type and intent in self.GENERIC_INTENTS:
                missing.append("percent_dropped_to_generic")

        if cues["who_cue"]:
            is_person_intent = any(w in intent for w in ("staff", "mechanic", "customer", "who", "top", "best", "person", "debtor"))
            if target_type in {"number", "percent", "boolean", "date"} or intent in self.GENERIC_INTENTS:
                if not is_person_intent:
                    missing.append("who_cue_dropped")

        if cues["date_cue"] and not cues["boolean_cue"]:
            is_date_intent = any(w in intent for w in ("date", "when", "day", "expiry", "peak_day", "best_day"))
            if target_type in {"number", "percent", "boolean"} or (intent in self.GENERIC_INTENTS and not is_date_intent):
                missing.append("date_cue_dropped")

        if cues["table_cue"]:
            is_struct_intent = any(w in intent for w in ("table", "list", "top", "rank", "breakdown", "summary"))
            if target_type in {"number", "percent", "boolean", "date"} or (intent in self.GENERIC_INTENTS and not is_struct_intent):
                missing.append("table_or_list_cue_dropped")

        if cues["cancel_cue"]:
            has_cancel_slot = self._has_non_empty_slot(
                plan_dict,
                ("only_canceled", "include_canceled", "is_canceled", "status_filter", "order_status", "canceled"),
            ) or any(w in intent for w in ("cancel", "return", "refund", "void", "huy"))
            if not has_cancel_slot:
                missing.append("cancel_constraint_dropped")

        if cues["shift_cue"]:
            has_shift_slot = self._has_non_empty_slot(
                plan_dict, ("shift", "work_shift", "time_of_day", "session")
            ) or "shift" in intent
            if not has_shift_slot:
                missing.append("shift_constraint_dropped")

        if cues["customer_group_cue"]:
            has_group_slot = self._has_non_empty_slot(
                plan_dict, ("customer_group", "customer_tier", "group", "segment", "customer_type")
            ) or any(w in intent for w in ("group", "segment", "tier", "customer"))
            if not has_group_slot and intent in self.GENERIC_INTENTS:
                missing.append("customer_group_constraint_dropped")

        if cues["threshold_cue"]:
            has_thresh_slot = self._has_non_empty_slot(
                plan_dict, ("threshold", "min_amount", "max_amount", "threshold_amount", "comparison_value", "limit")
            ) or any(w in intent for w in ("threshold", "above", "below", "filter", "over"))
            if not has_thresh_slot and intent in self.GENERIC_INTENTS:
                missing.append("threshold_constraint_dropped")

        if cues["explicit_codes"]:
            plan_str = str(plan_dict).upper()
            for code in cues["explicit_codes"]:
                code_norm = code.replace("-", "").replace("_", "")
                if code not in plan_str and code_norm not in plan_str.replace("-", "").replace("_", ""):
                    missing.append(f"explicit_entity_code_dropped:{code}")

        if missing:
            return GuardVerdict(
                action="ABSTAIN",
                reason_code=missing[0],
                missing_constraints=tuple(missing),
            )

        return GuardVerdict(action="ALLOW", reason_code=None)

    def check(self, question: str, plan: Any) -> Optional[str]:
        """Phương thức tương thích ngược với giao diện guard.check() hiện có."""
        verdict = self.verify_plan(question, plan)
        if verdict.action == "ALLOW":
            return None
        return verdict.reason_code or verdict.action.lower()


Guard = ConstraintGuard
