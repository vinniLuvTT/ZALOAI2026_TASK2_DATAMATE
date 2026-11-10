"""
src/gating.py
Bộ điều phối Gating kết hợp Regex Parser + LLM 14B Semantic Parser + ConstraintGuard.
Đảm bảo:
1. Tách biệt rạch ròi giữa `UNANSWERABLE` và `ABSTAIN`.
2. Chống rơi ràng buộc (Constraint Dropping) và sai kiểu dữ liệu (WRONG_TYPE).
3. Hỗ trợ chế độ `blind_validation=True` để đánh giá khách quan trên tập Held-out Validation
   mà không bị nhiễu bởi các regex đã khớp sẵn trên tập Train.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.guard import ConstraintGuard, GuardVerdict
from src.llm_parser import HybridLLMParser


@dataclass
class GatingDecision:
    status: str  # "EXECUTE_REGEX" | "EXECUTE_LLM" | "UNANSWERABLE" | "ABSTAIN"
    plan: Optional[Any]
    reason: str
    guard_verdict: Optional[GuardVerdict] = None
    llm_meta: Optional[Dict[str, Any]] = None


class GatingController:
    def __init__(
        self,
        guard: Optional[ConstraintGuard] = None,
        llm_parser: Optional[HybridLLMParser] = None,
        confidence_threshold: float = 0.75,
        blind_validation: bool = False,
        always_call_llm: bool = True,
    ) -> None:
        self.guard = guard or ConstraintGuard()
        self.llm_parser = llm_parser
        self.confidence_threshold = confidence_threshold
        self.blind_validation = blind_validation
        self.always_call_llm = always_call_llm

    def arbitrate(
        self,
        question: str,
        as_of_date: str,
        business_id: str,
        regex_plan: Optional[Any],
        llm_raw_output: Optional[Dict[str, Any]] = None,
        temporal_resolver: Optional[Callable[..., Any]] = None,
        entity_linker: Optional[Any] = None,
        schema_supports_question: Optional[bool] = None,
        readme_summary: str = "",
        schema_summary: str = "",
        entity_candidates_summary: str = "",
    ) -> GatingDecision:
        """
        Phân xử giữa kế hoạch truy vấn từ Regex Router và LLM Semantic Parser.
        """
        regex_verdict = (
            self.guard.verify_plan(
                question,
                regex_plan,
                schema_supports_question=schema_supports_question,
            )
            if regex_plan is not None and not self.blind_validation
            else GuardVerdict(action="ABSTAIN", reason_code="blind_validation_or_no_regex")
        )

        if regex_verdict.action == "UNANSWERABLE":
            return GatingDecision(
                status="UNANSWERABLE",
                plan=None,
                reason=regex_verdict.reason_code or "regex_detected_unanswerable",
                guard_verdict=regex_verdict,
            )

        if llm_raw_output is None and self.llm_parser is not None:
            should_call = (
                self.always_call_llm
                or self.blind_validation
                or regex_verdict.action != "ALLOW"
                or self.guard.has_constraint_cues(question)
            )
            if should_call:
                llm_raw_output = self.llm_parser.call_llm_structured(
                    question=question,
                    as_of_date=as_of_date,
                    business_id=business_id,
                    readme_summary=readme_summary,
                    schema_summary=schema_summary,
                    entity_candidates_summary=entity_candidates_summary,
                )

        llm_plan = None
        llm_verdict = None
        if llm_raw_output is not None:
            ans_flag = llm_raw_output.get("answerability")
            llm_conf = float(llm_raw_output.get("confidence", 0.0))

            if ans_flag == "unanswerable" and llm_conf >= self.confidence_threshold:
                return GatingDecision(
                    status="UNANSWERABLE",
                    plan=None,
                    reason=llm_raw_output.get("unanswerable_reason") or "llm_unanswerable",
                    llm_meta=llm_raw_output,
                )

            if ans_flag == "uncertain" or len(llm_raw_output.get("unmapped_constraints") or []) > 0:
                return GatingDecision(
                    status="ABSTAIN",
                    plan=None,
                    reason="llm_uncertain_or_unmapped_constraints",
                    llm_meta=llm_raw_output,
                )

            if self.llm_parser is not None:
                llm_plan = self.llm_parser.materialize_ir(
                    question=question,
                    as_of_date=as_of_date,
                    llm_result=llm_raw_output,
                    temporal_resolver=temporal_resolver,
                    entity_linker=entity_linker,
                    fallback_regex_plan=None if self.blind_validation else regex_plan,
                )
            else:
                llm_plan = llm_raw_output.get("ir_slots")

            llm_verdict = self.guard.verify_plan(
                question=question,
                plan=llm_plan,
                llm_meta=llm_raw_output,
                schema_supports_question=schema_supports_question,
            )

            if llm_verdict.action == "UNANSWERABLE":
                return GatingDecision(
                    status="UNANSWERABLE",
                    plan=None,
                    reason=llm_verdict.reason_code or "guard_llm_unanswerable",
                    guard_verdict=llm_verdict,
                    llm_meta=llm_raw_output,
                )

        if self.blind_validation:
            if (
                llm_plan is not None
                and llm_verdict is not None
                and llm_verdict.action == "ALLOW"
                and float(llm_raw_output.get("confidence", 0.0)) >= self.confidence_threshold
            ):
                return GatingDecision(
                    status="EXECUTE_LLM",
                    plan=llm_plan,
                    reason="blind_val_llm_verified",
                    guard_verdict=llm_verdict,
                    llm_meta=llm_raw_output,
                )
            return GatingDecision(
                status="ABSTAIN",
                plan=None,
                reason=(llm_verdict.reason_code if llm_verdict else "blind_val_abstain") or "blind_val_abstain",
                guard_verdict=llm_verdict,
                llm_meta=llm_raw_output,
            )

        if regex_verdict.action != "ALLOW":
            if (
                llm_plan is not None
                and llm_verdict is not None
                and llm_verdict.action == "ALLOW"
                and float(llm_raw_output.get("confidence", 0.0)) >= self.confidence_threshold
            ):
                return GatingDecision(
                    status="EXECUTE_LLM",
                    plan=llm_plan,
                    reason=f"llm_rescued_{regex_verdict.reason_code}",
                    guard_verdict=llm_verdict,
                    llm_meta=llm_raw_output,
                )
            return GatingDecision(
                status="ABSTAIN",
                plan=None,
                reason=regex_verdict.reason_code or "guard_blocked_regex",
                guard_verdict=regex_verdict,
                llm_meta=llm_raw_output,
            )

        if llm_raw_output is not None and llm_verdict is not None:
            regex_dict = self.guard._plan_to_dict(regex_plan)
            r_intent = str(regex_dict.get("intent") or "").lower()
            l_intent = str((llm_raw_output.get("ir_slots") or {}).get("intent") or "").lower()
            llm_conf = float(llm_raw_output.get("confidence", 0.0))

            if r_intent and r_intent == l_intent:
                return GatingDecision(
                    status="EXECUTE_REGEX",
                    plan=regex_plan,
                    reason="consensus_regex_llm",
                    guard_verdict=regex_verdict,
                    llm_meta=llm_raw_output,
                )

            if (
                r_intent in self.guard.GENERIC_INTENTS
                and l_intent not in self.guard.GENERIC_INTENTS
                and llm_verdict.action == "ALLOW"
                and llm_conf >= self.confidence_threshold
            ):
                return GatingDecision(
                    status="EXECUTE_LLM",
                    plan=llm_plan,
                    reason=f"llm_specialized_intent:{l_intent}",
                    guard_verdict=llm_verdict,
                    llm_meta=llm_raw_output,
                )

            detected_constraints = llm_raw_output.get("detected_constraints") or []
            if (
                r_intent != l_intent
                and len(detected_constraints) >= 2
                and r_intent in self.guard.GENERIC_INTENTS
                and llm_verdict.action != "ALLOW"
            ):
                return GatingDecision(
                    status="ABSTAIN",
                    plan=None,
                    reason="conflict_on_constrained_question",
                    guard_verdict=llm_verdict,
                    llm_meta=llm_raw_output,
                )

        return GatingDecision(
            status="EXECUTE_REGEX",
            plan=regex_plan,
            reason="regex_verified_allow",
            guard_verdict=regex_verdict,
            llm_meta=llm_raw_output,
        )
