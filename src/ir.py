"""Typed Intermediate Representation (IR) and Output Schemas using Pydantic."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


AnswerType = Literal[
    "number",
    "percent",
    "date",
    "boolean",
    "string",
    "list",
    "table",
    "unanswerable",
    "abstain",
]


class FilterCond(BaseModel):
    field: str
    op: Literal["eq", "neq", "gt", "gte", "lt", "lte", "in", "contains"] = "eq"
    value: Any


class QueryPlanIR(BaseModel):
    """Validated Intermediate Representation of a Vietnamese question."""

    question_id: str
    business_id: str
    as_of_date: str
    intent: str = Field(
        description="High-level query intent, e.g. revenue, order_count, cancel_rate, top_product, debt_total, etc."
    )
    answer_type: AnswerType
    time_expr: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    compare_start_date: Optional[str] = None
    compare_end_date: Optional[str] = None
    multi_periods: List[Dict[str, str]] = Field(default_factory=list)
    shift: Optional[Literal["morning", "afternoon"]] = None
    entity_query: Optional[str] = None
    entity_query_2: Optional[str] = None
    unit_query: Optional[str] = None
    category_query: Optional[str] = None
    customer_group_query: Optional[str] = None
    channel_query: Optional[str] = None
    payment_method_query: Optional[str] = None
    threshold_value: Optional[float] = None
    threshold_op: Optional[Literal["gt", "gte", "lt", "lte", "eq"]] = None
    top_k: int = 1
    window_days: Optional[int] = None
    weekday_filter: Optional[int] = None
    filters: List[FilterCond] = Field(default_factory=list)
    unsupported_reason: Optional[str] = None
    ambiguities: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class SubmissionRecord(BaseModel):
    """Official submission record schema."""

    id: str
    type: AnswerType
    answer: Optional[Any] = None
    reason: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        if self.type == "abstain":
            return {"id": self.id, "type": "abstain"}
        if self.type == "unanswerable":
            out: Dict[str, Any] = {"id": self.id, "type": "unanswerable"}
            if isinstance(self.reason, str) and self.reason.strip():
                out["reason"] = self.reason.strip()
            return out
        return {"id": self.id, "type": self.type, "answer": self.answer}
