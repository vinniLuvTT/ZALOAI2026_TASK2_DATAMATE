"""
src/llm_schema.py
Tự động nội soi (introspect) cấu trúc QueryPlanIR từ src/ir.py lúc chạy (runtime)
để sinh JSON Schema cho vLLM Guided Decoding mà không bị gán cứng (hardcode) thiếu trường.
"""
import dataclasses
import inspect
from typing import Any, Dict, List, Optional, Type, get_args, get_origin, get_type_hints


CONSTRAINT_CATEGORIES: List[str] = [
    "person_or_staff",
    "mechanic_or_technician",
    "customer_or_group",
    "product_or_service",
    "category_or_group",
    "topping_or_addon",
    "time_range_or_date",
    "lunar_calendar_date",
    "shift_or_time_of_day",
    "cancellation_or_return",
    "negative_or_void_adjustment",
    "threshold_or_comparison",
    "percentage_or_ratio",
    "ranking_or_top_k",
    "debt_or_overdue",
    "inventory_or_expiry",
    "cogs_or_profit_margin",
    "attendance_or_timesheet",
    "grade_or_score",
    "branch_or_channel",
    "payment_method",
    "table_or_list_output",
]

ANSWER_TYPES: List[str] = [
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
]


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Chuyển đổi kiểu dữ liệu Python/typing sang JSON Schema tương thích vLLM."""
    if py_type is None or py_type is type(None):
        return {"type": "null"}

    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is not None:
        if str(origin).endswith("Union"):
            non_none = [a for a in args if a is not type(None)]
            has_none = len(non_none) != len(args)
            if len(non_none) == 1:
                sub = _python_type_to_json_schema(non_none[0])
                if has_none and "type" in sub and isinstance(sub["type"], str):
                    sub = dict(sub)
                    sub["type"] = [sub["type"], "null"]
                return sub
            return {"anyOf": [_python_type_to_json_schema(a) for a in args]}

        if origin in (list, List, tuple, set):
            item_type = args[0] if args else Any
            return {
                "type": "array",
                "items": _python_type_to_json_schema(item_type),
            }

        if origin in (dict, Dict):
            return {"type": "object"}

    if py_type is str:
        return {"type": ["string", "null"]}
    if py_type is int:
        return {"type": ["integer", "null"]}
    if py_type is float:
        return {"type": ["number", "null"]}
    if py_type is bool:
        return {"type": ["boolean", "null"]}

    return {"type": ["string", "number", "boolean", "object", "array", "null"]}


def introspect_ir_fields(ir_cls: Optional[Type[Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Đọc tự động toàn bộ các trường của QueryPlanIR từ src.ir nếu có."""
    if ir_cls is None:
        try:
            from src.ir import QueryPlanIR as _QueryPlanIR
            ir_cls = _QueryPlanIR
        except Exception:
            ir_cls = None

    properties: Dict[str, Dict[str, Any]] = {}
    if ir_cls is not None:
        if dataclasses.is_dataclass(ir_cls):
            type_hints = get_type_hints(ir_cls)
            for f in dataclasses.fields(ir_cls):
                f_type = type_hints.get(f.name, f.type)
                properties[f.name] = _python_type_to_json_schema(f_type)
        elif hasattr(ir_cls, "model_fields"):
            for name, f_info in ir_cls.model_fields.items():
                properties[name] = _python_type_to_json_schema(f_info.annotation)
        elif hasattr(ir_cls, "__fields__"):
            for name, f_info in ir_cls.__fields__.items():
                properties[name] = _python_type_to_json_schema(f_info.outer_type_)
        else:
            sig = inspect.signature(ir_cls.__init__)
            for name, param in sig.parameters.items():
                if name != "self":
                    properties[name] = _python_type_to_json_schema(param.annotation)

    default_slots = {
        "intent": {"type": "string"},
        "metric": {"type": ["string", "null"]},
        "target_type": {"type": "string", "enum": ANSWER_TYPES},
        "start_date": {"type": ["string", "null"]},
        "end_date": {"type": ["string", "null"]},
        "product_key": {"type": ["string", "null"]},
        "staff_id": {"type": ["string", "null"]},
        "mechanic_name": {"type": ["string", "null"]},
        "customer_id": {"type": ["string", "null"]},
        "customer_group": {"type": ["string", "null"]},
        "category": {"type": ["string", "null"]},
        "shift": {"type": ["string", "null"]},
        "status_filter": {"type": ["string", "null"]},
        "include_canceled": {"type": ["boolean", "null"]},
        "only_canceled": {"type": ["boolean", "null"]},
        "threshold": {"type": ["number", "null"]},
        "threshold_op": {"type": ["string", "null"]},
        "top_k": {"type": ["integer", "null"]},
        "group_by": {"type": ["array", "null"], "items": {"type": "string"}},
        "sort_order": {"type": ["string", "null"], "enum": ["asc", "desc", None]},
    }
    for k, v in default_slots.items():
        if k not in properties:
            properties[k] = v

    return properties


def build_semantic_parse_schema(
    ir_cls: Optional[Type[Any]] = None,
    allowed_intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Sinh JSON Schema hoàn chỉnh cho LLM Semantic Parser:
    - Tách biệt rõ `answerability` (answerable vs unanswerable vs uncertain).
    - Yêu cầu liệt kê `detected_constraints` và `unmapped_constraints` để chống bỏ rơi ràng buộc.
    - Giữ nguyên biểu thức thời gian thô (`raw_time_expression`) và thực thể thô (`raw_entity_mentions`)
      để đưa qua `src/temporal.py` và `src/entity_linker.py` chuẩn hóa.
    """
    ir_properties = introspect_ir_fields(ir_cls)
    if allowed_intents:
        ir_properties["intent"] = {"type": "string", "enum": allowed_intents}

    return {
        "type": "object",
        "properties": {
            "answerability": {
                "type": "string",
                "enum": ["answerable", "unanswerable", "uncertain"],
            },
            "unanswerable_reason": {
                "type": ["string", "null"],
            },
            "expected_answer_type": {
                "type": "string",
                "enum": ANSWER_TYPES,
            },
            "detected_constraints": {
                "type": "array",
                "items": {"type": "string", "enum": CONSTRAINT_CATEGORIES},
            },
            "unmapped_constraints": {
                "type": "array",
                "items": {"type": "string"},
            },
            "raw_time_expression": {
                "type": ["string", "null"],
            },
            "raw_entity_mentions": {
                "type": "object",
                "properties": {
                    "product": {"type": ["string", "null"]},
                    "staff_or_mechanic": {"type": ["string", "null"]},
                    "customer": {"type": ["string", "null"]},
                    "customer_group": {"type": ["string", "null"]},
                    "category": {"type": ["string", "null"]},
                    "shift": {"type": ["string", "null"]},
                    "explicit_code": {"type": ["string", "null"]},
                },
                "required": [
                    "product",
                    "staff_or_mechanic",
                    "customer",
                    "customer_group",
                    "category",
                    "shift",
                    "explicit_code",
                ],
            },
            "ir_slots": {
                "type": "object",
                "properties": ir_properties,
                "required": ["intent"],
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": [
            "answerability",
            "expected_answer_type",
            "detected_constraints",
            "unmapped_constraints",
            "raw_time_expression",
            "raw_entity_mentions",
            "ir_slots",
            "confidence",
        ],
    }
