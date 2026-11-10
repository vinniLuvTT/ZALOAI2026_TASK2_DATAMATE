"""
src/llm_parser.py
Bộ phân tích ngữ nghĩa sử dụng mô hình 14B (Qwen3-14B / Qwen2.5-Coder-14B-Instruct) qua vLLM,
kết hợp hậu xử lý tất định qua `src/temporal.py` và `src/entity_linker.py` để tạo ra `QueryPlanIR`
chuẩn xác cho `src/executor.py`.
"""
import dataclasses
import inspect
import json
import logging
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Type

from src.llm_schema import build_semantic_parse_schema

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là bộ phân tích ngữ nghĩa có cấu trúc (Structured Semantic Parser) cho hệ thống trả lời câu hỏi dữ liệu bảng tiếng Việt (DataMate).
Nhiệm vụ của bạn là đọc câu hỏi tiếng Việt (có thể chứa viết tắt, khẩu ngữ, teencode như: sp, nv, dthu, bnhiu, wan mik, k, dc, nhứt), ngày tham chiếu `as_of_date`, và tóm tắt schema/thực thể của doanh nghiệp để xuất ra JSON tuân thủ nghiêm ngặt schema được cấp.

NGUYÊN TẮC BẮT BUỘC (CHỐNG OVERFIT & CHỐNG BỎ RƠI RÀNG BUỘC):
1. KHÔNG tự tính toán số học và KHÔNG tự suy đoán ngày lịch âm-dương. Hãy trích xuất nguyên văn cụm thời gian vào `raw_time_expression` để bộ máy `temporal.py` giải quyết theo `as_of_date` và lịch âm-dương chuẩn.
2. Trích xuất đầy đủ mọi thực thể được nhắc tới vào `raw_entity_mentions` (sản phẩm, nhân viên/thợ sửa xe, khách hàng, nhóm khách hàng, danh mục, ca làm việc, mã định danh như NV001).
3. Kiểm toán toàn bộ điều kiện lọc trong câu hỏi:
   - Ghi nhận mọi loại điều kiện xuất hiện trong câu hỏi vào `detected_constraints`.
   - Nếu câu hỏi có bất kỳ điều kiện lọc hoặc yêu cầu nghiệp vụ nào mà các trường trong `ir_slots` KHÔNG thể biểu diễn được, BẮT BUỘC phải ghi rõ điều kiện đó vào mảng `unmapped_constraints` (để hệ thống chủ động ABSTAIN thay vì trả lời sai).
4. Phân biệt rõ `answerability`:
   - Đặt `answerability = "unanswerable"` CHỈ KHI câu hỏi đòi hỏi thông tin không hề tồn tại trong schema/README của doanh nghiệp hoặc câu hỏi mang tính chủ quan/dự đoán tương lai không có căn cứ dữ liệu.
   - Đặt `answerability = "uncertain"` nếu câu hỏi mơ hồ không thể xác định duy nhất một phép tính.
   - Đặt `answerability = "answerable"` khi dữ liệu bảng của doanh nghiệp hỗ trợ đầy đủ.
5. Kiểm tra chặt chẽ `expected_answer_type`:
   - Câu hỏi hỏi "ai", "nhân viên nào", "thợ nào", "mặt hàng gì", "sản phẩm nào" -> `expected_answer_type = "string"` (hoặc `"list"` nếu hỏi nhiều đối tượng). Tuyệt đối KHÔNG trả về `"number"`.
   - Câu hỏi hỏi "bao nhiêu %", "chiếm tỷ lệ", "tỷ trọng" -> `expected_answer_type = "percent"`.
   - Câu hỏi hỏi "ngày nào", "hôm nào", "khi nào" -> `expected_answer_type = "date"`."""


class HybridLLMParser:
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
        timeout_sec: float = 15.0,
        ir_cls: Optional[Type[Any]] = None,
        allowed_intents: Optional[List[str]] = None,
    ) -> None:
        self.api_url = f"{api_base.rstrip('/')}/chat/completions"
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.ir_cls = ir_cls or self._load_ir_cls()
        self.json_schema = build_semantic_parse_schema(self.ir_cls, allowed_intents=allowed_intents)

    @staticmethod
    def _load_ir_cls() -> Optional[Type[Any]]:
        try:
            from src.ir import QueryPlanIR
            return QueryPlanIR
        except Exception:
            return None

    def call_llm_structured(
        self,
        question: str,
        as_of_date: str,
        business_id: str,
        readme_summary: str = "",
        schema_summary: str = "",
        entity_candidates_summary: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Gọi vLLM với Guided JSON Schema Decoding để lấy bản phân tích ngữ nghĩa."""
        user_prompt = (
            f"Business ID: {business_id}\n"
            f"Ngày tham chiếu (as_of_date): {as_of_date}\n"
            f"Tóm tắt README & Nghiệp vụ: {readme_summary or 'Theo chuẩn Rulebook chung'}\n"
            f"Schema bảng/cột hiện có: {schema_summary or 'Canonical Tables'}\n"
            f"Ứng viên thực thể trong DB: {entity_candidates_summary or 'Tra cứu qua EntityLinker'}\n"
            f"Câu hỏi: {json.dumps(question, ensure_ascii=False)}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 600,
            "response_format": {
                "type": "json_object",
                "schema": self.json_schema,
            },
            "guided_json": self.json_schema,
        }

        try:
            req = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return json.loads(content)
        except Exception as exc:
            logger.debug("vLLM structured call không khả dụng hoặc lỗi (%s)", exc)
            return None

    def materialize_ir(
        self,
        question: str,
        as_of_date: str,
        llm_result: Dict[str, Any],
        temporal_resolver: Optional[Callable[..., Any]] = None,
        entity_linker: Optional[Any] = None,
        fallback_regex_plan: Optional[Any] = None,
    ) -> Any:
        """
        Chuyển kết quả JSON của LLM thành đối tượng `QueryPlanIR` chuẩn của hệ thống:
        - Luôn ưu tiên giải quyết thời gian bằng `src/temporal.py` từ `raw_time_expression` + `as_of_date`.
        - Luôn liên kết thực thể qua `src/entity_linker.py` để khớp canonical key (bao gồm cả `mechanic_name` / `mechanic_norm`).
        """
        slots: Dict[str, Any] = dict(llm_result.get("ir_slots") or {})
        raw_entities: Dict[str, Any] = dict(llm_result.get("raw_entity_mentions") or {})
        raw_time = llm_result.get("raw_time_expression")

        if temporal_resolver is not None:
            try:
                time_query = raw_time if raw_time else question
                resolved_time = temporal_resolver(time_query, as_of_date)
                if isinstance(resolved_time, dict):
                    slots.update({k: v for k, v in resolved_time.items() if v is not None})
                elif isinstance(resolved_time, (tuple, list)) and len(resolved_time) >= 2:
                    slots["start_date"], slots["end_date"] = resolved_time[0], resolved_time[1]
            except Exception as exc:
                logger.debug("Temporal resolver fallback: %s", exc)

        if fallback_regex_plan is not None:
            for time_field in ("start_date", "end_date", "period", "time_window", "comparison_range"):
                if slots.get(time_field) is None and hasattr(fallback_regex_plan, time_field):
                    val = getattr(fallback_regex_plan, time_field)
                    if val is not None:
                        slots[time_field] = val

        if entity_linker is not None:
            try:
                if hasattr(entity_linker, "link"):
                    linked = entity_linker.link(question)
                    if linked is not None:
                        slots["linked_entities"] = linked
                        for attr in ("product_key", "staff_id", "mechanic_name", "mechanic_norm", "customer_id", "category"):
                            if hasattr(linked, attr) and getattr(linked, attr) is not None and not slots.get(attr):
                                slots[attr] = getattr(linked, attr)
            except Exception as exc:
                logger.debug("EntityLinker fallback: %s", exc)

        if raw_entities.get("staff_or_mechanic") and not slots.get("staff_id") and not slots.get("mechanic_name"):
            slots["staff_id"] = raw_entities["staff_or_mechanic"]
        if raw_entities.get("product") and not slots.get("product_key"):
            slots["product_key"] = raw_entities["product"]
        if raw_entities.get("customer_group") and not slots.get("customer_group"):
            slots["customer_group"] = raw_entities["customer_group"]
        if raw_entities.get("shift") and not slots.get("shift"):
            slots["shift"] = raw_entities["shift"]

        if llm_result.get("expected_answer_type") and not slots.get("target_type"):
            slots["target_type"] = llm_result["expected_answer_type"]

        if self.ir_cls is None:
            return slots

        try:
            if dataclasses.is_dataclass(self.ir_cls):
                valid_fields = {f.name for f in dataclasses.fields(self.ir_cls)}
                filtered_kwargs = {}
                if fallback_regex_plan is not None and dataclasses.is_dataclass(fallback_regex_plan):
                    filtered_kwargs.update(dataclasses.asdict(fallback_regex_plan))
                for k, v in slots.items():
                    if k in valid_fields and v is not None:
                        filtered_kwargs[k] = v
                return self.ir_cls(**filtered_kwargs)
            else:
                sig = inspect.signature(self.ir_cls.__init__)
                valid_params = {k for k in sig.parameters.keys() if k != "self"}
                filtered_kwargs = {k: v for k, v in slots.items() if k in valid_params and v is not None}
                return self.ir_cls(**filtered_kwargs)
        except Exception as exc:
            logger.debug("Không thể khởi tạo trực tiếp QueryPlanIR (%s), trả về dict/fallback", exc)
            return slots
