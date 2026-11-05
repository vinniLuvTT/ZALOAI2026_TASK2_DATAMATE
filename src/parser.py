"""High-recall Vietnamese question parser for DataMate.

Combines:
1. `resolve_question_time_window` from `src/temporal.py`
2. `EntityLinker` from `src/entity_linker.py`
3. Rulebook-aware intent classification covering all general and domain-specific rules
   from `train/rulebook.md` and `public_test/rulebook.md`.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import List, Optional

from src.canonical import CanonicalBusiness, clean_str, norm_key, normalize_question_text, strip_accents
from src.entity_linker import EntityLinker, LinkedEntities
from src.ir import AnswerType, QueryPlanIR
from src.temporal import LunarSolarCalendar, TimeWindow, resolve_question_time_window
from src.value_parser import parse_vn_number


SUBJECTIVE_PATTERNS = [
    "trung thuc nhat",
    "nhiet tinh nhat",
    "dang tin cay",
    "nha cung cap uy tin",
    "than thien nhat",
    "kho tinh nhat",
    "chat luong tot nhat",
    "ngon nhat",
    "lanh nghe nhat",
    "tay nghe cao nhat",
    "khach hay khen",
    "tan tam",
    "le phep",
    "chu dao",
    "quy men nhat",
    "nen uong moi lan may vien",
]

CANCEL_REASON_PATTERNS = [
    "vi giao tre",
    "do giao tre",
    "goi nham mon",
    "mon bi loi",
    "mua nham",
    "ly do huy",
    "nguyen nhan huy",
    "vi sao bi huy",
    "tai sao bi huy",
    "bi huy vi",
    "bi huy do",
    "huy do giao",
    "huy do khach",
    "huy do het",
]


@dataclass
class ParsedQuestion:
    question_id: str
    business_id: str
    question_text: str
    as_of_date: dt.date
    intent: str
    expected_type: AnswerType
    time_window: TimeWindow
    entities: LinkedEntities
    top_k: int = 1
    sort_asc: bool = False
    threshold_value: Optional[float] = None
    threshold_op: Optional[str] = None
    debt_age_days: Optional[int] = None
    unsupported_reason: Optional[str] = None
    confidence: float = 1.0

    def to_ir(self) -> QueryPlanIR:
        return QueryPlanIR(
            question_id=self.question_id,
            business_id=self.business_id,
            as_of_date=self.as_of_date.isoformat(),
            intent=self.intent,
            answer_type=self.expected_type,
            time_expr=self.time_window.label,
            start_date=self.time_window.start_date.isoformat(),
            end_date=self.time_window.end_date.isoformat(),
            compare_start_date=self.time_window.compare_window.start_date.isoformat() if self.time_window.compare_window else None,
            compare_end_date=self.time_window.compare_window.end_date.isoformat() if self.time_window.compare_window else None,
            shift=self.time_window.shift_filter,  # type: ignore[arg-type]
            entity_query=self.entities.product_name or self.entities.customer_name or self.entities.mechanic_name,
            unit_query=self.entities.target_unit_norm,
            category_query=self.entities.category_display,
            customer_group_query=self.entities.customer_group_norm,
            channel_query=self.entities.channel_norm,
            payment_method_query=self.entities.payment_method_norm,
            threshold_value=self.threshold_value,
            threshold_op=self.threshold_op,  # type: ignore[arg-type]
            top_k=self.top_k,
            weekday_filter=self.time_window.specific_weekday,
            unsupported_reason=self.unsupported_reason,
            confidence=self.confidence,
        )


def _extract_top_k(q_nk: str) -> int:
    m = re.search(
        r"\b(?:top\s*(\d+)|(\d+)\s+(?:khach|mat hang|san pham|mon|sp|dong may|thuoc|nguoi|nhan vien|tho|xe|hang))\b",
        q_nk,
    )
    if m:
        k = int(m.group(1) or m.group(2))
        if 1 <= k <= 20:
            return k
    return 1


def parse_question(
    q_record: dict,
    cb: CanonicalBusiness,
    linker: EntityLinker,
    cal: LunarSolarCalendar,
) -> ParsedQuestion:
    qid = str(q_record["id"])
    biz_id = str(q_record["business_id"])
    q_text = clean_str(normalize_question_text(str(q_record["question"])))
    as_of_str = str(q_record["as_of_date"]).strip()
    as_of_dt = dt.date.fromisoformat(as_of_str[:10])

    q_low = q_text.lower()
    q_nk = norm_key(q_text)
    q_noacc = strip_accents(q_low)

    tw = resolve_question_time_window(q_text, as_of_dt, cal, data_start_date=cb.data_start_date)
    ent = linker.link(q_text)
    top_k = _extract_top_k(q_nk)

    # ------------------------------------------------------------------
    # 0. Subjective / Cancel Reason / Missing Category Unanswerable Checks
    # ------------------------------------------------------------------
    for pat in SUBJECTIVE_PATTERNS:
        if pat in q_nk:
            return ParsedQuestion(
                question_id=qid,
                business_id=biz_id,
                question_text=q_text,
                as_of_date=as_of_dt,
                intent="unanswerable_subjective",
                expected_type="unanswerable",
                time_window=tw,
                entities=ent,
                unsupported_reason="Câu hỏi mang tính đánh giá chủ quan hoặc kiến thức ngoài dữ liệu.",
            )

    for pat in CANCEL_REASON_PATTERNS:
        if pat in q_nk:
            return ParsedQuestion(
                question_id=qid,
                business_id=biz_id,
                question_text=q_text,
                as_of_date=as_of_dt,
                intent="unanswerable_cancel_reason",
                expected_type="unanswerable",
                time_window=tw,
                entities=ent,
                unsupported_reason="Dữ liệu của cửa hàng không ghi nhận lý do huỷ hoặc trả đơn.",
            )

    if ent.requested_category_missing:
        return ParsedQuestion(
            question_id=qid,
            business_id=biz_id,
            question_text=q_text,
            as_of_date=as_of_dt,
            intent="unanswerable_missing_category",
            expected_type="unanswerable",
            time_window=tw,
            entities=ent,
            unsupported_reason="Dữ liệu không có cột phân loại nhóm mặt hàng được hỏi.",
        )

    # ------------------------------------------------------------------
    # 1. Domain: Auto Repair (dev_012)
    # ------------------------------------------------------------------
    if cb.domain == "auto_repair":
        return _parse_auto_repair_question(qid, biz_id, q_text, q_nk, as_of_dt, tw, ent, top_k)

    # ------------------------------------------------------------------
    # 2. Installments & Warranties & Trade-in (Phone Shop dev_011, dev_014)
    # ------------------------------------------------------------------
    if "tra gop" in q_nk or "ky gop" in q_nk or "tien gop" in q_nk or "cham gop" in q_nk or "qua han gop" in q_nk or "gop toi han" in q_nk:
        if "toi han may ky" in q_nk or "may ky gop roi" in q_nk:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "installment_due_period_count", "number", tw, ent)
        if ("qua han" in q_nk or "cham gop" in q_nk) and any(k in q_nk for k in ("may khach", "bao nhieu khach", "bnhiu khach", "mi ngui", "may nguoi")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "installment_overdue_customer_count", "number", tw, ent)
        if "qua han" in q_nk or "cham gop" in q_nk:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "installment_overdue_amount", "number", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "installment_unpaid_amount", "number", tw, ent)

    if "bao hanh" in q_nk:
        if any(k in q_nk for k in ("ti le", "ty le", "chiem bao nhieu %", "chiem bnhiu %", "chiem may %", "so voi so may")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "warranty_rate_pct", "percent", tw, ent)
        if "dong may nao" in q_nk or "may nao" in q_nk and ("nhieu" in q_nk or "nhat" in q_nk or "nhut" in q_nk):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "warranty_top_model", "string", tw, ent)
        if any(k in q_nk for k in ("may ban ra", "may ban thang", "da mang lai bao hanh", "da duoc mang toi bao hanh")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "warranty_purchased_in_period_count", "number", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "warranty_count", "number", tw, ent)

    if "thu may cu" in q_nk or "mua may cu" in q_nk or "may cu thu lai" in q_nk:
        if any(k in q_nk for k in ("chiem may %", "bao nhieu %", "bnhiu %", "ti le")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "trade_in_order_pct", "percent", tw, ent)
        if "khach phai tra" not in q_nk:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "trade_in_total_amount", "number", tw, ent)

    # ------------------------------------------------------------------
    # 3. Deliveries & Payment Due Dates (Construction)
    # ------------------------------------------------------------------
    if any(k in q_nk for k in ("dot giao", "dot hang", "luot giao", "cho giao", "da giao", "da cho")):
        if any(k in q_nk for k in ("tu 2 dot", "2 dot tro len")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "delivery_multi_round_order_count", "number", tw, ent)
        if ent.product_key:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "delivery_product_qty", "number", tw, ent)
        if any(k in q_nk for k in ("dot giao", "dot hang", "cho giao", "giao duoc bao nhieu dot", "giao tong cong may dot")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "delivery_round_count", "number", tw, ent)

    if "ngay phai tra" in q_nk or "han thanh toan" in q_nk or "qua han thanh toan" in q_nk:
        if "muon hon hom nay" in q_nk or "sau hom nay" in q_nk or "chua toi han" in q_nk:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "due_date_after_today_count", "number", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "due_date_overdue_count", "number", tw, ent)

    # ------------------------------------------------------------------
    # 4. Mother & Baby Specifics (Voucher, Points, Shipping Fee, New Members)
    # ------------------------------------------------------------------
    if "voucher" in q_nk and any(k in q_nk for k in ("giam", "tien", "bao nhieu", "bnhiu", "bn")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "voucher_discount_total", "number", tw, ent)
    if ("dung diem" in q_nk or "doi diem" in q_nk or "nho dung diem" in q_nk) and any(k in q_nk for k in ("giam", "tien", "bao nhieu", "bnhiu", "bn")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "points_discount_total", "number", tw, ent)
    if "phi ship" in q_nk or "phi giao hang" in q_nk or "phi van chuyen" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "shipping_fee_total", "number", tw, ent)
    if ("thanh vien moi" in q_nk or "khach moi dang ky" in q_nk or "dang ky moi" in q_nk) and cb.domain == "mother_baby":
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "new_member_count", "number", tw, ent)

    # ------------------------------------------------------------------
    # 5. Debt / Credit / Repayment / Aging
    # ------------------------------------------------------------------
    if any(k in q_nk for k in ("no them", "ghi no them", "mua chiu them")) and any(k in q_nk for k in ("tra no", "tra bot no", "tra duoc")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "customer_debt_added_and_repaid", "list", tw, ent)

    if any(k in q_nk for k in ("tra bot no", "khach tra no", "thu no", "thu hoi no")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "debt_repaid_in_period", "number", tw, ent)

    if any(k in q_nk for k in ("tuoi no", "0 7 ngay", "8 15 ngay", "tren 15 ngay")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "debt_aging_buckets", "list", tw, ent)

    m_debt_age = re.search(r"(?:dat\s+qua|tuoi\s+qua|no\s+qua|qua)\s+(\d+)\s+ngay", q_nk)
    if m_debt_age and ("no" in q_nk or "chua tra" in q_nk):
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "debt_over_n_days", "number", tw, ent, debt_age_days=int(m_debt_age.group(1))
        )

    if any(k in q_nk for k in ("con no", "cong no", "chua tra het no", "dang no")):
        if any(k in q_nk for k in ("moi nguoi", "tung nguoi", "kem so tien", "va so no", "cho bang")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "debt_top_customers_table", "table", tw, ent, top_k=max(top_k, 3))
        if any(k in q_nk for k in ("bao nhieu khach", "bnhiu khach", "bn khach", "so khach", "may khach", "co may khach")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "debt_customer_count", "number", tw, ent)
        if any(k in q_nk for k in ("khach nao", "ai dang no", "top")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "debt_top_customers_list", "list" if top_k > 1 else "string", tw, ent, top_k=top_k)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "debt_total_remaining", "number", tw, ent)

    # ------------------------------------------------------------------
    # 6. Stock, Out-of-Stock, & Batch Expiry
    # ------------------------------------------------------------------
    if any(k in q_nk for k in ("gan het han nhat", "can date nhat", "can date nhut", "hsd som nhat", "han dung som nhat")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "nearest_expiry_date", "date", tw, ent)

    if any(k in q_nk for k in ("het han trong", "co han dung tu", "se het han tu")):
        if any(k in q_nk for k in ("tien von", "dong bao nhieu tien", "dong bnhiu tien", "bao nhieu tien von")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "expiring_stock_cost", "number", tw, ent)
        if any(k in q_nk for k in ("may lo", "bao nhieu lo", "bnhiu lo", "so lo")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "expiring_batch_count", "number", tw, ent)
        if any(k in q_nk for k in ("may mat hang", "bao nhieu mat hang", "so mat hang", "may san pham", "bao nhieu san pham")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "expiring_product_count", "number", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "expiring_product_list", "list", tw, ent)

    if any(k in q_nk for k in ("dang het hang", "bi het hang", "het hang luc cuoi ngay", "het hang cuoi ngay")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "out_of_stock_product_count", "number", tw, ent)

    if any(k in q_nk for k in ("ton kho", "ton cuoi", "kho con ton", "trong kho con", "nam trong kho")) or (
        any(k in q_nk for k in ("cuoi ngay", "het ngay", "het hom nay", "cuoi bua nay", "het bua nay", "den het"))
        and (
            ent.target_unit_pair is not None
            or any(k in q_nk for k in ("con bao nhieu", "con bnhiu", "con bn", "con may", "con lai bao nhieu", "con lai may", "trong kho"))
        )
    ):
        if any(k in q_nk for k in ("gia von", "tien hang nam trong kho", "gia tri ton kho", "tien ton kho")) and not ent.product_key:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "stock_ending_cost_value", "number", tw, ent)
        ans_t: AnswerType = "list" if ent.target_unit_pair else "number"
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "stock_ending_qty", ans_t, tw, ent)

    # ------------------------------------------------------------------
    # 7. Imports & Suppliers
    # ------------------------------------------------------------------
    if "nha cung cap chinh" in q_nk or ("nha cung cap" in q_nk and "nhieu phieu nhap nhat" in q_nk):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "main_supplier", "string", tw, ent)

    if any(k in q_nk for k in ("nha cung cap", "ncc", "nhap tu dau", "ben nao cung cap")) and any(
        k in q_nk for k in ("nhieu tien", "nhiu tien", "cao nhat", "lon nhat", "lon nhut", "nhiu tien nhut", "gia tri nhap cao", "tien hang nhat", "tien nhap nhat")
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "top_supplier_by_amount", "string", tw, ent)

    if ("gia nhap" in q_nk or ("nhap" in q_nk and "binh quan" in q_nk)) and (ent.product_key or ent.target_unit_norm):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "import_avg_unit_price", "number", tw, ent)

    if any(k in q_nk for k in ("phieu nhap", "tien nhap hang", "tien hang nhap", "nhap hang het")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "import_total_amount", "number", tw, ent)

    # ------------------------------------------------------------------
    # 8. Gross Profit, Expenses, Net Profit
    # ------------------------------------------------------------------
    if "lai rong" in q_nk or "loi nhuan rong" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "net_profit_total", "number", tw, ent)

    if "chi phi" in q_nk or "tien chi" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "expense_total", "number", tw, ent)

    if any(k in q_nk for k in ("lai gop", "loi gop", "loi nhuan gop", "tien lai", "tien loi", "loi nhieu nhat", "lai nhieu nhat", "loi nhat", "lai nhat", "lai cao nhat", "loi cao nhat")):
        if re.search(r"\b(?:mat hang|san pham|mon|sp|hang|thuoc|loai)\s+nao\b", q_nk) or "top" in q_nk:
            return ParsedQuestion(
                qid, biz_id, q_text, as_of_dt, "top_profit_product", "list" if top_k > 1 else "string", tw, ent, top_k=top_k
            )
        if any(k in q_nk for k in ("ty suat", "ti suat", "bao nhieu %", "bnhiu %", "bn %", "may %", "phan tram")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "gross_margin_pct", "percent", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "gross_profit_total", "number", tw, ent)

    # ------------------------------------------------------------------
    # 9. Special Rulebook §3 Customer / Staff / Product / Day / Hour Rankings
    # ------------------------------------------------------------------
    if "trung thanh nhat" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "loyal_customer", "string", tw, ent)

    if "khach sop nhat" in q_nk or "sop nhat" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "big_spender_customer", "string", tw, ent)

    if "doi y nhat" in q_nk or ("khach" in q_nk and "nhieu don huy nhat" in q_nk):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "fickle_customer", "string", tw, ent)

    if "ua chuong nhat" in q_nk or ("mat hang" in q_nk and "nhieu don nhat" in q_nk):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "popular_product", "string", tw, ent)

    if "cham chi nhat" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "hardworking_staff", "string", tw, ent)

    if any(k in q_nk for k in ("gio cao diem", "dong khach nhat vao may gio", "dong khach nhat dao may gio", "khung gio nao dong nhat")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "peak_hour", "number", tw, ent)

    if any(k in q_nk for k in ("ngay dat khach nhat", "ngay dong khach nhat")) or (
        any(k in q_nk for k in ("ngay nao", "bua nao", "hom nao", "ngay co"))
        and any(k in q_nk for k in ("dat khach nhat", "dong khach nhat", "nhieu don nhat", "nhieu don hang nhat", "so don cao nhat", "so don nhieu nhat"))
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "busiest_day_by_orders", "date", tw, ent)

    if any(k in q_nk for k in ("ngay nao", "bua nao", "hom nao", "ngay co")) and any(
        k in q_nk for k in (
            "doanh thu cao nhat", "doanh thu lon nhat", "doanh thu nhieu nhat", "ban dat nhat",
            "ban duoc nhieu tien nhat", "thu ve nhieu tien nhat", "doanh so cao nhat",
        )
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "busiest_day_by_revenue", "date", tw, ent)

    if ("ngay nao" in q_nk or "bua nao" in q_nk or "hom nao" in q_nk) and ("doanh thu" in q_nk or "dthu" in q_nk) and ("cao nhat" in q_nk or "lon nhat" in q_nk):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "busiest_day_by_revenue", "date", tw, ent)

    if any(k in q_nk for k in ("nhan vien nao", "ban nhan vien nao", "ai la nhan vien", "nguoi ban nao", "nguoi ban co", "nhan vien ban")) and any(
        k in q_nk for k in ("ban gioi nhat", "ban tot nhat", "doanh thu cao nhat", "doanh thu lon nhat", "nhieu doanh thu nhat", "ban duoc nhieu tien nhat")
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "top_staff_by_revenue", "string", tw, ent)

    if (
        top_k > 1
        or any(k in q_nk for k in ("khach nao", "khach hang nao", "ai la khach", "la ai", "ai mua", "nguoi mua nao", "khach mua", "top", "liet ke"))
    ) and ("khach" in q_nk or "ai mua" in q_nk or "nguoi mua" in q_nk):
        if any(k in q_nk for k in ("nhieu don", "nhiu don", "so don", "don mua", "nhieu lan", "nhiu lan")):
            return ParsedQuestion(
                qid, biz_id, q_text, as_of_dt, "top_customer_by_orders", "list" if top_k > 1 else "string", tw, ent, top_k=top_k
            )
        if any(
            k in q_nk
            for k in (
                "chi tieu", "mua nhieu tien", "mua hang nhieu tien", "tong tien mua", "tien mua cao nhat",
                "doanh thu", "mua nhieu nhat", "chi nhieu", "ung ho nhieu", "tong tien cao nhat", "tong tien lon nhat",
            )
        ):
            return ParsedQuestion(
                qid, biz_id, q_text, as_of_dt, "top_customer_by_spend", "list" if top_k > 1 else "string", tw, ent, top_k=top_k
            )

    if ent.address_query_norm and any(k in q_nk for k in ("bao nhieu khach", "bnhiu khach", "may khach", "so khach")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "customer_count_by_address", "number", tw, ent)

    # ------------------------------------------------------------------
    # 10. Unsold Products ('bị ế', 'không bán được cái nào')
    # ------------------------------------------------------------------
    if re.search(r"\b(?:bi e|mat hang e|hang e)\b", q_nk) or any(k in q_nk for k in ("khong ban ra dc", "khong ban duoc", "chua ban duoc")):
        if any(k in q_nk for k in ("bao nhieu", "bnhiu", "bn", "may mat hang", "co may", "dem giup", "dem so", "so mat hang")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "unsold_product_count", "number", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "unsold_product_list", "list", tw, ent)

    # ------------------------------------------------------------------
    # 11. Top Selling / Top Revenue Product or Phone Model
    # ------------------------------------------------------------------
    if "dong may nao" in q_nk or ("dong may" in q_nk and top_k > 1):
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "top_phone_model", "list" if top_k > 1 else "string", tw, ent, top_k=top_k
        )

    if any(
        k in q_nk
        for k in (
            "ban chay nhat", "ban chay nhut", "chay hang nhat", "hang nao chay nhat", "mon nao chay nhat",
            "sp nao chay nhat", "mat hang chay nhat", "hang chay nhat", "sp chay nhat", "mon chay nhat",
            "thuoc chay nhat", "mat hang hot nhat", "hang hot nhat", "mon hot nhat", "best seller",
            "ban duoc nhieu nhat", "ban duoc nhieu may nhat", "ban nhieu nhat", "sp ban duoc nhieu nhat",
            "ban cham nhat", "ban it nhat",
        )
    ):
        is_asc = ("ban cham nhat" in q_nk or "ban it nhat" in q_nk)
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "top_selling_product", "list" if top_k > 1 else "string", tw, ent, top_k=top_k, sort_asc=is_asc
        )

    if re.search(r"\b(?:mat hang|san pham|mon|sp|hang|thuoc|loai)\s+nao\b", q_nk) and any(
        k in q_nk for k in ("doanh thu cao nhat", "doanh thu lon nhat", "nhieu tien nhat", "mang ve nhieu doanh thu", "mang lai nhieu doanh thu", "thu ve nhieu tien nhat")
    ):
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "top_revenue_product", "list" if top_k > 1 else "string", tw, ent, top_k=top_k
        )

    # ------------------------------------------------------------------
    # 12. Cancelled / Returned Orders & Rates
    # ------------------------------------------------------------------
    if any(k in q_nk for k in ("bi huy", "don huy", "da huy", "huy don", "trang thai huy")):
        if any(k in q_nk for k in ("%", "phan tram", "ti le", "ty le")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "cancel_rate_pct", "percent", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "cancelled_order_count", "number", tw, ent)

    if any(k in q_nk for k in ("tra hang", "doi tra", "bi tra", "khach tra lai")) and "khach tra no" not in q_nk:
        if any(k in q_nk for k in ("%", "phan tram", "ti le", "ty le")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "return_rate_pct", "percent", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "returned_order_count", "number", tw, ent)

    # ------------------------------------------------------------------
    # 13. Share / Percentage ('tỉ trọng', 'chiếm bao nhiêu %', 'tỉ lệ kê đơn')
    # ------------------------------------------------------------------
    m_cust_sh = re.search(r"(?:mua|co|dat)\s+(?:tu\s+|it nhat\s+|tren\s+)?(\d+)\s*don", q_nk)
    if m_cust_sh and "khach" in q_nk and any(k in q_nk for k in ("%", "phan tram", "ti le", "ty le")):
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "customer_order_threshold_share_pct", "percent", tw, ent, threshold_value=float(m_cust_sh.group(1))
        )

    if tw.compare_window is None and (
        any(
            k in q_nk
            for k in (
                "ti trong", "ty trong", "chiem bao nhieu %", "chiem bnhiu %", "chiem bn %", "chiem may %",
                "chiem ti le", "chiem ty le", "ti le ke don", "ty le ke don", "ti le doanh thu", "ty le doanh thu",
                "bao nhieu % doanh thu", "bao nhieu phan tram doanh thu", "bnhiu % dthu", "bn % dthu", "may % tien ban",
                "bao nhieu % tong doanh thu", "bao nhieu phan tram tong doanh thu", "dong gop bao nhieu %",
                "dong gop bao nhieu phan tram", "dong gop bn %", "phan cua", "bang bao nhieu % tong",
                "bang bn % tong", "chiem bao nhieu phan tram",
            )
        )
        or (("trong tong doanh thu" in q_nk or "tren tong doanh thu" in q_nk) and ("%" in q_nk or "phan tram" in q_nk))
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "share_pct", "percent", tw, ent)

    # ------------------------------------------------------------------
    # 14. Period-over-Period Comparison ('tăng hay giảm bao nhiêu %', 'cao hay thấp hơn ... %')
    # ------------------------------------------------------------------
    if tw.compare_window is not None and any(
        k in q_nk for k in ("%", "phan tram", "tang hay giam", "thay doi", "chenh", "nhieu hay it hon", "cao hay thap hon", "hon hay kem", "tang truong")
    ):
        if any(k in q_nk for k in ("so don", "luong don", "don hang")) and not ent.product_key:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "order_count_change_pct", "percent", tw, ent)
        if (ent.product_key or tw.label.startswith("price_change_")) and not any(
            k in q_nk for k in ("doanh thu", "dthu", "doanh so", "tien ban", "so tien", "nhieu tien", "it tien")
        ):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "product_qty_change_pct", "percent", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "revenue_change_pct", "percent", tw, ent)

    # ------------------------------------------------------------------
    # 15. Product Selling Price on Specific Date
    # ------------------------------------------------------------------
    if ent.product_key and (
        any(
            k in q_nk
            for k in (
                "don gia", "gia ban mot", "gia ban 1", "gia ban bao nhieu", "gia bao nhieu tien mot",
                "gia bao nhieu mot", "gia 1 ", "gia moi ", "ban gia bao nhieu",
            )
        )
        or (tw.label == "single_date" and "gia" in q_nk and "doanh thu" not in q_nk and "tong" not in q_nk)
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "product_unit_price_on_date", "number", tw, ent)

    # ------------------------------------------------------------------
    # 16. Average Order Value / Average Phone Price / Customer Payable
    # ------------------------------------------------------------------
    if "khach phai tra" in q_nk or "khach phai thanh toan" in q_nk:
        if any(k in q_nk for k in ("trung binh", "binh quan", "moi don")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "customer_payable_avg", "number", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "customer_payable_total", "number", tw, ent)

    if cb.domain == "phone_shop" and any(k in q_nk for k in ("trung binh 1 may", "trung binh moi may", "binh quan 1 may", "binh quan moi may")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "phone_avg_price", "number", tw, ent)

    if any(
        k in q_nk
        for k in (
            "trung binh moi don", "trung binh tren mot don", "gia tri trung binh mot don",
            "binh quan mot don", "binh quan moi don", "binh quan 1 don", "trung binh 1 don",
        )
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "revenue_per_order_avg", "number", tw, ent)

    if any(k in q_nk for k in ("trung binh moi ngay", "binh quan moi ngay")):
        if any(k in q_nk for k in ("so don", "bao nhieu don", "may don")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "order_count_daily_avg", "number", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "revenue_daily_avg", "number", tw, ent)

    # ------------------------------------------------------------------
    # 17. Threshold Filters (Order Revenue, Daily Revenue, Product Order Count, Customer Order Count)
    # ------------------------------------------------------------------
    m_prod_ord_thr = re.search(r"(?:xuat hien|co mat|ban ra).*?(?:tu\s+|it nhat\s+|tren\s+)?(\d+)\s*don", q_nk)
    if m_prod_ord_thr and any(k in q_nk for k in ("san pham", "mat hang", "mon", "sp", "thuoc")):
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "product_count_by_order_threshold", "number", tw, ent, threshold_value=float(m_prod_ord_thr.group(1))
        )

    m_cust_ord_thr = re.search(r"(?:mua|co|dat)\s+(?:da\s+mua\s+)?(?:tu\s+|it nhat\s+|tren\s+)?(\d+)\s*don", q_nk)
    if m_cust_ord_thr and any(k in q_nk for k in ("bao nhieu khach", "nhieu khach", "bnhiu khach", "bn khach", "may khach", "so khach", "dem so khach")):
        op = "gt" if ("tren " in q_nk and "tro len" not in q_nk and "≥" not in q_text and ">=" not in q_text) else "gte"
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "customer_count_by_order_threshold", "number", tw, ent, threshold_value=float(m_cust_ord_thr.group(1)), threshold_op=op
        )

    m_thr = re.search(
        r"(?:co\s+)?(?:tien\s+hang|gia\s+tri|doanh\s+thu|tong\s+tien|don\s+hang|don)\s+(?:dat\s+|tu\s+|tren\s+|co\s+doanh\s+thu\s+(?:tu\s+)?|≥\s*|>=?\s*)?([\d.,]+\s*(?:dong|trieu|nghin|ngan|tr\d*|k|d|đ)?)\s*(?:tro\s+len|hon)?",
        q_noacc,
    )
    if m_thr:
        val_t = parse_vn_number(m_thr.group(1))
        if val_t is not None and val_t >= 1000:
            op = "gt" if ("tren " in q_noacc and "tro len" not in q_noacc and "≥" not in q_text and ">=" not in q_text) else "gte"
            if any(k in q_nk for k in ("bao nhieu ngay", "bnhiu ngay", "bn ngay", "may ngay", "so ngay")):
                return ParsedQuestion(
                    qid, biz_id, q_text, as_of_dt, "day_count_by_revenue_threshold", "number", tw, ent, threshold_value=float(val_t), threshold_op=op
                )
            if any(k in q_nk for k in ("bao nhieu don", "bnhiu don", "bn don", "may don", "so don")):
                return ParsedQuestion(
                    qid, biz_id, q_text, as_of_dt, "order_count", "number", tw, ent, threshold_value=float(val_t), threshold_op=op
                )

    # ------------------------------------------------------------------
    # 18. Product / Phone Quantity Sold vs Revenue vs Order Count
    # ------------------------------------------------------------------
    is_money_question = any(
        k in q_nk
        for k in (
            "doanh thu", "dthu", "doanh so", "bao nhieu tien", "bnhiu tien", "bn tien",
            "nhieu tien", "nhiu tien", "thu ve bao nhieu", "ban duoc bao nhieu tien", "tien ban",
        )
    )
    is_order_count_question = any(
        k in q_nk
        for k in (
            "bao nhieu don", "bnhiu don", "bn don", "may don", "so don", "so dh", "tat ca bao nhieu don",
            "ban dc may don", "duoc bnhiu don", "so bill", "bao nhieu bill", "may bill", "so hoa don", "bao nhieu hoa don",
        )
    )

    if is_order_count_question and not is_money_question:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "order_count", "number", tw, ent)

    # Multi-period comparison, monotonic check, or list ("tháng nào doanh thu cao nhất" vs "lần lượt mỗi tháng bao nhiêu")
    if tw.multi_windows:
        if any(k in q_nk for k in ("tang lien tuc", "luon cao hon", "luon lon hon", "thang sau cao hon", "thang sau lon hon", "quy sau cao hon", "xu huong doanh thu", "co phai thang")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "is_revenue_monotonic_increasing", "boolean", tw, ent)
        if "thang nao" in q_nk or "quy nao" in q_nk:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "multi_period_best", "date" if "thang" in q_nk else "string", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "multi_period_revenue_list", "list", tw, ent)

    # Two-filter revenue pair list ("doanh thu bán qua X và bán qua Y mỗi bên / lần lượt là bao nhiêu")
    if is_money_question and (
        any(k in q_nk for k in ("lan luot", "moi ca", "moi ben", "tung ca", "moi kenh", "moi hinh thuc"))
        or q_nk.count("doanh thu cac don") >= 2
        or (" va " in q_nk and any(k in q_nk for k in ("moi loai", "tung loai")))
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "revenue_pair_list", "list", tw, ent)

    if not is_money_question and (
        ent.product_key
        or ent.phone_model_norm
        or ent.size_norm
        or ent.cafe_size_filter
        or ent.target_unit_norm
        or any(k in q_nk for k in ("bao nhieu chiec", "bnhiu chiec", "may chiec", "bao nhieu may", "bnhiu may", "may may", "so ly", "bao nhieu ly", "may ly"))
    ):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "product_qty_sold", "number", tw, ent)

    return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "revenue_total", "number", tw, ent)


def _parse_auto_repair_question(
    qid: str,
    biz_id: str,
    q_text: str,
    q_nk: str,
    as_of_dt: dt.date,
    tw: TimeWindow,
    ent: LinkedEntities,
    top_k: int,
) -> ParsedQuestion:
    if "cham chi nhat" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "hardworking_staff", "string", tw, ent)

    if "hoa hong" in q_nk:
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_mechanic_commission", "number", tw, ent)

    if "bao hanh" in q_nk:
        if any(k in q_nk for k in ("%", "phan tram", "ti le", "ty le")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_warranty_rate_pct", "percent", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_warranty_count", "number", tw, ent)

    if "khach quen" in q_nk or "toi nhieu lan" in q_nk or "sua nhieu lan" in q_nk:
        if "xe nao" in q_nk or "bien so nao" in q_nk:
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_top_repeat_vehicle", "string", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_repeat_vehicle_count", "number", tw, ent)

    if "tho nao" in q_nk or "ai la tho" in q_nk or "nhan nhieu phieu" in q_nk:
        if any(k in q_nk for k in ("doanh thu", "dthu", "nhieu tien")):
            return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_top_mechanic_by_revenue", "string", tw, ent)
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_top_mechanic_by_tickets", "string", tw, ent)

    if "dong xe nao" in q_nk or "loai xe nao" in q_nk:
        return ParsedQuestion(
            qid, biz_id, q_text, as_of_dt, "repair_top_vehicle_model", "list" if top_k > 1 else "string", tw, ent, top_k=top_k
        )

    if any(k in q_nk for k in ("trung binh mot phieu", "trung binh 1 phieu", "binh quan mot phieu")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_avg_paid_ticket", "number", tw, ent)

    if any(k in q_nk for k in ("bao nhieu chiec xe", "bnhiu chiec xe", "may chiec xe", "so chiec xe")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_distinct_vehicles", "number", tw, ent)

    if tw.compare_window is not None and any(k in q_nk for k in ("%", "phan tram", "tang hay giam", "thay doi")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_revenue_change_pct", "percent", tw, ent)

    if any(k in q_nk for k in ("doanh thu", "dthu", "tien sua", "thu duoc bao nhieu tien", "ban duoc bao nhieu tien")):
        return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_revenue_total", "number", tw, ent)

    return ParsedQuestion(qid, biz_id, q_text, as_of_dt, "repair_ticket_count", "number", tw, ent)
