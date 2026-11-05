"""Deterministic Rulebook-Compliant Query Executor for DataMate.

Executes `ParsedQuestion` plans against `CanonicalBusiness` DataFrames,
strictly enforcing all rules in `train/rulebook.md` and `public_test/rulebook.md`.
"""

from __future__ import annotations

import calendar as py_calendar
import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.canonical import CanonicalBusiness, clean_str, norm_key, strip_accents
from src.entity_linker import LinkedEntities
from src.ir import AnswerType
from src.parser import ParsedQuestion
from src.rule_engine import round_2dec, round_money_vnd
from src.temporal import LunarSolarCalendar, TimeWindow


@dataclass
class ExecutionResult:
    answer_type: AnswerType
    value: Any = None
    reason: Optional[str] = None
    confidence: float = 1.0


def _unanswerable(reason: str) -> ExecutionResult:
    return ExecutionResult(answer_type="unanswerable", value=None, reason=reason, confidence=1.0)


def _abstain(reason: str = "Low confidence") -> ExecutionResult:
    return ExecutionResult(answer_type="abstain", value=None, reason=reason, confidence=0.0)


def _format_qty(val: float) -> int | float:
    r = round_2dec(val)
    if abs(r - round(r)) < 1e-9:
        return int(round(r))
    return r


def _is_period_outside_data(cb: CanonicalBusiness, tw: TimeWindow) -> bool:
    if cb.data_start_date is None or cb.data_end_date is None:
        return False
    if tw.end_date < cb.data_start_date or tw.start_date > cb.data_end_date:
        return True
    if tw.compare_window is not None:
        cw = tw.compare_window
        if cw.end_date < cb.data_start_date or cw.start_date > cb.data_end_date:
            return True
    return False


def _filter_orders_in_window(
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_date: dt.date,
    ent: Optional[LinkedEntities] = None,
    valid_only: bool = True,
) -> pd.DataFrame:
    df = cb.orders
    if df.empty:
        return df
    end_d = min(tw.end_date, as_of_date)
    mask = (df["date"] >= tw.start_date) & (df["date"] <= end_d)
    if valid_only:
        mask &= df["status"].isin(["completed", "delivering"])
    if tw.shift_filter == "morning":
        mask &= df["hour"].notna() & (df["hour"] >= 6) & (df["hour"] <= 13)
    elif tw.shift_filter == "afternoon":
        mask &= df["hour"].notna() & (df["hour"] >= 14) & (df["hour"] <= 21)
    if tw.specific_weekday is not None:
        mask &= df["date"].apply(lambda d: d.isoweekday() == tw.specific_weekday if isinstance(d, dt.date) else False)

    if ent is not None:
        if ent.customer_key:
            mask &= df["customer_key"] == ent.customer_key
        if ent.staff_id:
            mask &= df["staff_id"].apply(norm_key) == norm_key(ent.staff_id)
        if ent.channel_norm:
            mask &= df["channel"] == ent.channel_norm
        if ent.payment_method_norm:
            mask &= df["payment_method"] == ent.payment_method_norm
        if ent.room_area_norm:
            mask &= df["room_area"].apply(norm_key).str.contains(ent.room_area_norm, na=False)
        if ent.customer_group_norm:
            if "customer_group_norm" in df.columns and (df["customer_group_norm"] != "").any():
                mask &= df["customer_group_norm"] == ent.customer_group_norm
            elif not cb.customers.empty:
                grp_cks = set(cb.customers[cb.customers["customer_group_norm"] == ent.customer_group_norm]["customer_key"])
                mask &= df["customer_key"].isin(grp_cks)
        if ent.customer_gender:
            if "customer_gender" in df.columns and (df["customer_gender"] != "").any():
                mask &= df["customer_gender"] == ent.customer_gender
            elif not cb.customers.empty:
                gen_cks = set(cb.customers[cb.customers["gender"] == ent.customer_gender]["customer_key"])
                mask &= df["customer_key"].isin(gen_cks)
        if ent.address_query_norm:
            if "customer_address_norm" in df.columns and (df["customer_address_norm"] != "").any():
                mask &= df["customer_address_norm"].str.contains(re.escape(ent.address_query_norm), na=False)
            elif not cb.customers.empty:
                addr_cks = set(
                    cb.customers[cb.customers["address_norm"].str.contains(re.escape(ent.address_query_norm), na=False)]["customer_key"]
                )
                mask &= df["customer_key"].isin(addr_cks)
    return df[mask]


def _filter_lines_in_window(
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_date: dt.date,
    ent: Optional[LinkedEntities] = None,
    valid_only: bool = True,
    apply_product_filter: bool = True,
) -> pd.DataFrame:
    if cb.order_lines.empty:
        return cb.order_lines
    df_ord = _filter_orders_in_window(cb, tw, as_of_date, ent=ent, valid_only=valid_only)
    if df_ord.empty:
        return cb.order_lines.iloc[0:0]
    valid_oids = set(df_ord["order_id"])
    lines = cb.order_lines[cb.order_lines["order_id"].isin(valid_oids)]
    if lines.empty or not apply_product_filter or ent is None:
        return lines

    mask = pd.Series(True, index=lines.index)
    if ent.product_key:
        mask &= lines["product_key"] == ent.product_key
    elif ent.cafe_base_keys:
        mask &= lines["product_key"].isin(ent.cafe_base_keys)
    elif ent.phone_model_norm and not cb.products.empty:
        pks = set(cb.products[cb.products["phone_model_norm"] == ent.phone_model_norm]["product_key"])
        mask &= lines["product_key"].isin(pks)
    elif ent.active_ingredient_norm and not cb.products.empty:
        pks = set(cb.products[cb.products["active_ingredient_norm"] == ent.active_ingredient_norm]["product_key"])
        mask &= lines["product_key"].isin(pks)
    elif ent.size_norm and not cb.products.empty:
        pks = set(cb.products[cb.products["size_norm"] == ent.size_norm]["product_key"])
        if ent.category_norm:
            cat_pks = set(cb.products[cb.products["category_norm"] == ent.category_norm]["product_key"])
            pks &= cat_pks
        mask &= lines["product_key"].isin(pks)
    elif ent.brand_norm and not cb.products.empty:
        pks = set(cb.products[cb.products["brand_norm"] == ent.brand_norm]["product_key"])
        if ent.category_norm:
            cat_pks = set(cb.products[cb.products["category_norm"] == ent.category_norm]["product_key"])
            pks &= cat_pks
        mask &= lines["product_key"].isin(pks)
    elif ent.category_norm:
        mask &= lines["category_norm"] == ent.category_norm

    if ent.cafe_size_filter == "m" and not ent.product_key:
        mask &= lines["product_name"].str.lower().str.strip().str.endswith("(m)", na=False)
    elif ent.cafe_size_filter == "l" and not ent.product_key:
        mask &= lines["product_name"].str.lower().str.strip().str.endswith("(l)", na=False)

    return lines[mask]


def _has_product_level_filter(ent: LinkedEntities) -> bool:
    return bool(
        ent.product_key
        or ent.cafe_base_keys
        or ent.phone_model_norm
        or ent.active_ingredient_norm
        or ent.size_norm
        or ent.brand_norm
        or ent.category_norm
        or ent.cafe_size_filter
    )


def _compute_revenue(cb: CanonicalBusiness, tw: TimeWindow, as_of_date: dt.date, ent: LinkedEntities, q_nk: str) -> float:
    # Check if 'không tính phụ kiện' / 'bỏ phụ kiện' in phone_shop
    exclude_acc = cb.domain == "phone_shop" and any(k in q_nk for k in ("khong tinh phu kien", "khong ke phu kien", "bo phu kien"))
    # Check pharmacy prescription filter
    is_rx_only = cb.domain == "pharmacy" and any(k in q_nk for k in ("thuoc ke don", "ke don")) and not any(k in q_nk for k in ("khong ke don", "ngoai tru thuoc ke don", "tru thuoc ke don"))
    is_non_rx_only = cb.domain == "pharmacy" and any(k in q_nk for k in ("khong ke don", "ngoai tru thuoc ke don", "tru thuoc ke don"))

    if _has_product_level_filter(ent) or exclude_acc or is_rx_only or is_non_rx_only:
        lines = _filter_lines_in_window(cb, tw, as_of_date, ent=ent, valid_only=True, apply_product_filter=True)
        if exclude_acc and not lines.empty:
            lines = lines[lines["category_norm"].isin(["may moi", "may cu"])]
        if is_rx_only and not lines.empty and not cb.products.empty:
            rx_pks = set(cb.products[cb.products["is_prescription"] == True]["product_key"])  # noqa: E712
            lines = lines[lines["product_key"].isin(rx_pks)]
        elif is_non_rx_only and not lines.empty and not cb.products.empty:
            rx_pks = set(cb.products[cb.products["is_prescription"] == True]["product_key"])  # noqa: E712
            lines = lines[~lines["product_key"].isin(rx_pks)]
        return float(lines["line_amount"].sum()) if not lines.empty else 0.0
    df_ord = _filter_orders_in_window(cb, tw, as_of_date, ent=ent, valid_only=True)
    return float(df_ord["revenue"].sum()) if not df_ord.empty else 0.0


def execute_question(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    cal: LunarSolarCalendar,
) -> ExecutionResult:
    intent = pq.intent
    tw = pq.time_window
    ent = pq.entities
    as_of_dt = pq.as_of_date
    q_nk = norm_key(pq.question_text)

    # 1. Explicit unanswerable intents from parser
    if intent.startswith("unanswerable_"):
        return _unanswerable(pq.unsupported_reason or "Không đủ dữ liệu để trả lời.")

    # 2. Check temporal shift capability
    if (tw.shift_filter is not None or (tw.compare_window and tw.compare_window.shift_filter is not None)) and not cb.capabilities.get("has_hour_of_day"):
        return _unanswerable("Dữ liệu của cửa hàng không có cột giờ tạo đơn để phân biệt ca sáng/chiều.")

    # 3. Check period coverage against [data_start_date, data_end_date]
    # (Skip for future batch expiry queries and due date queries)
    if intent not in (
        "expiring_stock_cost", "expiring_batch_count", "expiring_product_count", "expiring_product_list", "nearest_expiry_date",
        "due_date_after_today_count", "due_date_overdue_count",
        "customer_count_by_address", "stock_ending_cost_value", "stock_ending_qty", "out_of_stock_product_count",
    ):
        if _is_period_outside_data(cb, tw):
            return _unanswerable("Kỳ được hỏi nằm hoàn toàn ngoài khoảng thời gian có dữ liệu của cửa hàng.")

    # 4. Check filter capabilities
    if ent.address_query_norm and not cb.capabilities.get("has_customer_address"):
        return _unanswerable("Dữ liệu của cửa hàng không ghi địa chỉ khách hàng.")
    if ent.channel_norm and not cb.capabilities.get("has_channel"):
        return _unanswerable("Dữ liệu không có cột kênh bán hàng.")
    if ent.room_area_norm and not cb.capabilities.get("has_room_area"):
        return _unanswerable("Dữ liệu không có cột khu vực/phòng bàn.")
    if ent.customer_group_norm and not cb.capabilities.get("has_customer_group"):
        return _unanswerable("Dữ liệu không có thông tin phân nhóm khách hàng.")
    if ent.customer_gender and not cb.capabilities.get("has_customer_gender"):
        return _unanswerable("Dữ liệu không có thông tin giới tính khách hàng.")
    if _has_product_level_filter(ent) and not cb.capabilities.get("has_line_items") and cb.domain != "auto_repair":
        return _unanswerable("Dữ liệu chỉ có danh sách hoá đơn tổng hợp, không có chi tiết mặt hàng.")

    # ------------------------------------------------------------------
    # Auto Repair Domain (dev_012)
    # ------------------------------------------------------------------
    if cb.domain == "auto_repair":
        return _execute_auto_repair(pq, cb, cal)

    # ------------------------------------------------------------------
    # Revenue, Order Count, Averages, Comparisons
    # ------------------------------------------------------------------
    if intent == "revenue_total":
        rev = _compute_revenue(cb, tw, as_of_dt, ent, q_nk)
        if tw.specific_weekday is not None and any(k in q_nk for k in ("trung binh", "binh quan", "moi ngay thu", "moi chu nhat")):
            n_wd = cal.count_weekday_in_range(
                max(tw.start_date, cb.data_start_date or tw.start_date),
                min(tw.end_date, as_of_dt, cb.data_end_date or tw.end_date),
                tw.specific_weekday,
            )
            if n_wd == 0:
                return _unanswerable("Không có ngày thứ được hỏi trong kỳ.")
            return ExecutionResult("number", round_money_vnd(rev / n_wd))
        return ExecutionResult("number", round_money_vnd(rev))

    if intent == "revenue_daily_avg":
        rev = _compute_revenue(cb, tw, as_of_dt, ent, q_nk)
        s_d = max(tw.start_date, cb.data_start_date or tw.start_date)
        e_d = min(tw.end_date, as_of_dt, cb.data_end_date or tw.end_date)
        if tw.specific_weekday is not None:
            n_days = cal.count_weekday_in_range(s_d, e_d, tw.specific_weekday)
        else:
            n_days = max(1, (e_d - s_d).days + 1)
        return ExecutionResult("number", round_money_vnd(rev / n_days))

    if intent == "revenue_per_order_avg":
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Sổ bán hàng không ghi theo mã đơn nên không tính được giá trị trung bình một đơn.")
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        if df_ord.empty:
            return ExecutionResult("number", 0)
        return ExecutionResult("number", round_money_vnd(float(df_ord["revenue"].sum()) / len(df_ord)))

    if intent in ("customer_payable_total", "customer_payable_avg"):
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        if df_ord.empty:
            return ExecutionResult("number", 0)
        tot_pay = float(df_ord["total_payable"].sum())
        if intent == "customer_payable_avg":
            if not cb.capabilities.get("has_order_id"):
                return _unanswerable("Không có mã đơn hàng.")
            return ExecutionResult("number", round_money_vnd(tot_pay / len(df_ord)))
        return ExecutionResult("number", round_money_vnd(tot_pay))

    if intent == "phone_avg_price":
        lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True, apply_product_filter=True)
        if not ent.category_norm:
            lines = lines[lines["category_norm"].isin(["may moi", "may cu"])]
        if lines.empty or float(lines["base_qty"].sum()) == 0:
            return ExecutionResult("number", 0)
        return ExecutionResult("number", round_money_vnd(float(lines["line_amount"].sum()) / float(lines["base_qty"].sum())))

    if intent in ("order_count", "order_count_daily_avg"):
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Sổ bán hàng chỉ ghi từng dòng hàng, không có mã đơn hàng nên không đếm được số đơn.")
        if _has_product_level_filter(ent):
            lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True, apply_product_filter=True)
            cnt = int(lines["order_id"].nunique()) if not lines.empty else 0
        else:
            df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
            if pq.threshold_value is not None:
                if pq.threshold_op == "gt":
                    df_ord = df_ord[df_ord["revenue"] > pq.threshold_value]
                else:
                    df_ord = df_ord[df_ord["revenue"] >= pq.threshold_value]
            cnt = len(df_ord)

        if intent == "order_count_daily_avg" or (
            tw.specific_weekday is not None and any(k in q_nk for k in ("trung binh", "binh quan", "moi ngay"))
        ):
            s_d = max(tw.start_date, cb.data_start_date or tw.start_date)
            e_d = min(tw.end_date, as_of_dt, cb.data_end_date or tw.end_date)
            if tw.specific_weekday is not None:
                n_days = cal.count_weekday_in_range(s_d, e_d, tw.specific_weekday)
            else:
                n_days = max(1, (e_d - s_d).days + 1)
            return ExecutionResult("number", round_2dec(cnt / n_days))
        return ExecutionResult("number", int(cnt))

    if intent in ("cancelled_order_count", "cancel_rate_pct", "returned_order_count", "return_rate_pct"):
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Không có mã đơn hàng.")
        df_all = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=False)
        if intent == "cancelled_order_count":
            return ExecutionResult("number", int((df_all["status"] == "cancelled").sum()))
        if intent == "returned_order_count":
            return ExecutionResult("number", int((df_all["status"] == "returned").sum()))
        if df_all.empty:
            return ExecutionResult("percent", 0.0)
        target_st = "cancelled" if intent == "cancel_rate_pct" else "returned"
        n_target = int((df_all["status"] == target_st).sum())
        return ExecutionResult("percent", round_2dec(n_target / len(df_all) * 100.0))

    if intent == "revenue_change_pct" and tw.compare_window is not None:
        cur_rev = _compute_revenue(cb, tw, as_of_dt, ent, q_nk)
        prev_rev = _compute_revenue(cb, tw.compare_window, as_of_dt, ent, q_nk)
        if prev_rev == 0:
            return _unanswerable("Doanh thu kỳ gốc bằng 0 nên không tính được phần trăm thay đổi.")
        return ExecutionResult("percent", round_2dec((cur_rev - prev_rev) / prev_rev * 100.0))

    if intent == "order_count_change_pct" and tw.compare_window is not None:
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Không có mã đơn hàng.")
        c_ord = len(_filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True))
        p_ord = len(_filter_orders_in_window(cb, tw.compare_window, as_of_dt, ent=ent, valid_only=True))
        if p_ord == 0:
            return _unanswerable("Số đơn kỳ gốc bằng 0.")
        return ExecutionResult("percent", round_2dec((c_ord - p_ord) / p_ord * 100.0))

    if intent == "product_qty_change_pct" and tw.compare_window is not None:
        w_cur = tw
        w_prev = tw.compare_window
        if tw.label.startswith("price_change_lookup_"):
            if cb.price_history.empty or not ent.product_key:
                return _unanswerable("Không có lịch sử điều chỉnh giá bán.")
            n_days = int(tw.label.split("_")[-1])
            ph = cb.price_history[
                (cb.price_history["product_key"] == ent.product_key)
                & (cb.price_history["effective_date"] <= as_of_dt)
            ].sort_values("effective_date")
            if ph.empty:
                return _unanswerable("Không tìm thấy ngày điều chỉnh giá của sản phẩm.")
            d0: dt.date = ph.iloc[-1]["effective_date"]
            w_cur = TimeWindow(d0, d0 + dt.timedelta(days=n_days - 1), label="price_change_post")
            w_prev = TimeWindow(d0 - dt.timedelta(days=n_days), d0 - dt.timedelta(days=1), label="price_change_pre")

        c_lines = _filter_lines_in_window(cb, w_cur, as_of_dt, ent=ent, valid_only=True, apply_product_filter=True)
        p_lines = _filter_lines_in_window(cb, w_prev, as_of_dt, ent=ent, valid_only=True, apply_product_filter=True)
        c_qty = float(c_lines["base_qty"].sum()) if not c_lines.empty else 0.0
        p_qty = float(p_lines["base_qty"].sum()) if not p_lines.empty else 0.0
        if p_qty == 0:
            return _unanswerable("Sản lượng kỳ gốc bằng 0.")
        return ExecutionResult("percent", round_2dec((c_qty - p_qty) / p_qty * 100.0))

    if intent in ("multi_period_revenue_list", "multi_period_best", "is_revenue_monotonic_increasing"):
        vals = []
        labels = []
        for mw in tw.multi_windows:
            r_v = round_money_vnd(_compute_revenue(cb, mw, as_of_dt, ent, q_nk))
            vals.append(r_v)
            labels.append(mw.label)
        if intent == "multi_period_revenue_list":
            return ExecutionResult("list", vals)
        if intent == "is_revenue_monotonic_increasing":
            is_mono = bool(len(vals) >= 2 and all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)))
            return ExecutionResult("boolean", is_mono)
        best_idx = int(np.argmax(vals))
        return ExecutionResult(pq.expected_type, labels[best_idx])

    if intent == "revenue_pair_list":
        # Check shifts first ("ca sáng và ca chiều")
        if "ca sang" in q_nk and "ca chieu" in q_nk:
            if not cb.capabilities.get("has_hour_of_day"):
                return _unanswerable("Dữ liệu không có giờ tạo đơn.")
            pos_s = q_nk.find("ca sang")
            pos_c = q_nk.find("ca chieu")
            shifts = ["morning", "afternoon"] if pos_s < pos_c else ["afternoon", "morning"]
            out_s = []
            for sh in shifts:
                tw_sh = TimeWindow(tw.start_date, tw.end_date, label=sh, shift_filter=sh)
                out_s.append(round_money_vnd(_compute_revenue(cb, tw_sh, as_of_dt, None, q_nk)))
            return ExecutionResult("list", out_s)

        # Check channels in order of appearance
        ch_pats = [
            (r"\b(?:tai quay|tai tiem|tai quan|ban truc tiep)\b", "in_store"),
            (r"\bzalo\b", "zalo"),
            (r"\b(?:qua dien thoai|goi dien)\b", "phone"),
            (r"\b(?:app giao do an|grabfood|shopeefood|befood|qua app|tren app)\b", "delivery_app"),
            (r"\b(?:san tmdt|thuong mai dien tu|shopee|lazada|tiktok)\b", "ecommerce"),
            (r"\b(?:facebook|fanpage|fb|website|web)\b", "online"),
            (r"\b(?:tai ban|an tai cho)\b", "tai ban"),
            (r"\bmang ve\b", "mang ve"),
        ]
        found_ch = []
        for pat, ch_val in ch_pats:
            m_c = re.search(pat, q_nk)
            if m_c:
                found_ch.append((m_c.start(), ch_val))
        found_ch.sort(key=lambda x: x[0])
        if len(found_ch) >= 2:
            if not cb.capabilities.get("has_channel"):
                return _unanswerable("Dữ liệu không có cột kênh bán hàng.")
            out_c = []
            for _, ch_v in found_ch[:2]:
                e_sub = LinkedEntities(
                    category_norm=ent.category_norm,
                    brand_norm=ent.brand_norm,
                    product_key=ent.product_key,
                    channel_norm=ch_v,
                )
                out_c.append(round_money_vnd(_compute_revenue(cb, tw, as_of_dt, e_sub, q_nk)))
            return ExecutionResult("list", out_c)

        # Check payment methods in order of appearance
        pm_pats = [
            (r"\b(?:tien mat|tm)\b", "cash"),
            (r"\b(?:chuyen khoan|ck)\b", "transfer"),
            (r"\b(?:vi dien tu|momo|zalopay|vnpay)\b", "ewallet"),
            (r"\b(?:thu ho|cod|ship cod)\b", "cod"),
            (r"\bquet the\b", "card"),
        ]
        found_pm = []
        for pat, pm_val in pm_pats:
            m_p = re.search(pat, q_nk)
            if m_p:
                found_pm.append((m_p.start(), pm_val))
        found_pm.sort(key=lambda x: x[0])
        if len(found_pm) >= 2:
            if not cb.capabilities.get("has_payment_method"):
                return _unanswerable("Dữ liệu không có cột phương thức thanh toán.")
            out_p = []
            for _, pm_v in found_pm[:2]:
                e_sub = LinkedEntities(
                    category_norm=ent.category_norm,
                    brand_norm=ent.brand_norm,
                    product_key=ent.product_key,
                    payment_method_norm=pm_v,
                )
                out_p.append(round_money_vnd(_compute_revenue(cb, tw, as_of_dt, e_sub, q_nk)))
            return ExecutionResult("list", out_p)

        return ExecutionResult("number", round_money_vnd(_compute_revenue(cb, tw, as_of_dt, ent, q_nk)))

    if intent == "day_count_by_revenue_threshold":
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        if df_ord.empty or pq.threshold_value is None:
            return ExecutionResult("number", 0)
        grp = df_ord.groupby("date", as_index=False)["revenue"].sum()
        if pq.threshold_op == "gt":
            cnt = int((grp["revenue"] > pq.threshold_value).sum())
        else:
            cnt = int((grp["revenue"] >= pq.threshold_value).sum())
        return ExecutionResult("number", cnt)

    if intent == "product_count_by_order_threshold":
        if not cb.capabilities.get("has_line_items") or not cb.capabilities.get("has_order_id"):
            return _unanswerable("Dữ liệu không có chi tiết mặt hàng hoặc mã đơn hàng.")
        lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=None, valid_only=True, apply_product_filter=False)
        if not lines.empty and "is_service_fee" in lines.columns:
            lines = lines[lines["is_service_fee"] == False]  # noqa: E712
        if lines.empty or pq.threshold_value is None:
            return ExecutionResult("number", 0)
        grp = lines.groupby("product_key", as_index=False)["order_id"].nunique()
        cnt = int((grp["order_id"] >= pq.threshold_value).sum())
        return ExecutionResult("number", cnt)

    if intent in ("customer_count_by_order_threshold", "customer_order_threshold_share_pct"):
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Dữ liệu không có mã đơn hàng.")
        has_cust = bool(not cb.orders.empty and (cb.orders["customer_key"] != "").any())
        if not has_cust:
            return _unanswerable("Cửa hàng không có dữ liệu thông tin khách hàng.")
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        df_c = df_ord[(df_ord["is_walkin"] == False) & (df_ord["customer_key"] != "")].copy()  # noqa: E712
        if df_c.empty:
            return ExecutionResult("percent" if intent.endswith("_pct") else "number", 0.0 if intent.endswith("_pct") else 0)
        grp = df_c.groupby("customer_key", as_index=False)["order_id"].nunique()
        thr = pq.threshold_value or 2.0
        if pq.threshold_op == "gt":
            n_qual = int((grp["order_id"] > thr).sum())
        else:
            n_qual = int((grp["order_id"] >= thr).sum())
        if intent == "customer_order_threshold_share_pct":
            return ExecutionResult("percent", round_2dec(n_qual / len(grp) * 100.0))
        return ExecutionResult("number", n_qual)

    # ------------------------------------------------------------------
    # Product Quantity Sold & Price on Date & Share %
    # ------------------------------------------------------------------
    if intent == "product_qty_sold":
        if not cb.capabilities.get("has_line_items"):
            return _unanswerable("Không có dữ liệu chi tiết mặt hàng.")
        exclude_acc = cb.domain == "phone_shop" and any(
            k in q_nk for k in ("khong tinh phu kien", "khong ke phu kien", "bo phu kien", "chiec dien thoai", "may moi lan may cu", "may moi voi may cu")
        )
        lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True, apply_product_filter=True)
        if exclude_acc and not _has_product_level_filter(ent):
            lines = lines[lines["category_norm"].isin(["may moi", "may cu"])]
        if cb.domain == "cafe" and not ent.product_key and (ent.target_unit_norm == "ly" or any(k in q_nk for k in ("so ly", "bao nhieu ly", "may ly", "ban duoc bao nhieu ly"))):
            lines = lines[~lines["category_norm"].isin(["banh", "topping", "snack", "do an"])]
        if lines.empty:
            return ExecutionResult("number", 0)
        tot_base = float(lines["base_qty"].sum())
        if ent.target_unit_norm and ent.product_key:
            fac = cb.unit_factors.get((ent.product_key, ent.target_unit_norm), 1.0)
            return ExecutionResult("number", _format_qty(tot_base / fac if fac > 0 else tot_base))
        return ExecutionResult("number", _format_qty(tot_base))

    if intent == "product_unit_price_on_date":
        if not ent.product_key:
            return _unanswerable("Không xác định được mặt hàng.")
        target_dt = tw.end_date
        # Check price_history first if available
        if not cb.price_history.empty:
            ph = cb.price_history[cb.price_history["product_key"] == ent.product_key].sort_values("effective_date")
            if not ph.empty:
                before = ph[ph["effective_date"] <= target_dt]
                if not before.empty:
                    prc = float(before.iloc[-1]["new_price"])
                    return ExecutionResult("number", round_money_vnd(prc))
                after = ph[ph["effective_date"] > target_dt]
                if not after.empty and pd.notna(after.iloc[0]["old_price"]):
                    prc = float(after.iloc[0]["old_price"])
                    return ExecutionResult("number", round_money_vnd(prc))
        # Check actual sales lines on target_dt (or most recent on/before target_dt)
        if not cb.order_lines.empty:
            p_lines = cb.order_lines[
                (cb.order_lines["product_key"] == ent.product_key)
                & (cb.order_lines["status"].isin(["completed", "delivering"]))
            ]
            on_dt = p_lines[p_lines["date"] == target_dt]
            if on_dt.empty:
                on_dt = p_lines[p_lines["date"] <= target_dt].sort_values("date")
            if not on_dt.empty:
                row = on_dt.iloc[-1]
                u_prc = float(row["unit_price"])
                row_fac = cb.unit_factors.get((ent.product_key, str(row["unit_norm"])), 1.0)
                base_prc = u_prc / row_fac if row_fac > 0 else u_prc
                if ent.target_unit_norm:
                    t_fac = cb.unit_factors.get((ent.product_key, ent.target_unit_norm), 1.0)
                    return ExecutionResult("number", round_money_vnd(base_prc * t_fac))
                return ExecutionResult("number", round_money_vnd(base_prc))
        return _unanswerable("Không tìm thấy đơn giá của mặt hàng tại ngày được hỏi.")

    if intent == "share_pct":
        return _execute_share_pct(pq, cb, tw, as_of_dt, ent, q_nk)

    # ------------------------------------------------------------------
    # Product & Phone Model Rankings & Unsold Products
    # ------------------------------------------------------------------
    if intent in ("top_selling_product", "top_revenue_product", "top_profit_product", "popular_product"):
        return _execute_product_ranking(pq, cb, tw, as_of_dt, ent)

    if intent == "top_phone_model":
        lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True, apply_product_filter=False)
        if lines.empty or cb.products.empty:
            return _unanswerable("Không có dữ liệu bán máy.")
        phone_pks = set(cb.products[cb.products["category_norm"].isin(["may moi", "may cu"])]["product_key"])
        pk_to_model = dict(zip(cb.products["product_key"], cb.products["phone_model"]))
        p_lines = lines[lines["product_key"].isin(phone_pks)].copy()
        if p_lines.empty:
            return ExecutionResult("list" if pq.top_k > 1 else "string", [] if pq.top_k > 1 else "")
        p_lines["phone_model"] = p_lines["product_key"].map(pk_to_model)
        grp = p_lines.groupby("phone_model", as_index=False)["base_qty"].sum()
        grp["norm_name"] = grp["phone_model"].apply(norm_key)
        grp = grp.sort_values(by=["base_qty", "norm_name"], ascending=[False, True])
        top_models = grp["phone_model"].head(pq.top_k).tolist()
        return ExecutionResult("list" if pq.top_k > 1 else "string", top_models if pq.top_k > 1 else (top_models[0] if top_models else ""))

    if intent in ("unsold_product_count", "unsold_product_list"):
        return _execute_unsold_products(pq, cb, tw, as_of_dt)

    # ------------------------------------------------------------------
    # Gross Profit, Expenses, Net Profit
    # ------------------------------------------------------------------
    if intent in ("gross_profit_total", "gross_margin_pct"):
        if not cb.capabilities.get("has_cost_price"):
            return _unanswerable("Dữ liệu của cửa hàng không có giá vốn hàng bán.")
        lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True, apply_product_filter=True)
        if lines.empty:
            return ExecutionResult("percent" if intent == "gross_margin_pct" else "number", 0.0 if intent == "gross_margin_pct" else 0)
        gp = float(lines["gross_profit"].dropna().sum())
        if intent == "gross_margin_pct":
            rev = _compute_revenue(cb, tw, as_of_dt, ent, q_nk)
            if rev == 0:
                return _unanswerable("Doanh thu bằng 0.")
            return ExecutionResult("percent", round_2dec(gp / rev * 100.0))
        return ExecutionResult("number", round_money_vnd(gp))

    if intent == "expense_total":
        if not cb.capabilities.get("has_expenses"):
            return _unanswerable("Dữ liệu không có sổ chi phí.")
        df_ex = cb.expenses[
            (cb.expenses["date"] >= tw.start_date) & (cb.expenses["date"] <= min(tw.end_date, as_of_dt))
        ]
        if ent.expense_category_norm:
            df_ex = df_ex[df_ex["category_norm"] == ent.expense_category_norm]
        return ExecutionResult("number", round_money_vnd(float(df_ex["amount"].sum()) if not df_ex.empty else 0.0))

    if intent == "net_profit_total":
        if not cb.capabilities.get("has_cost_price") or not cb.capabilities.get("has_expenses"):
            return _unanswerable("Dữ liệu không đủ giá vốn hoặc sổ chi phí để tính lãi ròng.")
        lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True, apply_product_filter=False)
        gp = float(lines["gross_profit"].dropna().sum()) if not lines.empty else 0.0
        df_ex = cb.expenses[
            (cb.expenses["date"] >= tw.start_date) & (cb.expenses["date"] <= min(tw.end_date, as_of_dt))
        ]
        ex = float(df_ex["amount"].sum()) if not df_ex.empty else 0.0
        return ExecutionResult("number", round_money_vnd(gp - ex))

    # ------------------------------------------------------------------
    # Customer, Staff, Day, & Hour Rankings
    # ------------------------------------------------------------------
    if intent in (
        "top_customer_by_spend", "top_customer_by_orders", "loyal_customer",
        "big_spender_customer", "fickle_customer",
    ):
        return _execute_customer_ranking(pq, cb, tw, as_of_dt, ent)

    if intent == "new_member_count":
        if cb.customers.empty or "reg_date" not in cb.customers.columns or cb.customers["reg_date"].isna().all():
            return _unanswerable("Không có dữ liệu ngày đăng ký thành viên.")
        df_c = cb.customers[
            cb.customers["reg_date"].notna()
            & (cb.customers["reg_date"] >= tw.start_date)
            & (cb.customers["reg_date"] <= min(tw.end_date, as_of_dt))
        ]
        return ExecutionResult("number", len(df_c))

    if intent == "customer_count_by_address":
        if not cb.capabilities.get("has_customer_address"):
            return _unanswerable("Dữ liệu của cửa hàng không ghi địa chỉ khách hàng.")
        q_addr = ent.address_query_norm or ""
        df_c = cb.customers[cb.customers["address_norm"].str.contains(re.escape(q_addr), na=False)]
        return ExecutionResult("number", len(df_c))

    if intent == "top_staff_by_revenue":
        if not cb.capabilities.get("has_staff"):
            return _unanswerable("Dữ liệu không có thông tin nhân viên bán hàng.")
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=None, valid_only=True)
        df_ord = df_ord[df_ord["staff_id"] != ""]
        if df_ord.empty:
            return _unanswerable("Không có đơn hàng của nhân viên trong kỳ.")
        grp = df_ord.groupby("staff_id", as_index=False)["revenue"].sum()
        grp["norm_id"] = grp["staff_id"].apply(norm_key)
        grp = grp.sort_values(by=["revenue", "norm_id"], ascending=[False, True])
        return ExecutionResult("string", str(grp.iloc[0]["staff_id"]))

    if intent == "hardworking_staff":
        if not cb.capabilities.get("has_staff"):
            return _unanswerable("Dữ liệu không có thông tin nhân viên.")
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=None, valid_only=False)
        df_ord = df_ord[df_ord["staff_id"] != ""]
        if df_ord.empty:
            return _unanswerable("Không có dữ liệu ngày làm việc của nhân viên.")
        grp = df_ord.groupby("staff_id", as_index=False)["date"].nunique()
        grp["norm_id"] = grp["staff_id"].apply(norm_key)
        grp = grp.sort_values(by=["date", "norm_id"], ascending=[False, True])
        return ExecutionResult("string", str(grp.iloc[0]["staff_id"]))

    if intent in ("busiest_day_by_orders", "busiest_day_by_revenue"):
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        if df_ord.empty:
            return _unanswerable("Không có đơn hàng trong kỳ.")
        if intent == "busiest_day_by_orders":
            if not cb.capabilities.get("has_order_id"):
                return _unanswerable("Không có mã đơn hàng.")
            grp = df_ord.groupby("date", as_index=False)["order_id"].nunique()
            grp = grp.sort_values(by=["order_id", "date"], ascending=[False, True])
        else:
            grp = df_ord.groupby("date", as_index=False)["revenue"].sum()
            grp = grp.sort_values(by=["revenue", "date"], ascending=[False, True])
        best_dt: dt.date = grp.iloc[0]["date"]
        return ExecutionResult("date", best_dt.isoformat())

    if intent == "peak_hour":
        if not cb.capabilities.get("has_hour_of_day"):
            return _unanswerable("Dữ liệu không có giờ tạo đơn.")
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        df_ord = df_ord[df_ord["hour"].notna()]
        if df_ord.empty:
            return _unanswerable("Không có dữ liệu giờ tạo đơn trong kỳ.")
        grp = df_ord.groupby("hour", as_index=False)["order_id"].nunique()
        grp = grp.sort_values(by=["order_id", "hour"], ascending=[False, True])
        return ExecutionResult("number", int(grp.iloc[0]["hour"]))

    # ------------------------------------------------------------------
    # Imports, Suppliers, Stock, & Batch Expiry
    # ------------------------------------------------------------------
    if intent in ("import_total_amount", "import_avg_unit_price", "top_supplier_by_amount", "main_supplier"):
        return _execute_imports_and_suppliers(pq, cb, tw, as_of_dt, ent)

    if intent in (
        "stock_ending_qty", "out_of_stock_product_count", "stock_ending_cost_value",
        "expiring_stock_cost", "expiring_batch_count", "expiring_product_count", "expiring_product_list", "nearest_expiry_date",
    ):
        return _execute_stock_and_expiry(pq, cb, tw, as_of_dt, ent)

    # ------------------------------------------------------------------
    # Debt, Repayments, Aging, Installments, Warranties, Deliveries, Due Dates
    # ------------------------------------------------------------------
    if intent in (
        "debt_total_remaining", "debt_customer_count", "debt_top_customers_table", "debt_top_customers_list",
        "debt_aging_buckets", "debt_over_n_days", "debt_repaid_in_period", "customer_debt_added_and_repaid",
    ):
        return _execute_debt(pq, cb, tw, as_of_dt, ent)

    if intent in (
        "installment_due_period_count", "installment_overdue_customer_count",
        "installment_overdue_amount", "installment_unpaid_amount",
    ):
        return _execute_installments(pq, cb, tw, as_of_dt, ent)

    if intent in ("warranty_count", "warranty_rate_pct", "warranty_top_model", "warranty_purchased_in_period_count"):
        return _execute_warranties(pq, cb, tw, as_of_dt, ent)

    if intent in ("trade_in_total_amount", "trade_in_order_pct"):
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        if df_ord.empty:
            return ExecutionResult("percent" if intent == "trade_in_order_pct" else "number", 0.0 if intent == "trade_in_order_pct" else 0)
        if intent == "trade_in_order_pct":
            n_trd = int((df_ord["trade_in_amount"] > 0).sum())
            return ExecutionResult("percent", round_2dec(n_trd / len(df_ord) * 100.0))
        return ExecutionResult("number", round_money_vnd(float(df_ord["trade_in_amount"].sum())))

    if intent in ("delivery_round_count", "delivery_multi_round_order_count", "delivery_product_qty"):
        return _execute_deliveries(pq, cb, tw, as_of_dt, ent)

    if intent in ("due_date_after_today_count", "due_date_overdue_count"):
        if not cb.capabilities.get("has_due_dates"):
            return _unanswerable("Không có sổ hạn nợ.")
        df_ord = _filter_orders_in_window(cb, TimeWindow(cb.data_start_date or dt.date(2025, 1, 1), as_of_dt), as_of_dt, valid_only=True)
        valid_oids = set(df_ord["order_id"])
        dd = cb.due_dates[cb.due_dates["order_id"].isin(valid_oids)]
        if intent == "due_date_after_today_count":
            cnt = int((dd["due_date"] > as_of_dt).sum())
        else:
            cnt = int((dd["due_date"] < as_of_dt).sum())
        return ExecutionResult("number", cnt)

    if intent in ("voucher_discount_total", "points_discount_total", "shipping_fee_total"):
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        col = {
            "voucher_discount_total": "voucher_discount",
            "points_discount_total": "points_discount",
            "shipping_fee_total": "shipping_fee",
        }[intent]
        val = float(df_ord[col].sum()) if not df_ord.empty and col in df_ord.columns else 0.0
        return ExecutionResult("number", round_money_vnd(val))

    return _abstain("Unhandled intent")



def _execute_share_pct(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
    q_nk: str,
) -> ExecutionResult:
    # 1. Prescription drug revenue share (pharmacy)
    if "ke don" in q_nk:
        if cb.domain != "pharmacy" or cb.products.empty:
            return _unanswerable("Cửa hàng không có danh mục thuốc kê đơn.")
        df_ord = _filter_orders_in_window(cb, tw, as_of_dt, valid_only=True)
        tot_rev = float(df_ord["revenue"].sum()) if not df_ord.empty else 0.0
        if tot_rev == 0:
            return _unanswerable("Tổng doanh thu kỳ bằng 0.")
        lines = _filter_lines_in_window(cb, tw, as_of_dt, valid_only=True, apply_product_filter=False)
        rx_pks = set(cb.products[cb.products["is_prescription"] == True]["product_key"])  # noqa: E712
        rx_rev = float(lines[lines["product_key"].isin(rx_pks)]["line_amount"].sum()) if not lines.empty else 0.0
        return ExecutionResult("percent", round_2dec(rx_rev / tot_rev * 100.0))

    # 2. Cafe size M / L share within a specific drink or all drinks
    if ent.cafe_size_filter in ("m", "l") and cb.capabilities.get("has_line_items"):
        lines_all = _filter_lines_in_window(cb, tw, as_of_dt, valid_only=True, apply_product_filter=False)
        if ent.cafe_base_keys:
            base_lines = lines_all[lines_all["product_key"].isin(ent.cafe_base_keys)]
        else:
            base_lines = lines_all[
                lines_all["product_name"].str.lower().str.strip().str.endswith(("(m)", "(l)"), na=False)
            ]
        if base_lines.empty:
            return ExecutionResult("percent", 0.0)
        target_sfx = f"({ent.cafe_size_filter})"
        sub_lines = base_lines[base_lines["product_name"].str.lower().str.strip().str.endswith(target_sfx, na=False)]
        metric_col = "line_amount" if any(k in q_nk for k in ("doanh thu", "dthu", "tien")) else "base_qty"
        denom = float(base_lines[metric_col].sum())
        if denom == 0:
            return ExecutionResult("percent", 0.0)
        numer = float(sub_lines[metric_col].sum())
        return ExecutionResult("percent", round_2dec(numer / denom * 100.0))

    # 3. Phone shop: share of new/used phones or brand in total phone revenue or total phone units
    if cb.domain == "phone_shop" and any(k in q_nk for k in ("tien ban may", "tong dthu may", "so may ban", "tong so may")):
        lines_all = _filter_lines_in_window(cb, tw, as_of_dt, valid_only=True, apply_product_filter=False)
        phone_lines = lines_all[lines_all["category_norm"].isin(["may moi", "may cu"])]
        if phone_lines.empty:
            return ExecutionResult("percent", 0.0)
        metric_col = "line_amount" if any(k in q_nk for k in ("tien", "doanh thu", "dthu")) else "base_qty"
        denom = float(phone_lines[metric_col].sum())
        if denom == 0:
            return ExecutionResult("percent", 0.0)
        sub = phone_lines
        if ent.category_norm in ("may moi", "may cu"):
            sub = sub[sub["category_norm"] == ent.category_norm]
        if ent.brand_norm and not cb.products.empty:
            b_pks = set(cb.products[cb.products["brand_norm"] == ent.brand_norm]["product_key"])
            sub = sub[sub["product_key"].isin(b_pks)]
        numer = float(sub[metric_col].sum())
        return ExecutionResult("percent", round_2dec(numer / denom * 100.0))

    # 4. Mother & Baby: diaper size / brand share in total diaper packs (Rulebook §6)
    if cb.domain == "mother_baby" and (ent.size_norm or ent.brand_norm) and any(k in q_nk for k in ("bim", "ta", "bich")):
        lines_all = _filter_lines_in_window(cb, tw, as_of_dt, valid_only=True, apply_product_filter=False)
        diaper_pks = set(cb.products[cb.products["category_norm"].str.contains("bim|ta", na=False)]["product_key"])
        d_lines = lines_all[lines_all["product_key"].isin(diaper_pks)]
        if d_lines.empty:
            return ExecutionResult("percent", 0.0)
        denom = float(d_lines["base_qty"].sum())
        if denom == 0:
            return ExecutionResult("percent", 0.0)
        sub_pks = set(diaper_pks)
        if ent.size_norm:
            sub_pks &= set(cb.products[cb.products["size_norm"] == ent.size_norm]["product_key"])
        if ent.brand_norm:
            sub_pks &= set(cb.products[cb.products["brand_norm"] == ent.brand_norm]["product_key"])
        numer = float(d_lines[d_lines["product_key"].isin(sub_pks)]["base_qty"].sum())
        return ExecutionResult("percent", round_2dec(numer / denom * 100.0))

    # 5. Order count share (% số đơn)
    if any(k in q_nk for k in ("so don", "tong don", "don hang")) and "doanh thu" not in q_nk and "dthu" not in q_nk:
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Không có mã đơn hàng.")
        all_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=None, valid_only=True)
        if all_ord.empty:
            return ExecutionResult("percent", 0.0)
        sub_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=ent, valid_only=True)
        return ExecutionResult("percent", round_2dec(len(sub_ord) / len(all_ord) * 100.0))

    # 6. Default: Revenue share (% tổng doanh thu cả shop)
    all_ord = _filter_orders_in_window(cb, tw, as_of_dt, ent=None, valid_only=True)
    tot_rev = float(all_ord["revenue"].sum()) if not all_ord.empty else 0.0
    if tot_rev == 0:
        return _unanswerable("Tổng doanh thu bằng 0.")
    sub_rev = _compute_revenue(cb, tw, as_of_dt, ent, q_nk)
    return ExecutionResult("percent", round_2dec(sub_rev / tot_rev * 100.0))


def _execute_product_ranking(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    if not cb.capabilities.get("has_line_items"):
        return _unanswerable("Dữ liệu không có chi tiết mặt hàng.")
    if pq.intent == "top_profit_product" and not cb.capabilities.get("has_cost_price"):
        return _unanswerable("Dữ liệu không có giá vốn để tính lãi gộp từng mặt hàng.")
    if pq.intent == "popular_product" and not cb.capabilities.get("has_order_id"):
        return _unanswerable("Dữ liệu không có mã đơn hàng.")

    # Filter lines (apply category/brand filter if present, but not single product_key)
    ent_filter = LinkedEntities(
        category_norm=ent.category_norm,
        brand_norm=ent.brand_norm,
        active_ingredient_norm=ent.active_ingredient_norm,
        room_area_norm=ent.room_area_norm,
        channel_norm=ent.channel_norm,
        customer_group_norm=ent.customer_group_norm,
        customer_gender=ent.customer_gender,
    )
    lines = _filter_lines_in_window(cb, tw, as_of_dt, ent=ent_filter, valid_only=True, apply_product_filter=True)
    if not lines.empty and "is_service_fee" in lines.columns:
        lines = lines[lines["is_service_fee"] == False]  # noqa: E712

    if lines.empty:
        return ExecutionResult("list" if pq.top_k > 1 else "string", [] if pq.top_k > 1 else "")

    if pq.intent == "top_selling_product":
        grp = lines.groupby("product_key", as_index=False)["base_qty"].sum().rename(columns={"base_qty": "metric"})
    elif pq.intent == "top_revenue_product":
        grp = lines.groupby("product_key", as_index=False)["line_amount"].sum().rename(columns={"line_amount": "metric"})
    elif pq.intent == "top_profit_product":
        grp = lines.groupby("product_key", as_index=False)["gross_profit"].sum().rename(columns={"gross_profit": "metric"})
    else:  # popular_product
        grp = lines.groupby("product_key", as_index=False)["order_id"].nunique().rename(columns={"order_id": "metric"})

    grp["product_name"] = grp["product_key"].map(lambda k: cb.prod_key_to_name.get(k, k))
    grp["tie_code"] = grp["product_key"].map(lambda k: cb.prod_key_to_code.get(k) or norm_key(cb.prod_key_to_name.get(k, k)))
    grp = grp.sort_values(by=["metric", "tie_code"], ascending=[pq.sort_asc, True])
    names = grp["product_name"].head(pq.top_k).tolist()
    if pq.top_k > 1:
        return ExecutionResult("list", names)
    return ExecutionResult("string", names[0] if names else "")


def _execute_unsold_products(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
) -> ExecutionResult:
    if not cb.capabilities.get("has_line_items"):
        return _unanswerable("Dữ liệu không có chi tiết mặt hàng.")
    lines = cb.order_lines
    valid_lines = lines[lines["status"].isin(["completed", "delivering"])]
    if "is_service_fee" in valid_lines.columns:
        valid_lines = valid_lines[valid_lines["is_service_fee"] == False]  # noqa: E712

    # Rulebook §3 (public_test): Mặt hàng không bán được (ế) trong kỳ =
    # mặt hàng đã từng bán trước kỳ (date < tw.start_date) nhưng trong kỳ có 0 đơn hợp lệ mua.
    sold_before = set(valid_lines[valid_lines["date"] < tw.start_date]["product_key"])
    if not sold_before and not cb.products.empty:
        # If query starts at data_start_date, fallback to active catalog products
        sold_before = set(cb.products[cb.products["is_service_fee"] == False]["product_key"])  # noqa: E712

    end_d = min(tw.end_date, as_of_dt)
    sold_in_period = set(
        valid_lines[(valid_lines["date"] >= tw.start_date) & (valid_lines["date"] <= end_d)]["product_key"]
    )
    unsold_keys = sorted(
        sold_before - sold_in_period,
        key=lambda k: (cb.prod_key_to_code.get(k) or norm_key(cb.prod_key_to_name.get(k, k))),
    )
    if pq.intent == "unsold_product_count":
        return ExecutionResult("number", len(unsold_keys))
    return ExecutionResult("list", [cb.prod_key_to_name.get(k, k) for k in unsold_keys])


def _execute_customer_ranking(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    df_all = _filter_orders_in_window(
        cb, tw, as_of_dt, ent=ent, valid_only=(pq.intent != "fickle_customer")
    )
    # Exclude walk-in customers (Rulebook §3)
    df_c = df_all[(df_all["is_walkin"] == False) & (df_all["customer_key"] != "")].copy()  # noqa: E712
    if pq.intent == "fickle_customer":
        df_c = df_c[df_c["status"] == "cancelled"]

    if df_c.empty:
        return _unanswerable("Không có dữ liệu khách hàng định danh trong kỳ.")

    if pq.intent == "top_customer_by_spend":
        grp = df_c.groupby("customer_key", as_index=False)["revenue"].sum().rename(columns={"revenue": "metric"})
    elif pq.intent == "top_customer_by_orders":
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Không có mã đơn hàng.")
        grp = df_c.groupby("customer_key", as_index=False)["order_id"].nunique().rename(columns={"order_id": "metric"})
    elif pq.intent == "loyal_customer":
        grp = df_c.groupby("customer_key", as_index=False)["date"].nunique().rename(columns={"date": "metric"})
    elif pq.intent == "big_spender_customer":
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Không có mã đơn hàng.")
        agg = df_c.groupby("customer_key", as_index=False).agg(
            tot_rev=("revenue", "sum"),
            n_ord=("order_id", "nunique"),
        )
        agg = agg[agg["n_ord"] >= 3].copy()
        if agg.empty:
            return _unanswerable("Không có khách hàng nào có từ 3 đơn trở lên trong kỳ.")
        agg["metric"] = agg["tot_rev"] / agg["n_ord"]
        grp = agg[["customer_key", "metric"]].copy()
    else:  # fickle_customer
        if not cb.capabilities.get("has_order_id"):
            return _unanswerable("Không có mã đơn hàng.")
        grp = df_c.groupby("customer_key", as_index=False)["order_id"].nunique().rename(columns={"order_id": "metric"})

    grp["customer_name"] = grp["customer_key"].map(lambda k: cb.cust_key_to_name.get(k, k))
    grp["tie_code"] = grp["customer_key"].map(lambda k: cb.cust_key_to_code.get(k) or norm_key(cb.cust_key_to_name.get(k, k)))
    grp = grp.sort_values(by=["metric", "tie_code"], ascending=[False, True])
    names = grp["customer_name"].head(pq.top_k).tolist()
    if pq.top_k > 1:
        return ExecutionResult("list", names)
    return ExecutionResult("string", names[0] if names else "")


def _execute_imports_and_suppliers(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    if not cb.capabilities.get("has_imports"):
        return _unanswerable("Dữ liệu của cửa hàng không có lịch sử nhập hàng.")
    if pq.intent in ("import_total_amount", "import_avg_unit_price", "top_supplier_by_amount") and not cb.capabilities.get("has_import_price"):
        return _unanswerable("Bảng nhập hàng của cửa hàng không ghi đơn giá/thành tiền nhập.")

    end_d = min(tw.end_date, as_of_dt)
    df_imp = cb.imports[
        (cb.imports["date"] >= tw.start_date)
        & (cb.imports["date"] <= end_d)
        & (cb.imports["status"] != "cancelled")
    ]

    if pq.intent == "import_total_amount":
        return ExecutionResult("number", round_money_vnd(float(df_imp["total_amount"].sum()) if not df_imp.empty else 0.0))

    if pq.intent == "top_supplier_by_amount":
        if df_imp.empty:
            return _unanswerable("Không có phiếu nhập hàng trong kỳ.")
        grp = df_imp.groupby(["supplier_key", "supplier_name"], as_index=False)["total_amount"].sum()
        grp["tie_code"] = grp["supplier_key"].apply(norm_key)
        grp = grp.sort_values(by=["total_amount", "tie_code"], ascending=[False, True])
        return ExecutionResult("string", str(grp.iloc[0]["supplier_name"]))

    if pq.intent == "main_supplier":
        if df_imp.empty:
            return _unanswerable("Không có phiếu nhập hàng trong kỳ.")
        grp = df_imp.groupby(["supplier_key", "supplier_name"], as_index=False)["import_id"].nunique()
        grp["tie_code"] = grp["supplier_key"].apply(norm_key)
        grp = grp.sort_values(by=["import_id", "tie_code"], ascending=[False, True])
        return ExecutionResult("string", str(grp.iloc[0]["supplier_name"]))

    # import_avg_unit_price
    if not ent.product_key:
        return _unanswerable("Không xác định được mặt hàng cần tính giá nhập bình quân.")
    il = cb.import_lines[
        (cb.import_lines["date"] >= tw.start_date)
        & (cb.import_lines["date"] <= end_d)
        & (cb.import_lines["status"] != "cancelled")
        & (cb.import_lines["product_key"] == ent.product_key)
    ]
    if il.empty or float(il["base_qty"].sum()) == 0:
        return _unanswerable("Mặt hàng không có lượt nhập nào trong kỳ được hỏi.")
    tot_amt = float(il["line_amount"].sum())
    tot_bqty = float(il["base_qty"].sum())
    t_unit = ent.target_unit_norm or norm_key(cb.prod_base_unit.get(ent.product_key, ""))
    fac = cb.unit_factors.get((ent.product_key, t_unit), 1.0)
    tot_target_qty = tot_bqty / fac if fac > 0 else tot_bqty
    return ExecutionResult("number", round_money_vnd(tot_amt / tot_target_qty))


def _execute_stock_and_expiry(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    intent = pq.intent

    # 1. Batch Expiry intents
    if intent in ("expiring_stock_cost", "expiring_batch_count", "expiring_product_count", "expiring_product_list", "nearest_expiry_date"):
        if not cb.capabilities.get("has_batch_expiry") or cb.batches.empty:
            return _unanswerable("Cửa hàng không quản lý tồn kho theo lô và hạn sử dụng.")
        b_df = cb.batches[cb.batches["remaining_base_qty"] > 0].copy()
        if intent == "nearest_expiry_date":
            if not ent.product_key:
                return _unanswerable("Không xác định được mặt hàng.")
            ref_d = tw.end_date if tw.is_cumulative else as_of_dt
            sub = b_df[
                (b_df["product_key"] == ent.product_key)
                & b_df["expiry_date"].notna()
                & (b_df["expiry_date"] >= ref_d)
            ].sort_values("expiry_date")
            if sub.empty:
                return _unanswerable("Không còn lô hàng nào chưa hết hạn.")
            best_exp: dt.date = sub.iloc[0]["expiry_date"]
            return ExecutionResult("date", best_exp.isoformat())

        sub = b_df[
            b_df["expiry_date"].notna()
            & (b_df["expiry_date"] >= tw.start_date)
            & (b_df["expiry_date"] <= tw.end_date)
        ].copy()
        if intent == "expiring_batch_count":
            return ExecutionResult("number", int(sub["batch_id"].nunique()))
        if intent == "expiring_product_count":
            return ExecutionResult("number", int(sub["product_key"].nunique()))
        if intent == "expiring_stock_cost":
            if sub["unit_cost_base"].isna().all() and not cb.batches["unit_cost_base"].notna().any():
                return _unanswerable("Không có giá nhập của lô hàng.")
            cost_sum = float((sub["remaining_base_qty"] * sub["unit_cost_base"].fillna(0.0)).sum())
            return ExecutionResult("number", round_money_vnd(cost_sum))
        # expiring_product_list: Rulebook §9: liệt kê hàng theo hạn dùng thì hạn sớm nhất của mặt hàng trước, cùng hạn thì theo mã
        if sub.empty:
            return ExecutionResult("list", [])
        grp = sub.groupby("product_key", as_index=False)["expiry_date"].min()
        grp["tie_code"] = grp["product_key"].map(lambda k: cb.prod_key_to_code.get(k) or norm_key(cb.prod_key_to_name.get(k, k)))
        grp = grp.sort_values(by=["expiry_date", "tie_code"], ascending=[True, True])
        return ExecutionResult("list", [cb.prod_key_to_name.get(k, k) for k in grp["product_key"]])

    # 2. Stock Snapshot / Out-of-Stock intents
    if not cb.capabilities.get("has_stock_data") or cb.stock_snapshot.empty:
        return _unanswerable("Dữ liệu của cửa hàng không có bảng báo cáo tồn kho.")

    if intent == "stock_ending_cost_value":
        if not cb.capabilities.get("has_cost_price"):
            return _unanswerable("Dữ liệu không có giá vốn mặt hàng để tính giá trị tồn kho.")
        df_s = cb.stock_snapshot.copy()
        if "cost_price" not in df_s.columns or df_s["cost_price"].isna().all():
            if not cb.products.empty and "cost_price" in cb.products.columns:
                pk_cost = dict(zip(cb.products["product_key"], cb.products["cost_price"]))
                df_s["cost_price"] = df_s["product_key"].map(pk_cost)
            else:
                return _unanswerable("Không có giá vốn mặt hàng.")
        tot_val = float((df_s["ending_base_qty"] * df_s["cost_price"].fillna(0.0)).sum())
        return ExecutionResult("number", round_money_vnd(tot_val))

    if intent == "out_of_stock_product_count":
        target_d = tw.end_date
        # If historical date and cb.stock_movements (the_kho) is available (e.g. dev_006 q_c1a3be0d7376)
        if not cb.stock_movements.empty and cb.data_end_date and target_d < cb.data_end_date:
            mv = cb.stock_movements[cb.stock_movements["date"] <= target_d]
            if mv.empty:
                return ExecutionResult("number", 0)
            last_per_prod = mv.groupby("product_key", as_index=False).last()
            return ExecutionResult("number", int((last_per_prod["ending_stock"] <= 0).sum()))
        return ExecutionResult("number", int((cb.stock_snapshot["ending_base_qty"] <= 0).sum()))

    # stock_ending_qty
    if not ent.product_key:
        return _unanswerable("Không xác định được mặt hàng cần tra cứu tồn kho.")
    s_row = cb.stock_snapshot[cb.stock_snapshot["product_key"] == ent.product_key]
    base_qty = float(s_row.iloc[0]["ending_base_qty"]) if not s_row.empty else 0.0

    if ent.target_unit_pair:
        lg_u, _ = ent.target_unit_pair
        fac = cb.unit_factors.get((ent.product_key, lg_u), 1.0)
        if fac <= 1.0:
            # Fallback: find largest conversion factor > 1 for this product
            cands = [f for (pk, _), f in cb.unit_factors.items() if pk == ent.product_key and f > 1.0]
            if cands:
                fac = min(cands)
        q_int = int(round(base_qty)) // int(round(fac))
        r_int = int(round(base_qty)) % int(round(fac))
        return ExecutionResult("list", [q_int, r_int])

    if ent.target_unit_norm:
        fac = cb.unit_factors.get((ent.product_key, ent.target_unit_norm), 1.0)
        return ExecutionResult("number", _format_qty(base_qty / fac if fac > 0 else base_qty))
    return ExecutionResult("number", _format_qty(base_qty))


def _compute_order_remaining_debts_at_date(
    cb: CanonicalBusiness, target_date: dt.date
) -> pd.DataFrame:
    """Return DataFrame of credit orders (date <= target_date, status != 'cancelled')
    with `remaining_debt` > 0 at `target_date` and `age_days = (target_date - date).days`.
    """
    if cb.orders.empty:
        return pd.DataFrame()
    df_o = cb.orders[
        (cb.orders["date"] <= target_date)
        & (cb.orders["status"] != "cancelled")
        & (cb.orders["initial_debt"] > 0)
    ].copy()
    if df_o.empty:
        return df_o

    if not cb.payments.empty:
        # In dev_014 (phone_shop), cb.payments only contains post-purchase debt/installment repayments.
        # In other businesses (SoQuy, thanh_toan, BanHang_PhieuThu), initial_debt = total_payable - same_day_paid,
        # so we subtract payments where order_date < payment_date <= target_date.
        oid_to_odt = dict(zip(df_o["order_id"], df_o["date"]))
        repaid_by_oid: Dict[str, float] = {}
        is_phone_shop = cb.domain == "phone_shop"
        for p_oid, p_dt, p_amt in zip(cb.payments["order_id"], cb.payments["date"], cb.payments["amount"]):
            if p_oid not in oid_to_odt or p_dt is None or p_dt > target_date:
                continue
            if is_phone_shop or p_dt > oid_to_odt[p_oid]:
                repaid_by_oid[p_oid] = repaid_by_oid.get(p_oid, 0.0) + float(p_amt)
        df_o["remaining_debt"] = [
            max(0.0, float(init_d) - repaid_by_oid.get(str(oid), 0.0))
            for oid, init_d in zip(df_o["order_id"], df_o["initial_debt"])
        ]
    else:
        df_o["remaining_debt"] = df_o["initial_debt"].astype(float)

    df_o = df_o[df_o["remaining_debt"] > 0.49].copy()
    df_o["age_days"] = df_o["date"].apply(lambda d: (target_date - d).days if isinstance(d, dt.date) else 0)
    return df_o


def _execute_debt(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    if not cb.capabilities.get("has_debt_data"):
        return _unanswerable("Dữ liệu của cửa hàng không theo dõi công nợ khách hàng.")

    intent = pq.intent
    target_d = min(tw.end_date, as_of_dt)

    if intent == "debt_repaid_in_period":
        if cb.payments.empty:
            return _unanswerable("Dữ liệu không có lịch sử thu nợ theo ngày.")
        df_o = cb.orders[(cb.orders["status"] != "cancelled") & (cb.orders["initial_debt"] > 0)]
        oid_to_odt = dict(zip(df_o["order_id"], df_o["date"]))
        is_phone_shop = cb.domain == "phone_shop"
        tot_repaid = 0.0
        for p_oid, p_dt, p_amt in zip(cb.payments["order_id"], cb.payments["date"], cb.payments["amount"]):
            if p_oid not in oid_to_odt or p_dt is None:
                continue
            if tw.start_date <= p_dt <= target_d and (is_phone_shop or p_dt > oid_to_odt[p_oid]):
                tot_repaid += float(p_amt)
        return ExecutionResult("number", round_money_vnd(tot_repaid))

    if intent == "customer_debt_added_and_repaid":
        if not ent.customer_key:
            return _unanswerable("Không xác định được khách hàng.")
        df_o = cb.orders[
            (cb.orders["customer_key"] == ent.customer_key)
            & (cb.orders["status"] != "cancelled")
            & (cb.orders["initial_debt"] > 0)
        ]
        in_per = df_o[(df_o["date"] >= tw.start_date) & (df_o["date"] <= target_d)]
        added = float(in_per["initial_debt"].sum()) if not in_per.empty else 0.0

        oid_to_odt = dict(zip(df_o["order_id"], df_o["date"]))
        is_phone_shop = cb.domain == "phone_shop"
        repaid = 0.0
        if not cb.payments.empty:
            for p_oid, p_dt, p_amt in zip(cb.payments["order_id"], cb.payments["date"], cb.payments["amount"]):
                if p_oid in oid_to_odt and p_dt is not None and tw.start_date <= p_dt <= target_d:
                    if is_phone_shop or p_dt > oid_to_odt[p_oid]:
                        repaid += float(p_amt)
        return ExecutionResult("list", [round_money_vnd(added), round_money_vnd(repaid)])

    # Point-in-time remaining debt intents at target_d
    df_rem = _compute_order_remaining_debts_at_date(cb, target_d)

    if intent == "debt_total_remaining":
        tot = float(df_rem["remaining_debt"].sum()) if not df_rem.empty else 0.0
        return ExecutionResult("number", round_money_vnd(tot))

    if intent == "debt_customer_count":
        if df_rem.empty:
            return ExecutionResult("number", 0)
        grp = df_rem[df_rem["customer_key"] != ""].groupby("customer_key", as_index=False)["remaining_debt"].sum()
        grp = grp[grp["remaining_debt"] >= 0.5]
        return ExecutionResult("number", len(grp))

    if intent == "debt_aging_buckets":
        if df_rem.empty:
            return ExecutionResult("list", [0, 0, 0])
        b1 = float(df_rem[df_rem["age_days"] <= 7]["remaining_debt"].sum())
        b2 = float(df_rem[(df_rem["age_days"] >= 8) & (df_rem["age_days"] <= 15)]["remaining_debt"].sum())
        b3 = float(df_rem[df_rem["age_days"] > 15]["remaining_debt"].sum())
        return ExecutionResult("list", [round_money_vnd(b1), round_money_vnd(b2), round_money_vnd(b3)])

    if intent == "debt_over_n_days":
        n_d = pq.debt_age_days if pq.debt_age_days is not None else 15
        if df_rem.empty:
            return ExecutionResult("number", 0)
        tot = float(df_rem[df_rem["age_days"] > n_d]["remaining_debt"].sum())
        return ExecutionResult("number", round_money_vnd(tot))

    # debt_top_customers_table or debt_top_customers_list
    if df_rem.empty:
        if intent == "debt_top_customers_table":
            return ExecutionResult("table", {"columns": ["khach_hang", "so_tien_no"], "rows": []})
        return ExecutionResult("list" if pq.top_k > 1 else "string", [] if pq.top_k > 1 else "")

    grp = df_rem[df_rem["customer_key"] != ""].groupby("customer_key", as_index=False)["remaining_debt"].sum()
    grp["debt_vnd"] = grp["remaining_debt"].apply(round_money_vnd)
    grp = grp[grp["debt_vnd"] > 0].copy()
    grp["customer_name"] = grp["customer_key"].map(lambda k: cb.cust_key_to_name.get(k, k))
    grp["tie_code"] = grp["customer_key"].map(lambda k: cb.cust_key_to_code.get(k) or norm_key(cb.cust_key_to_name.get(k, k)))
    grp = grp.sort_values(by=["debt_vnd", "tie_code"], ascending=[False, True]).head(pq.top_k)

    if intent == "debt_top_customers_table":
        rows = [[str(r["customer_name"]), int(r["debt_vnd"])] for _, r in grp.iterrows()]
        return ExecutionResult("table", {"columns": ["khach_hang", "so_tien_no"], "rows": rows})

    names = grp["customer_name"].tolist()
    return ExecutionResult("list" if pq.top_k > 1 else "string", names if pq.top_k > 1 else (names[0] if names else ""))


def _add_months(d: dt.date, n_months: int, due_day: int) -> dt.date:
    y = d.year + (d.month - 1 + n_months) // 12
    m = (d.month - 1 + n_months) % 12 + 1
    max_d = py_calendar.monthrange(y, m)[1]
    return dt.date(y, m, min(due_day, max_d))


def _execute_installments(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    if not cb.capabilities.get("has_installments") or cb.installments.empty:
        return _unanswerable("Cửa hàng không có sổ theo dõi trả góp.")

    intent = pq.intent
    target_d = min(tw.end_date, as_of_dt)
    df_inst = cb.installments[cb.installments["date"] <= target_d].copy()
    if ent.customer_key:
        df_inst = df_inst[df_inst["customer_key"] == ent.customer_key]

    # Pre-sum installment payments up to target_d per order_id
    paid_by_oid: Dict[str, float] = {}
    if not cb.installment_payments.empty:
        ip = cb.installment_payments[cb.installment_payments["date"] <= target_d]
        for oid, amt in zip(ip["order_id"], ip["amount"]):
            paid_by_oid[str(oid)] = paid_by_oid.get(str(oid), 0.0) + float(amt)

    if intent == "installment_due_period_count":
        if df_inst.empty:
            return _unanswerable("Không tìm thấy hợp đồng trả góp của khách hàng.")
        tot_due_periods = 0
        for _, r in df_inst.iterrows():
            b_dt = r["date"]
            n_p = int(r["num_periods"])
            d_day = int(r["due_day"])
            for k in range(1, n_p + 1):
                if _add_months(b_dt, k, d_day) <= target_d:
                    tot_due_periods += 1
        return ExecutionResult("number", tot_due_periods)

    if intent == "installment_unpaid_amount":
        tot_unpaid = 0.0
        for _, r in df_inst.iterrows():
            oid = str(r["order_id"])
            init_d = float(r["initial_debt"])
            paid = paid_by_oid.get(oid, 0.0)
            tot_unpaid += max(0.0, init_d - paid)
        return ExecutionResult("number", round_money_vnd(tot_unpaid))

    # Overdue installment calculations (Rulebook §6: Kỳ góp có hạn TRƯỚC ngày hỏi mà chưa góp đủ là quá hạn;
    # kỳ có hạn đúng ngày hỏi thì chưa quá hạn -> due_date < target_d)
    overdue_by_cust: Dict[str, float] = {}
    tot_overdue_amt = 0.0
    for _, r in df_inst.iterrows():
        oid = str(r["order_id"])
        ck = str(r["customer_key"])
        b_dt = r["date"]
        n_p = int(r["num_periods"])
        m_amt = float(r["monthly_amount"])
        d_day = int(r["due_day"])
        n_past_due = sum(1 for k in range(1, n_p + 1) if _add_months(b_dt, k, d_day) < target_d)
        expected_paid = n_past_due * m_amt
        actual_paid = paid_by_oid.get(oid, 0.0)
        ovd = max(0.0, expected_paid - actual_paid)
        if ovd > 0.49:
            tot_overdue_amt += ovd
            overdue_by_cust[ck] = overdue_by_cust.get(ck, 0.0) + ovd

    if intent == "installment_overdue_customer_count":
        return ExecutionResult("number", len(overdue_by_cust))
    return ExecutionResult("number", round_money_vnd(tot_overdue_amt))


def _execute_warranties(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    if cb.warranties.empty:
        return _unanswerable("Cửa hàng không có sổ bảo hành.")
    intent = pq.intent
    end_d = min(tw.end_date, as_of_dt)

    if intent == "warranty_purchased_in_period_count":
        # Phones purchased in [tw.start_date, end_d] that were brought in for warranty up to as_of_dt
        w_sub = cb.warranties[
            cb.warranties["purchase_date"].notna()
            & (cb.warranties["purchase_date"] >= tw.start_date)
            & (cb.warranties["purchase_date"] <= end_d)
            & cb.warranties["receive_date"].notna()
            & (cb.warranties["receive_date"] <= as_of_dt)
        ]
        return ExecutionResult("number", int(w_sub["imei"].nunique()) if not w_sub.empty else 0)

    w_in = cb.warranties[
        cb.warranties["receive_date"].notna()
        & (cb.warranties["receive_date"] >= tw.start_date)
        & (cb.warranties["receive_date"] <= end_d)
    ]
    if intent == "warranty_count":
        return ExecutionResult("number", len(w_in))

    if intent == "warranty_top_model":
        if w_in.empty:
            return _unanswerable("Không có lượt bảo hành trong kỳ.")
        grp = w_in.groupby("device_model", as_index=False)["warranty_id"].count()
        grp["norm_name"] = grp["device_model"].apply(norm_key)
        grp = grp.sort_values(by=["warranty_id", "norm_name"], ascending=[False, True])
        return ExecutionResult("string", str(grp.iloc[0]["device_model"]))

    # warranty_rate_pct (Rulebook §6 Phone Shop: số lượt bảo hành có ngày nhận trong kỳ / số máy bán trong kỳ * 100)
    lines = _filter_lines_in_window(cb, tw, as_of_dt, valid_only=True, apply_product_filter=False)
    phone_lines = lines[lines["category_norm"].isin(["may moi", "may cu"])]
    n_phones_sold = float(phone_lines["base_qty"].sum()) if not phone_lines.empty else 0.0
    if n_phones_sold == 0:
        return _unanswerable("Số máy bán trong kỳ bằng 0.")
    return ExecutionResult("percent", round_2dec(len(w_in) / n_phones_sold * 100.0))


def _execute_deliveries(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    tw: TimeWindow,
    as_of_dt: dt.date,
    ent: LinkedEntities,
) -> ExecutionResult:
    if not cb.capabilities.get("has_deliveries") or cb.deliveries.empty:
        return _unanswerable("Cửa hàng không có bảng kê phiếu giao hàng.")

    intent = pq.intent
    end_d = min(tw.end_date, as_of_dt)

    if intent == "delivery_round_count":
        # Rulebook §6 Construction: Số đợt giao trong tháng X = số đợt có ngày giao trong tháng X;
        # mỗi đợt của mỗi đơn tính một đợt, kể cả đợt của đơn sau đó trả hàng.
        dl = cb.deliveries[
            cb.deliveries["date"].notna()
            & (cb.deliveries["date"] >= tw.start_date)
            & (cb.deliveries["date"] <= end_d)
        ]
        if dl.empty:
            return ExecutionResult("number", 0)
        n_rounds = dl[["order_id", "delivery_round"]].drop_duplicates().shape[0]
        return ExecutionResult("number", int(n_rounds))

    # For delivery_multi_round_order_count and delivery_product_qty:
    # Filter orders placed in [tw.start_date, end_d], and deliveries up to as_of_dt
    df_ord = cb.orders[
        (cb.orders["date"] >= tw.start_date)
        & (cb.orders["date"] <= end_d)
        & (cb.orders["status"] != "cancelled")
    ]
    if df_ord.empty:
        return ExecutionResult("number", 0)
    ord_ids = set(df_ord["order_id"])
    dl = cb.deliveries[
        cb.deliveries["order_id"].isin(ord_ids)
        & cb.deliveries["date"].notna()
        & (cb.deliveries["date"] <= as_of_dt)
    ]
    if dl.empty:
        return ExecutionResult("number", 0)

    if intent == "delivery_multi_round_order_count":
        grp = dl.groupby("order_id", as_index=False)["delivery_round"].nunique()
        return ExecutionResult("number", int((grp["delivery_round"] >= 2).sum()))

    # delivery_product_qty
    if not ent.product_key:
        return _unanswerable("Không xác định được mặt hàng giao.")
    p_dl = dl[dl["product_key"] == ent.product_key]
    if p_dl.empty:
        return ExecutionResult("number", 0)
    tot_base = float(p_dl["base_qty"].sum())
    if ent.target_unit_norm:
        fac = cb.unit_factors.get((ent.product_key, ent.target_unit_norm), 1.0)
        return ExecutionResult("number", _format_qty(tot_base / fac if fac > 0 else tot_base))
    return ExecutionResult("number", _format_qty(tot_base))


def _match_repair_work(work_items: List[str], target_work: str) -> bool:
    for w in work_items:
        if target_work == "nhot may":
            if w in ("nhot", "thay nhot", "nhot may", "thay nhot may"):
                return True
        elif target_work == "nhot lap":
            if "nhot lap" in w or "nhot hop so" in w:
                return True
        elif target_work == "thay ruot":
            if w in ("ruot", "thay ruot", "ruot moi"):
                return True
        elif target_work == "va":
            if w == "va" or w.startswith("va "):
                return True
        elif target_work == "sua dien":
            # Rulebook §6: Sửa điện không gồm thay bóng đèn, thay ắc quy
            if "dien" in w and "bong den" not in w and "ac quy" not in w and "binh" not in w:
                return True
        elif target_work in w:
            return True
    return False


def _execute_auto_repair(
    pq: ParsedQuestion,
    cb: CanonicalBusiness,
    cal: LunarSolarCalendar,
) -> ExecutionResult:
    df_t = cb.repair_tickets
    if df_t.empty:
        return _unanswerable("Không có dữ liệu sổ sửa xe.")

    intent = pq.intent
    tw = pq.time_window
    ent = pq.entities
    as_of_dt = pq.as_of_date
    q_nk = norm_key(pq.question_text)
    end_d = min(tw.end_date, as_of_dt)

    sub = df_t[(df_t["date"] >= tw.start_date) & (df_t["date"] <= end_d)].copy()
    if tw.specific_weekday is not None:
        sub = sub[sub["date"].apply(lambda d: d.isoweekday() == tw.specific_weekday if isinstance(d, dt.date) else False)]
    if ent.mechanic_norm:
        sub = sub[sub["mechanic_norm"] == ent.mechanic_norm]
    if ent.vehicle_model_norm:
        sub = sub[sub["vehicle_model_norm"] == ent.vehicle_model_norm]
    if ent.license_plate:
        sub = sub[sub["license_plate"] == ent.license_plate]
    if ent.repair_work_norm:
        # Rulebook §6: Phiếu bảo hành không tính là lượt làm việc đó
        sub = sub[
            (sub["is_warranty"] == False)  # noqa: E712
            & sub["work_items"].apply(lambda ws: _match_repair_work(ws, ent.repair_work_norm or ""))
        ]

    if intent == "repair_revenue_total":
        rev = float(sub["amount"].sum()) if not sub.empty else 0.0
        if tw.specific_weekday is not None and any(k in q_nk for k in ("trung binh", "binh quan", "moi ngay thu", "moi chu nhat")):
            n_wd = cal.count_weekday_in_range(
                max(tw.start_date, cb.data_start_date or tw.start_date),
                min(end_d, cb.data_end_date or end_d),
                tw.specific_weekday,
            )
            return ExecutionResult("number", round_money_vnd(rev / max(1, n_wd)))
        return ExecutionResult("number", round_money_vnd(rev))

    if intent == "repair_revenue_change_pct" and tw.compare_window is not None:
        cw = tw.compare_window
        sub_p = df_t[(df_t["date"] >= cw.start_date) & (df_t["date"] <= min(cw.end_date, as_of_dt))]
        if ent.mechanic_norm:
            sub_p = sub_p[sub_p["mechanic_norm"] == ent.mechanic_norm]
        if ent.vehicle_model_norm:
            sub_p = sub_p[sub_p["vehicle_model_norm"] == ent.vehicle_model_norm]
        if ent.repair_work_norm:
            sub_p = sub_p[
                (sub_p["is_warranty"] == False)  # noqa: E712
                & sub_p["work_items"].apply(lambda ws: _match_repair_work(ws, ent.repair_work_norm or ""))
            ]
        c_rev = float(sub["amount"].sum()) if not sub.empty else 0.0
        p_rev = float(sub_p["amount"].sum()) if not sub_p.empty else 0.0
        if p_rev == 0:
            return _unanswerable("Doanh thu kỳ trước bằng 0.")
        return ExecutionResult("percent", round_2dec((c_rev - p_rev) / p_rev * 100.0))

    if intent == "repair_ticket_count":
        cnt = len(sub)
        if tw.specific_weekday is not None and any(k in q_nk for k in ("trung binh", "binh quan", "moi ngay thu", "moi chu nhat")):
            n_wd = cal.count_weekday_in_range(
                max(tw.start_date, cb.data_start_date or tw.start_date),
                min(end_d, cb.data_end_date or end_d),
                tw.specific_weekday,
            )
            return ExecutionResult("number", round_2dec(cnt / max(1, n_wd)))
        return ExecutionResult("number", cnt)

    if intent == "repair_distinct_vehicles":
        return ExecutionResult("number", int(sub["license_plate"].nunique()) if not sub.empty else 0)

    if intent == "repair_avg_paid_ticket":
        paid_sub = sub[sub["is_paid"] == True]  # noqa: E712
        if paid_sub.empty:
            return ExecutionResult("number", 0)
        return ExecutionResult("number", round_money_vnd(float(paid_sub["amount"].sum()) / len(paid_sub)))

    if intent == "repair_mechanic_commission":
        paid_sub = sub[sub["is_paid"] == True]  # noqa: E712
        comm = float((paid_sub["amount"] * paid_sub["commission_pct"] / 100.0).sum()) if not paid_sub.empty else 0.0
        return ExecutionResult("number", round_money_vnd(comm))

    if intent == "repair_warranty_count":
        return ExecutionResult("number", int((sub["is_warranty"] == True).sum()))  # noqa: E712

    if intent == "repair_warranty_rate_pct":
        n_war = int((sub["is_warranty"] == True).sum())  # noqa: E712
        n_paid = int((sub["is_paid"] == True).sum())  # noqa: E712
        if n_paid == 0:
            return _unanswerable("Số phiếu có thu tiền bằng 0.")
        return ExecutionResult("percent", round_2dec(n_war / n_paid * 100.0))

    if intent in ("repair_repeat_vehicle_count", "repair_top_repeat_vehicle"):
        # Rulebook §6: Chỉ đếm phiếu có thu tiền; Xe khách quen = xe có từ 2 phiếu có thu tiền trong kỳ
        paid_sub = sub[(sub["is_paid"] == True) & (sub["license_plate"] != "")]  # noqa: E712
        if paid_sub.empty:
            return ExecutionResult("number" if intent == "repair_repeat_vehicle_count" else "string", 0 if intent == "repair_repeat_vehicle_count" else "")
        grp = paid_sub.groupby("license_plate", as_index=False)["ticket_id"].count()
        if intent == "repair_repeat_vehicle_count":
            return ExecutionResult("number", int((grp["ticket_id"] >= 2).sum()))
        grp = grp.sort_values(by=["ticket_id", "license_plate"], ascending=[False, True])
        return ExecutionResult("string", str(grp.iloc[0]["license_plate"]))

    if intent in ("repair_top_mechanic_by_tickets", "repair_top_mechanic_by_revenue", "hardworking_staff"):
        base_t = df_t[(df_t["date"] >= tw.start_date) & (df_t["date"] <= end_d)]
        if base_t.empty:
            return _unanswerable("Không có phiếu sửa xe trong kỳ.")
        if intent == "repair_top_mechanic_by_tickets":
            grp = base_t.groupby("mechanic_name", as_index=False)["ticket_id"].count().rename(columns={"ticket_id": "metric"})
        elif intent == "repair_top_mechanic_by_revenue":
            grp = base_t.groupby("mechanic_name", as_index=False)["amount"].sum().rename(columns={"amount": "metric"})
        else:  # hardworking_staff
            grp = base_t.groupby("mechanic_name", as_index=False)["date"].nunique().rename(columns={"date": "metric"})
        grp["tie_code"] = grp["mechanic_name"].apply(norm_key)
        grp = grp.sort_values(by=["metric", "tie_code"], ascending=[False, True])
        return ExecutionResult("string", str(grp.iloc[0]["mechanic_name"]))

    if intent == "repair_top_vehicle_model":
        if sub.empty:
            return ExecutionResult("list" if pq.top_k > 1 else "string", [] if pq.top_k > 1 else "")
        grp = sub.groupby("vehicle_model", as_index=False)["ticket_id"].count()
        grp["tie_code"] = grp["vehicle_model"].apply(norm_key)
        grp = grp.sort_values(by=["ticket_id", "tie_code"], ascending=[False, True])
        models = grp["vehicle_model"].head(pq.top_k).tolist()
        return ExecutionResult("list" if pq.top_k > 1 else "string", models if pq.top_k > 1 else (models[0] if models else ""))

    return _abstain("Unhandled auto_repair intent")
