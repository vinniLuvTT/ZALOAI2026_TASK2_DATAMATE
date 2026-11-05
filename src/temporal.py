"""Deterministic Temporal Resolver for DataMate (Zalo AI Challenge 2026).

Resolves all Vietnamese relative and absolute temporal expressions strictly
against `as_of_date` and `calendar.csv` per Rulebook Section 2.
"""

from __future__ import annotations

import calendar as py_calendar
import csv
import datetime as dt
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TimeWindow:
    start_date: dt.date
    end_date: dt.date
    label: str = ""
    is_cumulative: bool = False  # True for "tính đến hết ngày X", "hết hôm nay"
    shift_filter: Optional[str] = None  # "morning" (06:00-13:59) or "afternoon" (14:00-21:59)
    compare_window: Optional["TimeWindow"] = None
    multi_windows: List["TimeWindow"] = field(default_factory=list)
    specific_weekday: Optional[int] = None  # 1=Mon .. 7=Sun for "trung bình mỗi ngày thứ X"


class LunarSolarCalendar:
    """Loads `calendar.csv` and provides exact lunar-to-solar lookups."""

    def __init__(self, calendar_csv_path: Path | str) -> None:
        self.by_solar: Dict[dt.date, Dict[str, int | bool | str]] = {}
        self.by_lunar: Dict[Tuple[int, int, int, bool], dt.date] = {}
        with open(calendar_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s_date = dt.date.fromisoformat(row["date"].strip())
                l_day = int(row["lunar_day"])
                l_month = int(row["lunar_month"])
                l_year = int(row["lunar_year"])
                l_leap = row["lunar_leap"].strip().lower() == "true"
                wday = int(row["weekday"])
                hol = row.get("holiday", "").strip()
                self.by_solar[s_date] = {
                    "weekday": wday,
                    "lunar_day": l_day,
                    "lunar_month": l_month,
                    "lunar_year": l_year,
                    "lunar_leap": l_leap,
                    "holiday": hol,
                }
                self.by_lunar[(l_year, l_month, l_day, l_leap)] = s_date

    def get_tet_window(self, solar_year: int) -> Tuple[dt.date, dt.date, dt.date, dt.date]:
        """Rulebook §2:
        Tết năm Y, dịp Tết, mùa Tết: từ ngày dương của 23 tháng Chạp (năm âm Y - 1)
        đến hết ngày dương của mùng 10 tháng Giêng (năm âm Y).
        So với trước Tết = cùng số ngày ngay trước 23 tháng Chạp.
        Returns: (tet_start, tet_end, pre_tet_start, pre_tet_end)
        """
        # Find 23/12 of lunar year (solar_year - 1) (prefer non-leap or leap if exists)
        start = self.by_lunar.get((solar_year - 1, 12, 23, False)) or self.by_lunar.get(
            (solar_year - 1, 12, 23, True)
        )
        end = self.by_lunar.get((solar_year, 1, 10, False))
        if not start or not end:
            raise ValueError(f"Lunar calendar does not cover Tet {solar_year}")
        num_days = (end - start).days + 1
        pre_end = start - dt.timedelta(days=1)
        pre_start = pre_end - dt.timedelta(days=num_days - 1)
        return start, end, pre_start, pre_end

    def count_weekday_in_range(self, start: dt.date, end: dt.date, weekday: int) -> int:
        """Rulebook §2 (public_test):
        Trung bình mỗi ngày thứ X của kỳ: chia cho số ngày thứ X trong kỳ,
        kể cả ngày không có đơn, phiếu.
        `weekday` follows calendar.csv convention: 1=Mon, 2=Tue, ..., 6=Sat, 7=Sun.
        """
        if end < start:
            return 0
        cnt = 0
        cur = start
        while cur <= end:
            if cur.isoweekday() == weekday:
                cnt += 1
            cur += dt.timedelta(days=1)
        return cnt


def last_day_of_month(year: int, month: int) -> int:
    return py_calendar.monthrange(year, month)[1]


def month_bounds(year: int, month: int) -> Tuple[dt.date, dt.date]:
    return dt.date(year, month, 1), dt.date(year, month, last_day_of_month(year, month))


def prev_month_ym(year: int, month: int) -> Tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def quarter_bounds(year: int, quarter: int) -> Tuple[dt.date, dt.date]:
    m_start = (quarter - 1) * 3 + 1
    m_end = m_start + 2
    return dt.date(year, m_start, 1), dt.date(year, m_end, last_day_of_month(year, m_end))


def prev_quarter_yq(year: int, quarter: int) -> Tuple[int, int]:
    if quarter == 1:
        return year - 1, 4
    return year, quarter - 1


def parse_flexible_date(val: object, default_year: Optional[int] = None) -> Optional[dt.date]:
    """Parse any date cell from CSV/Excel/SQLite into a `datetime.date`.

    Handles:
    - `datetime.datetime` and `datetime.date`
    - Excel serial numbers (e.g., 45658 -> 2025-01-01)
    - ISO strings `YYYY-MM-DD` (with or without timestamp)
    - Vietnamese/slash strings `DD/MM/YYYY`, `D/M/YYYY`, `DD-MM-YYYY`
    - Vietnamese `D thg M YYYY` (e.g., `1 thg 7 2025`)
    - Yearless `D/M` or `D thg M` using `default_year` per Rulebook §8
    """
    if val is None:
        return None
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, dt.date):
        return val
    if isinstance(val, (int, float)):
        if isinstance(val, bool) or not math.isfinite(float(val)):
            return None
        ival = int(val)
        if 30000 <= ival <= 60000:
            return dt.date(1899, 12, 30) + dt.timedelta(days=ival)
        return None

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None

    # Pure integer string representing Excel serial date
    if re.match(r"^\d{5}$", s):
        ival = int(s)
        if 30000 <= ival <= 60000:
            return dt.date(1899, 12, 30) + dt.timedelta(days=ival)

    # ISO YYYY-MM-DD [HH:MM:SS]
    m_iso = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:\b|\s|T)", s)
    if m_iso:
        return dt.date(int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3)))

    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    m_dmy = re.match(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})(?:\b|\s|$)", s)
    if m_dmy:
        d, m, y = int(m_dmy.group(1)), int(m_dmy.group(2)), int(m_dmy.group(3))
        if y < 100:
            y += 2000
        return dt.date(y, m, d)

    # "1 thg 7 2025" or "1 tháng 7 2025" or "01 thg 07, 2025"
    m_thg = re.match(
        r"^(\d{1,2})\s*(?:thg|tháng|thang)\s*(\d{1,2})(?:[\s,/.-]+(\d{2,4}))?$",
        s,
        flags=re.IGNORECASE,
    )
    if m_thg:
        d, m = int(m_thg.group(1)), int(m_thg.group(2))
        y = int(m_thg.group(3)) if m_thg.group(3) else default_year
        if y is not None:
            if y < 100:
                y += 2000
            return dt.date(y, m, d)

    # Yearless DD/MM (Rulebook §8: "Ngày không ghi năm: năm của sheet chứa dòng đó")
    m_dm = re.match(r"^(\d{1,2})[/\-.](\d{1,2})$", s)
    if m_dm and default_year is not None:
        return dt.date(default_year, int(m_dm.group(2)), int(m_dm.group(1)))

    return None


def parse_flexible_hour(val: object) -> Optional[int]:
    """Extract hour (0..23) from a time/datetime cell or Excel serial float."""
    if val is None:
        return None
    if isinstance(val, dt.datetime):
        return val.hour
    if isinstance(val, dt.time):
        return val.hour
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        fval = float(val)
        if not math.isfinite(fval):
            return None
        frac = fval - int(fval)
        if frac > 1e-7:
            total_sec = int(round(frac * 86400)) % 86400
            return total_sec // 3600
        if 0 <= fval < 24 and fval == int(fval):
            return int(fval)
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    m = re.search(r"(?:^|\s|T)(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h
    # Check float string (Excel serial datetime stored as string)
    if re.match(r"^\d{5}\.\d+$", s):
        fval = float(s)
        frac = fval - int(fval)
        total_sec = int(round(frac * 86400)) % 86400
        return total_sec // 3600
    return None



def _strip_acc(text: str) -> str:
    if not text:
        return ""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def resolve_question_time_window(
    question: str,
    as_of_date: dt.date,
    cal: LunarSolarCalendar,
    data_start_date: Optional[dt.date] = None,
) -> TimeWindow:
    """Deterministically resolve the question's temporal scope relative to `as_of_date`."""
    from src.canonical import normalize_question_text

    q_raw = normalize_question_text(question.strip())
    q_low = q_raw.lower()
    q_noacc = _strip_acc(q_low)

    # Detect shift filter (Rulebook §2: ca sáng 06:00-13:59, ca chiều 14:00-21:59)
    shift_filter: Optional[str] = None
    has_both_shifts = ("ca sang" in q_noacc and "ca chieu" in q_noacc)
    if not has_both_shifts:
        if "ca sang" in q_noacc or "buoi sang" in q_noacc:
            shift_filter = "morning"
        elif "ca chieu" in q_noacc or "buoi chieu" in q_noacc:
            shift_filter = "afternoon"

    # Detect specific weekday filter ("trung bình mỗi ngày thứ hai", "các ngày chủ nhật")
    # Match on accented q_low so "doanh thu từ" ("thu tu") or "doanh thu năm" ("thu nam") never collide!
    specific_wday: Optional[int] = None
    wday_map = [
        (r"(?<!doanh\s)\bthứ\s*(?:hai|2)\b", 1),
        (r"(?<!doanh\s)\bthứ\s*(?:ba|3)\b", 2),
        (r"(?<!doanh\s)\bthứ\s*(?:tư|4)\b", 3),
        (r"(?<!doanh\s)\bthứ\s*(?:năm|5)\b", 4),
        (r"(?<!doanh\s)\bthứ\s*(?:sáu|6)\b", 5),
        (r"(?<!doanh\s)\bthứ\s*(?:bảy|7)\b", 6),
        (r"\bchủ\s*nhật\b", 7),
    ]
    for pat, wd in wday_map:
        if re.search(pat, q_low):
            specific_wday = wd
            break

    # 1. Price change +/- N days window ("từ ngày DD/MM/YYYY ... 14 ngày kể từ ngày ... so với 14 ngày trước")
    m_prc_win = re.search(
        r"(\d{1,2})[/\-](\d{1,2})[/\-](202\d).*?(\d+)\s*ngay\s*(?:dau|ke\s*tu|sau).*?(\d+)\s*ngay\s*(?:ngay\s*|lien\s*)?truoc",
        q_noacc,
    )
    if m_prc_win:
        d0 = dt.date(int(m_prc_win.group(3)), int(m_prc_win.group(2)), int(m_prc_win.group(1)))
        n_days = int(m_prc_win.group(4))
        w_cur = TimeWindow(
            start_date=d0,
            end_date=d0 + dt.timedelta(days=n_days - 1),
            label="price_change_post",
        )
        w_cur.compare_window = TimeWindow(
            start_date=d0 - dt.timedelta(days=n_days),
            end_date=d0 - dt.timedelta(days=1),
            label="price_change_pre",
        )
        return w_cur

    m_prc_win2 = re.search(
        r"(\d+)\s*ngay\s*(?:dau\s*)?(?:ke\s*tu|sau)\s*(?:ngay\s*)?(\d{1,2})[/\-](\d{1,2})[/\-](202\d).*?(\d+)\s*ngay\s*(?:ngay\s*|lien\s*)?truoc",
        q_noacc,
    )
    if m_prc_win2:
        n_days = int(m_prc_win2.group(1))
        d0 = dt.date(int(m_prc_win2.group(4)), int(m_prc_win2.group(3)), int(m_prc_win2.group(2)))
        w_cur = TimeWindow(
            start_date=d0,
            end_date=d0 + dt.timedelta(days=n_days - 1),
            label="price_change_post",
        )
        w_cur.compare_window = TimeWindow(
            start_date=d0 - dt.timedelta(days=n_days),
            end_date=d0 - dt.timedelta(days=1),
            label="price_change_pre",
        )
        return w_cur

    m_prc_win3 = re.search(
        r"(?:len\s*gia|doi\s*gia|tang\s*gia|giam\s*gia|gia\s*moi|dieu\s*chinh\s*gia).*?(\d+)\s*ngay\s*(?:dau|ke\s*tu|sau).*?(\d+)\s*ngay\s*(?:ngay\s*|lien\s*)?truoc",
        q_noacc,
    )
    if m_prc_win3:
        n_days = int(m_prc_win3.group(1))
        w_cur = TimeWindow(
            start_date=as_of_date,
            end_date=as_of_date,
            label=f"price_change_lookup_{n_days}",
        )
        w_cur.compare_window = TimeWindow(
            start_date=as_of_date,
            end_date=as_of_date,
            label=f"price_change_pre_{n_days}",
        )
        return w_cur

    # 2. Multi-month list:
    # e.g. "các tháng 10, 11, 12 năm 2025", "th5/2025, th6/2025, th7/2025", "3 tháng 9/2025 → 10/2025 → 11/2025"
    m_mm1 = re.search(
        r"(?:cac\s+thang|3\s+thang|thang|giua\s+thang|th)\s*(\d{1,2})(?:[/\-](202\d))?\s*(?:,|va|→|->)\s*(?:thang|th)?\s*(\d{1,2})(?:[/\-](202\d))?\s*(?:,|va|→|->)\s*(?:thang|th)?\s*(\d{1,2})\s*(?:[/\-]|nam\s*)(202\d)",
        q_noacc,
    )
    if m_mm1:
        y3 = int(m_mm1.group(6))
        y1 = int(m_mm1.group(2)) if m_mm1.group(2) else y3
        y2 = int(m_mm1.group(4)) if m_mm1.group(4) else y3
        m1, m2, m3 = int(m_mm1.group(1)), int(m_mm1.group(3)), int(m_mm1.group(5))
        wins = []
        for y_i, m_i in ((y1, m1), (y2, m2), (y3, m3)):
            s_i, e_i = month_bounds(y_i, m_i)
            wins.append(TimeWindow(s_i, min(e_i, as_of_date), label=f"{y_i:04d}-{m_i:02d}"))
        res = TimeWindow(wins[0].start_date, wins[-1].end_date, label="multi_months", multi_windows=wins)
        return res

    # "3 tháng liền nhau kết thúc ở tháng M/YYYY"
    m_mm2 = re.search(
        r"(\d+)\s*thang\s*(?:lien\s*nhau|lien\s*tiep)\s*ket\s*thuc\s*(?:o|vao|tai)?\s*(?:thang|th)?\s*(\d{1,2})\s*(?:[/\-]|nam\s*)(202\d)",
        q_noacc,
    )
    if m_mm2:
        n_m = int(m_mm2.group(1))
        cur_m = int(m_mm2.group(2))
        cur_y = int(m_mm2.group(3))
        ym_list = []
        for _ in range(min(n_m, 12)):
            ym_list.append((cur_y, cur_m))
            cur_y, cur_m = prev_month_ym(cur_y, cur_m)
        ym_list.reverse()
        wins = []
        for y_i, m_i in ym_list:
            s_i, e_i = month_bounds(y_i, m_i)
            wins.append(TimeWindow(s_i, min(e_i, as_of_date), label=f"{y_i:04d}-{m_i:02d}"))
        return TimeWindow(wins[0].start_date, wins[-1].end_date, label="multi_months", multi_windows=wins)

    # "từ tháng M1/Y1 qua tháng M2/Y2 rồi tháng M3/Y3" or "từ tháng M1/Y1 đến tháng M3/Y3"
    m_mm3 = re.search(
        r"tu\s*(?:thang|th)\s*(\d{1,2})(?:[/\-](202\d))?\s*(?:den|toi|qua)\s*(?:het\s*)?(?:thang|th)?\s*(\d{1,2})(?:[/\-](202\d))?(?:\s*roi\s*(?:thang|th)?\s*(\d{1,2}))?\s*(?:[/\-]|nam\s*)(202\d)",
        q_noacc,
    )
    if m_mm3:
        end_y = int(m_mm3.group(6))
        y1 = int(m_mm3.group(2)) if m_mm3.group(2) else end_y
        m1 = int(m_mm3.group(1))
        end_m = int(m_mm3.group(5)) if m_mm3.group(5) else int(m_mm3.group(3))
        wins = []
        cy, cm = y1, m1
        while (cy, cm) <= (end_y, end_m) and len(wins) < 12:
            s_i, e_i = month_bounds(cy, cm)
            wins.append(TimeWindow(s_i, min(e_i, as_of_date), label=f"{cy:04d}-{cm:02d}"))
            if cm == 12:
                cy, cm = cy + 1, 1
            else:
                cm += 1
        if len(wins) >= 2:
            return TimeWindow(wins[0].start_date, wins[-1].end_date, label="multi_months", multi_windows=wins)

    # Multi-quarter list ("các quý năm 2025", "từng quý năm 2025", "4 quý năm 2025", "quý 1, quý 2, quý 3, quý 4 năm 2025")
    m_mq = re.search(r"(?:cac\s+quy|tung\s+quy|4\s+quy|moi\s+quy).*?nam\s*(202\d)|nam\s*(202\d).*?(?:moi\s+quy|tung\s+quy|4\s+quy)", q_noacc)
    if m_mq:
        yr = int(m_mq.group(1) or m_mq.group(2))
        wins = []
        for q_i in (1, 2, 3, 4):
            s_i, e_i = quarter_bounds(yr, q_i)
            wins.append(TimeWindow(s_i, min(e_i, as_of_date), label=f"Q{q_i}/{yr}"))
        return TimeWindow(wins[0].start_date, wins[-1].end_date, label="multi_quarters", multi_windows=wins)

    # 3. Lunar Tet window ("dịp Tết 2026", "mùa Tết 2026", "Tết Nguyên đán 2026", "Tết âm lịch 2026")
    m_tet = re.search(r"\b(?:dip\s+|mua\s+)?(?:tet\s+nguyen\s+dan|tet\s+am\s+lich|tet)\s*(?:nam\s*)?(202\d)?\b", q_noacc)
    if m_tet and "mung" not in q_noacc and "29 tet" not in q_noacc and "30 tet" not in q_noacc:
        t_year = int(m_tet.group(1)) if m_tet.group(1) else as_of_date.year
        t_start, t_end, pre_start, pre_end = cal.get_tet_window(t_year)
        w = TimeWindow(t_start, min(t_end, as_of_date), label=f"tet_{t_year}", shift_filter=shift_filter, specific_weekday=specific_wday)
        if "truoc tet" in q_noacc or "ngay truoc" in q_noacc:
            w.compare_window = TimeWindow(pre_start, pre_end, label=f"pre_tet_{t_year}")
        return w

    # Specific Lunar Date ("mùng 1 Tết 2026", "rằm tháng 7 năm 2025", "mùng 1 tháng 8 âm lịch 2025")
    m_ram = re.search(r"\bram\s+thang\s+(\d{1,2})(?:\s+nam\s+(202\d)|\s*/\s*(202\d))?", q_noacc)
    if m_ram:
        lm = int(m_ram.group(1))
        ly = int(m_ram.group(2) or m_ram.group(3) or as_of_date.year)
        s_dt = cal.by_lunar.get((ly, lm, 15, False))
        if s_dt:
            return TimeWindow(s_dt, s_dt, label="lunar_day", shift_filter=shift_filter)

    m_mung_tet = re.search(r"\bmung\s+(\d{1,2})\s+tet(?:\s+(202\d))?", q_noacc)
    if m_mung_tet:
        ld = int(m_mung_tet.group(1))
        ly = int(m_mung_tet.group(2) or as_of_date.year)
        s_dt = cal.by_lunar.get((ly, 1, ld, False))
        if s_dt:
            return TimeWindow(s_dt, s_dt, label="lunar_tet_day", shift_filter=shift_filter)

    # 4. Explicit date range: "từ DD/MM/YYYY đến (hết) DD/MM/YYYY"
    m_rng = re.search(
        r"tu\s+(?:ngay\s+)?(\d{1,2})[/\-](\d{1,2})[/\-](202\d)\s+(?:den|toi)\s+(?:het\s+)?(?:ngay\s+)?(\d{1,2})[/\-](\d{1,2})[/\-](202\d)",
        q_noacc,
    )
    if m_rng:
        d1 = dt.date(int(m_rng.group(3)), int(m_rng.group(2)), int(m_rng.group(1)))
        d2 = dt.date(int(m_rng.group(6)), int(m_rng.group(5)), int(m_rng.group(4)))
        return TimeWindow(d1, d2, label="explicit_range", shift_filter=shift_filter, specific_weekday=specific_wday)

    # 5. Expiry window "hết hạn trong N ngày tới" (without explicit range)
    m_exp_n = re.search(r"het\s+han\s+trong\s+(\d+)\s+ngay\s+toi", q_noacc)
    if m_exp_n:
        n_d = int(m_exp_n.group(1))
        # Check if there is an explicit anchor date "tính tại ngày DD/MM/YYYY" or "theo tồn kho ngày DD/MM/YYYY"
        m_anc = re.search(r"(?:tai\s+ngay|ngay)\s+(\d{1,2})[/\-](\d{1,2})[/\-](202\d)", q_noacc)
        anc = dt.date(int(m_anc.group(3)), int(m_anc.group(2)), int(m_anc.group(1))) if m_anc else as_of_date
        return TimeWindow(anc, anc + dt.timedelta(days=n_d), label="expiry_window")

    # 6. Cumulative / Point-in-time explicit date ("tính đến hết ngày DD/MM/YYYY", "tại ngày DD/MM/YYYY", "cuối ngày DD/MM/YYYY", "hết ngày DD/MM/YYYY")
    m_cum = re.search(
        r"(?:tinh\s+den\s+het|den\s+het|toi\s+het|het|tai|cuoi|theo\s+ton\s+kho|den)\s*(?:ngay\s+)?(\d{1,2})[/\-](\d{1,2})[/\-](202\d)",
        q_noacc,
    )
    if m_cum:
        d_target = dt.date(int(m_cum.group(3)), int(m_cum.group(2)), int(m_cum.group(1)))
        start_d = data_start_date if data_start_date else dt.date(2025, 1, 1)
        return TimeWindow(start_d, d_target, label="cumulative_date", is_cumulative=True, shift_filter=shift_filter)

    # 7. Single explicit date ("ngày DD/MM/YYYY", "hôm DD/MM/YYYY")
    m_single_dt = re.search(r"(?:ngay|hom|vao)\s+(\d{1,2})[/\-](\d{1,2})[/\-](202\d)", q_noacc)
    if m_single_dt and "thang" not in q_noacc:
        d_target = dt.date(int(m_single_dt.group(3)), int(m_single_dt.group(2)), int(m_single_dt.group(1)))
        return TimeWindow(d_target, d_target, label="single_date", shift_filter=shift_filter)

    # 8. Explicit Quarter + Year ("quý Q/YYYY", "Q1/2026", "quý Q năm YYYY") — MUST run before Month + Year!
    m_qy = re.search(r"\b(?:quy\s*([1-4])\s*(?:[/\-]|nam\s+)(202\d)|q([1-4])[/\-](202\d))\b", q_noacc)
    if m_qy:
        q_val = int(m_qy.group(1) or m_qy.group(3))
        y_val = int(m_qy.group(2) or m_qy.group(4))
        s_dt, e_dt = quarter_bounds(y_val, q_val)
        if s_dt <= as_of_date <= e_dt:
            e_dt = as_of_date
        w = TimeWindow(s_dt, e_dt, label=f"Q{q_val}/{y_val}", shift_filter=shift_filter, specific_weekday=specific_wday)
        if any(k in q_noacc for k in ("quy truoc", "quy lien truoc", "so voi quy")):
            pq_y, pq_q = prev_quarter_yq(y_val, q_val)
            ps, pe = quarter_bounds(pq_y, pq_q)
            w.compare_window = TimeWindow(ps, pe, label=f"Q{pq_q}/{pq_y}")
        return w

    # 9. Explicit Month + Year ("tháng M/YYYY", "thM/YYYY", "M-YYYY", "06/2026", "tháng M năm YYYY", "th M/YY")
    # Note: check comparison patterns like "tháng M/YYYY so với tháng liền trước" or two explicit months
    all_my = list(
        re.finditer(
            r"(?:thang|th|t)\s*(\d{1,2})\s*(?:[/\-]|nam\s+)(202\d|\d{2})\b|\b(0?[1-9]|1[0-2])[/\-](202\d)\b",
            q_noacc,
        )
    )
    if all_my:
        def _extract_my(m_obj: re.Match) -> Tuple[int, int]:
            if m_obj.group(1):
                mv = int(m_obj.group(1))
                yv = int(m_obj.group(2))
                if yv < 100:
                    yv += 2000
                return mv, yv
            return int(m_obj.group(3)), int(m_obj.group(4))

        m_val, y_val = _extract_my(all_my[0])
        if 1 <= m_val <= 12:
            # Check if two distinct explicit months are compared ("tháng 3 năm 2026 ... so với tháng 3 năm 2025")
            if len(all_my) >= 2:
                m2_val, y2_val = _extract_my(all_my[1])
                if (m2_val, y2_val) != (m_val, y_val) and 1 <= m2_val <= 12:
                    # If 'so voi' appears before the first month, the second month is primary and first is compare
                    prefix_before_first = q_noacc[: all_my[0].start()]
                    if "so voi" in prefix_before_first or "so vs" in prefix_before_first:
                        m_val, y_val, m2_val, y2_val = m2_val, y2_val, m_val, y_val
                    s_dt, e_dt = month_bounds(y_val, m_val)
                    if (y_val, m_val) == (as_of_date.year, as_of_date.month):
                        e_dt = min(e_dt, as_of_date)
                    ps, pe = month_bounds(y2_val, m2_val)
                    if (y2_val, m2_val) == (as_of_date.year, as_of_date.month):
                        pe = min(pe, as_of_date)
                    w = TimeWindow(s_dt, e_dt, label=f"{y_val:04d}-{m_val:02d}", shift_filter=shift_filter, specific_weekday=specific_wday)
                    w.compare_window = TimeWindow(ps, pe, label=f"{y2_val:04d}-{m2_val:02d}")
                    return w

            s_dt, e_dt = month_bounds(y_val, m_val)
            if (y_val, m_val) == (as_of_date.year, as_of_date.month):
                e_dt = min(e_dt, as_of_date)
            w = TimeWindow(s_dt, e_dt, label=f"{y_val:04d}-{m_val:02d}", shift_filter=shift_filter, specific_weekday=specific_wday)
            if any(k in q_noacc for k in ("thang lien truoc", "thang truoc do", "thang truoc", "thang roi", "th truoc", "th roi", "so voi thang", "so vs thang", "so vs th")):
                py, pm = prev_month_ym(y_val, m_val)
                ps, pe = month_bounds(py, pm)
                w.compare_window = TimeWindow(ps, pe, label=f"{py:04d}-{pm:02d}")
            elif "cung ky nam truoc" in q_noacc or "nam ngoai" in q_noacc or "nam truoc" in q_noacc:
                ps, pe = month_bounds(y_val - 1, m_val)
                w.compare_window = TimeWindow(ps, pe, label=f"{y_val-1:04d}-{m_val:02d}")
            elif has_both_shifts:
                w.shift_filter = "afternoon"
                w.compare_window = TimeWindow(s_dt, e_dt, label="morning_shift", shift_filter="morning")
            return w

    # 10. Last N days ("N ngày gần nhất", "N ngày qua", "N ngày vừa qua")
    m_ndays = re.search(r"(\d+)\s*ngay\s*(?:gan\s*nhat|qua|vua\s*qua|gan\s*day)", q_noacc)
    if m_ndays:
        n_d = int(m_ndays.group(1))
        return TimeWindow(
            as_of_date - dt.timedelta(days=n_d - 1),
            as_of_date,
            label=f"last_{n_d}_days",
            shift_filter=shift_filter,
            specific_weekday=specific_wday,
        )

    # 11. Relative Month ("tháng này", "tháng ni", "th này", "đầu tháng", "tháng trước", "tháng rồi", "từ 01/M đến hết hôm nay")
    m_mtd = re.search(r"tu\s+(?:ngay\s+)?0?1[/\-]0?(\d+)\s+(?:den|toi)\s+(?:het\s+)?(?:ngay\s+)?(?:hom\s+nay|hnay|bua\s+nay|bua\s+ni)", q_noacc)
    is_mtd_from_first = bool(m_mtd and int(m_mtd.group(1)) == as_of_date.month and "nam nay" not in q_noacc)
    if is_mtd_from_first or ("cung ky" in q_noacc and any(k in q_noacc for k in ("thang", "th "))) or any(k in q_noacc for k in ("thang nay", "thang ni", "th nay", "dau thang nay", "dau thang den")):
        s_dt = dt.date(as_of_date.year, as_of_date.month, 1)
        w = TimeWindow(s_dt, as_of_date, label="this_month", shift_filter=shift_filter, specific_weekday=specific_wday)
        if "cung ky" in q_noacc:
            py, pm = prev_month_ym(as_of_date.year, as_of_date.month)
            p_end_day = min(as_of_date.day, last_day_of_month(py, pm))
            w.compare_window = TimeWindow(dt.date(py, pm, 1), dt.date(py, pm, p_end_day), label="same_period_last_month")
        elif any(k in q_noacc for k in ("thang truoc", "thang roi", "thang lien truoc", "th truoc", "th roi")):
            py, pm = prev_month_ym(as_of_date.year, as_of_date.month)
            ps, pe = month_bounds(py, pm)
            w.compare_window = TimeWindow(ps, pe, label="last_month")
        elif has_both_shifts:
            w.shift_filter = "afternoon"
            w.compare_window = TimeWindow(s_dt, as_of_date, label="morning_shift", shift_filter="morning")
        return w

    if any(k in q_noacc for k in ("thang truoc", "thang roi", "thang vua roi")):
        py, pm = prev_month_ym(as_of_date.year, as_of_date.month)
        ps, pe = month_bounds(py, pm)
        w = TimeWindow(ps, pe, label="last_month", shift_filter=shift_filter, specific_weekday=specific_wday)
        if has_both_shifts:
            w.shift_filter = "afternoon"
            w.compare_window = TimeWindow(ps, pe, label="morning_shift", shift_filter="morning")
        return w

    # 12. Relative Week ("tuần này", "tuần trước", "tuần rồi")
    if "tuan nay" in q_noacc or "tuan ni" in q_noacc:
        mon = as_of_date - dt.timedelta(days=as_of_date.weekday())
        w = TimeWindow(mon, as_of_date, label="this_week", shift_filter=shift_filter, specific_weekday=specific_wday)
        if "tuan truoc" in q_noacc or "tuan roi" in q_noacc:
            p_mon = mon - dt.timedelta(days=7)
            p_sun = mon - dt.timedelta(days=1)
            w.compare_window = TimeWindow(p_mon, p_sun, label="last_week")
        elif has_both_shifts:
            w.shift_filter = "afternoon"
            w.compare_window = TimeWindow(mon, as_of_date, label="morning_shift", shift_filter="morning")
        return w

    if any(k in q_noacc for k in ("tuan truoc", "tuan roi", "tuan vua roi")):
        mon_this = as_of_date - dt.timedelta(days=as_of_date.weekday())
        p_mon = mon_this - dt.timedelta(days=7)
        p_sun = mon_this - dt.timedelta(days=1)
        w = TimeWindow(p_mon, p_sun, label="last_week", shift_filter=shift_filter, specific_weekday=specific_wday)
        if has_both_shifts:
            w.shift_filter = "afternoon"
            w.compare_window = TimeWindow(p_mon, p_sun, label="morning_shift", shift_filter="morning")
        return w

    # 13. Relative Quarter ("quý này", "quý trước", "quý rồi")
    cur_q = (as_of_date.month - 1) // 3 + 1
    if "quy nay" in q_noacc or "quy ni" in q_noacc:
        qs, _ = quarter_bounds(as_of_date.year, cur_q)
        w = TimeWindow(qs, as_of_date, label="this_quarter", shift_filter=shift_filter, specific_weekday=specific_wday)
        if "quy truoc" in q_noacc or "quy roi" in q_noacc:
            pq_y, pq_q = prev_quarter_yq(as_of_date.year, cur_q)
            ps, pe = quarter_bounds(pq_y, pq_q)
            w.compare_window = TimeWindow(ps, pe, label="last_quarter")
        return w

    if "quy truoc" in q_noacc or "quy roi" in q_noacc:
        pq_y, pq_q = prev_quarter_yq(as_of_date.year, cur_q)
        ps, pe = quarter_bounds(pq_y, pq_q)
        return TimeWindow(ps, pe, label="last_quarter", shift_filter=shift_filter, specific_weekday=specific_wday)

    # 14. Relative Year / YTD ("năm nay", "từ đầu năm", "từ 1/1 đến hết hôm nay", "năm ngoái")
    if any(k in q_noacc for k in ("nam nay", "tu dau nam", "tu 1/1", "tu 01/01")):
        ys = dt.date(as_of_date.year, 1, 1)
        return TimeWindow(ys, as_of_date, label="this_year", shift_filter=shift_filter, specific_weekday=specific_wday)

    if "nam ngoai" in q_noacc or "nam truoc" in q_noacc:
        py = as_of_date.year - 1
        return TimeWindow(dt.date(py, 1, 1), dt.date(py, 12, 31), label="last_year", shift_filter=shift_filter, specific_weekday=specific_wday)

    # 15. Explicit Year ("năm 2025", "trong năm 2026")
    m_yr = re.search(r"\bnam\s+(202\d)\b", q_noacc)
    if m_yr:
        yr = int(m_yr.group(1))
        s_dt = dt.date(yr, 1, 1)
        e_dt = min(dt.date(yr, 12, 31), as_of_date) if yr == as_of_date.year else dt.date(yr, 12, 31)
        return TimeWindow(s_dt, e_dt, label=f"year_{yr}", shift_filter=shift_filter, specific_weekday=specific_wday)

    # 16. Relative Days ("hôm qua", "hôm kia", "hôm nay")
    if any(k in q_noacc for k in ("hom qua", "hqua", "bua qua")) and not any(k in q_noacc for k in ("hom nay", "bua nay", "hnay", "bua ni")):
        yd = as_of_date - dt.timedelta(days=1)
        return TimeWindow(yd, yd, label="yesterday", shift_filter=shift_filter)

    if "hom kia" in q_noacc:
        d2 = as_of_date - dt.timedelta(days=2)
        return TimeWindow(d2, d2, label="day_before_yesterday", shift_filter=shift_filter)

    # Cumulative up to today ("tính đến hết hôm nay", "đến hết hôm nay", "tới hôm nay", "hết hôm nay", "hết bữa ni")
    if any(k in q_noacc for k in ("den het hom nay", "tinh den hom nay", "toi hom nay", "het hom nay", "het hnay", "het ngay hom nay", "toi hnay", "toi bua ni", "cuoi ngay hom nay", "cuoi hom nay", "cuoi bua nay", "het bua nay")):
        start_d = data_start_date if data_start_date else dt.date(2025, 1, 1)
        return TimeWindow(start_d, as_of_date, label="cumulative_today", is_cumulative=True, shift_filter=shift_filter)

    if any(k in q_noacc for k in ("hom nay", "bua nay", "hnay", "bua ni")):
        w = TimeWindow(as_of_date, as_of_date, label="today", shift_filter=shift_filter)
        if any(k in q_noacc for k in ("hom qua", "hqua", "bua qua")):
            yd = as_of_date - dt.timedelta(days=1)
            w.compare_window = TimeWindow(yd, yd, label="yesterday")
        return w

    # Default fallback: full dataset history up to as_of_date
    start_d = data_start_date if data_start_date else dt.date(2025, 1, 1)
    return TimeWindow(start_d, as_of_date, label="all_time_to_as_of", is_cumulative=True, shift_filter=shift_filter, specific_weekday=specific_wday)
