"""
Time resolution done in Python, never by the LLM.

- time_context(as_of)       : dict of named ranges (this_week, last_month, ...) relative to as_of
- Calendar (calendar.csv)   : solar<->lunar lookups, Tết window (23 tháng Chạp -> mùng 10)
- normalize_question(text)  : expand Vietnamese chat abbreviations (sp, dc, ko, ...)

calendar.csv format is not documented in the task text. Loader guesses columns; run
`python -m dm.timeutil path/to/calendar.csv` once on the real file and check the printout.
"""
from __future__ import annotations

import calendar as _cal
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


def _d(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    if isinstance(x, datetime):
        return x.date()
    return datetime.strptime(str(x)[:10], "%Y-%m-%d").date()


def shift_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)


def shift_months(d: date, months: int) -> date:
    m = d.month - 1 + months
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, _cal.monthrange(y, m)[1]))


def month_range(y: int, m: int) -> tuple[date, date]:
    return date(y, m, 1), date(y, m, _cal.monthrange(y, m)[1])


def time_context(as_of: date | str) -> dict[str, tuple[str, str]]:
    """Named inclusive ranges. Rulebook defaults: week = Mon-Sun; 'this month' = 1st..as_of."""
    a = _d(as_of)
    mon = a - timedelta(days=a.weekday())
    out: dict[str, tuple[date, date]] = {}
    out["today"] = (a, a)
    out["yesterday"] = (a - timedelta(days=1),) * 2
    out["this_week_to_date"] = (mon, a)
    out["last_week"] = (mon - timedelta(days=7), mon - timedelta(days=1))
    out["this_month_to_date"] = (a.replace(day=1), a)
    lm_start = shift_months(a.replace(day=1), -1)
    out["last_month"] = month_range(lm_start.year, lm_start.month)
    q0 = (a.month - 1) // 3 * 3 + 1
    qs = date(a.year, q0, 1)
    out["this_quarter_to_date"] = (qs, a)
    pq = shift_months(qs, -3)
    out["last_quarter"] = (pq, qs - timedelta(days=1))
    out["this_year_to_date"] = (date(a.year, 1, 1), a)
    out["last_year"] = (date(a.year - 1, 1, 1), date(a.year - 1, 12, 31))
    out["last_7_days"] = (a - timedelta(days=6), a)
    out["last_30_days"] = (a - timedelta(days=29), a)
    out["same_period_last_year__this_month_to_date"] = (shift_years(a.replace(day=1), -1), shift_years(a, -1))
    out["same_week_last_year"] = (shift_years(mon, -1), shift_years(a, -1))
    out["previous_month_same_days"] = (shift_months(a.replace(day=1), -1), shift_months(a, -1))
    out["this_month_last_year_full"] = month_range(a.year - 1, a.month)
    return {k: (v[0].isoformat(), v[1].isoformat()) for k, v in out.items()}


def time_context_text(as_of: date | str) -> str:
    a = _d(as_of)
    wd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][a.weekday()]
    lines = [f"as_of_date = {a.isoformat()} ({wd}). Pre-resolved ranges (inclusive, use as-is):"]
    lines += [f"- {k}: {s} .. {e}" for k, (s, e) in time_context(a).items()]
    lines.append("For 'tháng 3', 'quý 2', 'năm 2025' etc. build the range yourself; if no year is given, "
                 "use the most recent such period not after as_of_date.")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Lunar calendar
# ----------------------------------------------------------------------------
@dataclass
class Calendar:
    df: pd.DataFrame            # columns: solar (date), lm (int), ld (int), leap (bool)

    @classmethod
    def load(cls, path: str | Path | None) -> Optional["Calendar"]:
        if not path or not Path(path).exists():
            return None
        raw = pd.read_csv(path)
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        cols = list(raw.columns)

        def find(*pats, exclude=()):
            for c in cols:
                if any(re.search(p, c) for p in pats) and not any(re.search(e, c) for e in exclude):
                    return c
            return None

        solar = find(r"solar", r"duong", r"gregorian", r"^date$", r"^ngay$")
        lm = find(r"lunar.*month", r"am.*thang", r"lunar_m", r"^lm$", r"am_lich_thang", r"thang_am")
        ld = find(r"lunar.*day", r"am.*ngay", r"lunar_d", r"^ld$", r"ngay_am")
        ly = find(r"lunar.*year", r"am.*nam", r"lunar_y", r"^ly$", r"nam_am")
        leap = find(r"leap", r"nhuan")
        holiday = find(r"holiday", r"ngay_le", r"su_kien", r"^le$")
        if solar is None:  # fall back: first column that parses as date
            for c in cols:
                if pd.to_datetime(raw[c], errors="coerce").notna().mean() > 0.9:
                    solar = c
                    break
        if not (solar and lm and ld):
            print(f"[calendar] cannot recognize columns in {cols}; Tết tools disabled. Edit Calendar.load().")
            return None
        df = pd.DataFrame({
            "solar": pd.to_datetime(raw[solar], errors="coerce").dt.date,
            "lm": pd.to_numeric(raw[lm], errors="coerce"),
            "ld": pd.to_numeric(raw[ld], errors="coerce"),
            "ly": pd.to_numeric(raw[ly], errors="coerce") if ly else pd.NA,
            "leap": raw[leap].astype(str).str.lower().isin(["1", "true", "yes", "y", "t"]) if leap else False,
            "holiday": raw[holiday].fillna("").astype(str).str.strip() if holiday else "",
        }).dropna(subset=["solar", "lm", "ld"])
        df["lm"], df["ld"] = df["lm"].astype(int), df["ld"].astype(int)
        return cls(df.sort_values("solar").reset_index(drop=True))

    def solar_from_lunar(self, solar_year: int, lunar_month: int, lunar_day: int) -> Optional[date]:
        """Solar date of (lunar month, day) in the lunar year whose Tết falls in solar_year
        (months 11-12 resolve to the end of the previous solar year)."""
        d = self.df[(~self.df["leap"]) & (self.df["lm"] == lunar_month) & (self.df["ld"] == lunar_day)]
        if "ly" in self.df.columns and self.df["ly"].notna().any():
            target_ly = solar_year - 1 if lunar_month >= 11 else solar_year
            d_ly = d[d["ly"] == target_ly]
            if len(d_ly):
                return d_ly["solar"].iloc[-1]
        if lunar_month >= 11:
            # months 11/12 of the lunar year ending at Tết `solar_year`: the match just before that Tết
            tet = self._tet_day1(solar_year)
            if tet is None:
                return None
            d = d[(d["solar"] < tet) & (d["solar"] >= tet - timedelta(days=120))]
        else:
            d = d[d["solar"].map(lambda x: x.year) == solar_year]
        return d["solar"].iloc[-1] if len(d) else None

    def _tet_day1(self, year: int) -> Optional[date]:
        d = self.df[(~self.df["leap"]) & (self.df["lm"] == 1) & (self.df["ld"] == 1)
                    & (self.df["solar"].map(lambda x: x.year) == year)]
        return d["solar"].iloc[0] if len(d) else None

    def tet_window(self, year: int) -> Optional[tuple[date, date]]:
        """23 tháng Chạp (previous lunar year) .. mùng 10 tháng Giêng of Tết `year`."""
        s = self.solar_from_lunar(year, 12, 23)
        e = self.solar_from_lunar(year, 1, 10)
        return (s, e) if s and e else None

    def solar_to_lunar(self, d: date) -> Optional[tuple[int, int, bool]]:
        r = self.df[self.df["solar"] == d]
        if r.empty:
            return None
        x = r.iloc[0]
        return int(x["lm"]), int(x["ld"]), bool(x["leap"])

    def years(self) -> list[int]:
        return sorted({d.year for d in self.df["solar"]})


# ----------------------------------------------------------------------------
# Question normalization (light; the LLM still sees the original text)
# ----------------------------------------------------------------------------
_ABBR = {
    "sp": "sản phẩm", "dc": "được", "đc": "được", "ko": "không", "k": "không", "hok": "không", "hông": "không",
    "kg": "kg", "thg": "tháng", "th": "tháng", "tg": "thời gian", "nv": "nhân viên", "kh": "khách hàng",
    "dt": "doanh thu", "ln": "lợi nhuận", "sl": "số lượng", "tt": "thanh toán", "hd": "hóa đơn", "hđ": "hóa đơn",
    "cn": "chi nhánh", "vs": "so với", "nhiu": "nhiều", "bn": "bao nhiêu", "j": "gì", "gi": "gì", "ntn": "như thế nào",
    "nay": "nay", "rồi": "rồi", "trc": "trước", "ck": "cùng kỳ",
}


def normalize_question(q: str) -> str:
    return re.sub(r"\b(\w+)\b", lambda m: _ABBR.get(m.group(1).lower(), m.group(1)), q, flags=re.UNICODE)


if __name__ == "__main__":
    import sys
    print(time_context_text("2026-03-04"))
    if len(sys.argv) > 1:
        cal = Calendar.load(sys.argv[1])
        if cal:
            print("years:", cal.years()[:3], "...", cal.years()[-3:])
            for y in cal.years():
                w = cal.tet_window(y)
                if w:
                    print(y, "Tết window:", w)
