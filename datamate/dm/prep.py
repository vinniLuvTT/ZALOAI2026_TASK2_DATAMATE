"""
Data preparation: load -> profile -> roles -> joins -> flat `base` -> clean views.

Output per business (inside work_dir/<biz>/):
  cube.duckdb : raw tables + `base` + views `lines_all`, `lines_valid`, `orders_valid`,
                plus optional `inventory_now`, `debts_open`, `cost_view`
                (opened by the Cube server)
  sql.duckdb  : byte copy of the same file (opened read-only by the SQL tool, so
                Cube and the agent never fight over one file lock)

Views
  lines_all    every base row with standardized columns + flags
                 dt, amount, qty, order_id, product, product_id, customer, branch,
                 unit_cost, is_cancelled, is_returned_order, is_service_fee, + all original columns
  lines_valid  lines_all WHERE NOT is_cancelled AND NOT is_returned_order AND NOT is_non_item AND dt IS NOT NULL
               (rulebook default for revenue / quantity / "bán chạy")
  orders_valid one row per valid order

Only heuristics here (no LLM needed). An optional `llm` callable can refine roles;
its output is accepted only if it passes validate_roles().
"""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Optional

import duckdb
import pandas as pd


# ----------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------
def unaccent(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def norm_name(s: str) -> str:
    s = unaccent(str(s)).strip()
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    if not s:
        return "col"
    if s[0].isdigit():
        s = "c_" + s
    return s


def dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def qid(c: str) -> str:
    return '"' + c.replace('"', '""') + '"'


def qlit(v) -> str:
    return "'" + str(v).replace("'", "''") + "'"


# ----------------------------------------------------------------------------
# 1. Loading & Cleaning
# ----------------------------------------------------------------------------
def _is_header_cell(v) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    if not s or len(s) > 60:
        return False
    u = unaccent(s).lower()
    if re.fullmatch(r"-?[\d.,\s]+(d|vnd|k|tr|trieu|%)?", u):
        return False
    if re.fullmatch(r"\d{1,4}[-/.]\d{1,2}([-/.]\d{1,4})?(\s+\d{1,2}:\d{2}(:\d{2})?)?", s):
        return False
    if re.search(r"\d+\s+thg\s+\d+", u):
        return False
    if re.fullmatch(r"[a-z]{1,5}[-_]?\d{2,10}", u):
        return False
    return True


def _detect_header_row(raw: pd.DataFrame, max_scan: int = 15) -> int:
    best_i, best_nn = 0, 0
    limit = min(max_scan, max(len(raw), 1))
    for i in range(limit):
        row = raw.iloc[i]
        vals = [v for v in row if pd.notna(v) and str(v).strip() != ""]
        nn = len(vals)
        if nn < 2:
            continue
        hdr_ratio = sum(_is_header_cell(v) for v in vals) / nn
        if hdr_ratio >= 0.75 and nn > best_nn:
            best_i, best_nn = i, nn
    return best_i


_SUMMARY_LABEL_PAT = re.compile(
    r"^(tổng(\s+(cộng|ngày|tháng|năm|tuần|quý|kỳ|kết|số|đơn|hàng|tiền))?|"
    r"tong\s+(cong|ngay|thang|nam|tuan|quy|ky)|"
    r"cộng(\s+(ngày|tháng|năm|cuối|kỳ))?|"
    r"cong\s+(ngay|thang|nam)|"
    r"ghi\s+chú\s*:|ghi\s+chu\s*:|lưu\s+ý\s*:|luu\s+y\s*:|"
    r"total|subtotal|grand\s+total)\b",
    re.I,
)


def _drop_summary_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop embedded subtotal/total/footer rows (e.g. 'Tổng ngày 01/04', 'Tổng tháng', 'Ghi chú: ...') before ffill."""
    if df.shape[1] < 3 or df.empty:
        return df
    keep = []
    for idx, row in df.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() != ""]
        if len(vals) <= 1 or (len(vals) <= 3 and any(_SUMMARY_LABEL_PAT.search(v) for v in vals)):
            keep.append(False)
        else:
            keep.append(True)
    if not all(keep):
        df = df.loc[keep].reset_index(drop=True)
    return df


_SKIP_NUM_COL = re.compile(
    r"(^|_)(id|ma|code|sku|barcode|ma_vach|imei|sdt|so_dien_thoai|dien_thoai|phone|tel|mst|ma_so_thue|"
    r"so_hd|so_ct|so_lo|lo_sx|ngay|date|time|thoi_gian|gio|thang|nam|ky_bao_cao|ten|name|dia_chi|addr|"
    r"ghi_chu|note|trang_thai|status|ket_qua|loai|nhom|danh_muc|don_vi|dvt|unit|kenh|ca|khu_vuc|"
    r"hoat_chat|size|so_size|quy_doi|han_dung|han_su_dung|hsd)($|_)|^(ma_|id_|so_hd|so_ct|sdt|hang$|ban$|phong_ban$)"
)
_QTY_UNIT_COL = re.compile(r"so_luong|(^|_)sl($|_)|qty|quantity|ton")


def _parse_vn_num(v, allow_unit: bool = False) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return None if pd.isna(v) else float(v)
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", "-", "—"):
        return None
    # Reject zero-padded codes or phone numbers like 0901234567, 0012
    if re.fullmatch(r"0\d+", s):
        return None
    u = unaccent(s).lower().strip()
    u = re.sub(r"\b(vnd|dong)\b|[đ₫]", "", u).strip()
    u = re.sub(r"(?<=\d)\s*d$", "", u).strip()
    if not u:
        return None
    # 1tr440, 16,56tr, 2tr5
    m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)\s*(?:tr|trieu)(?:\s*(\d{1,4}))?", u)
    if m:
        head = float(m.group(1).replace(",", "."))
        tail = m.group(2)
        if tail:
            sign = -1.0 if head < 0 or m.group(1).startswith("-") else 1.0
            return sign * (abs(head) + int(tail) / (10 ** len(tail))) * 1_000_000.0
        return head * 1_000_000.0
    # 69k, 3950k, 12.5k
    m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)\s*(?:k|nghin|ngan)", u)
    if m:
        return float(m.group(1).replace(",", ".")) * 1_000.0
    # Quantity with trailing unit word (e.g. '3 chai', '2 thùng', '10 viên')
    if allow_unit:
        m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)\s+[a-z][a-z0-9]*", u)
        if m:
            return float(m.group(1).replace(",", "."))
    # Remove spaces used as thousand separators (e.g. '129 000')
    u_ns = re.sub(r"(?<=\d)\s+(?=\d{3}(\b|\D))", "", u)
    # Dot thousands: 55.000 or 1.480.338 or -53.500
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+(,\d+)?", u_ns):
        return float(u_ns.replace(".", "").replace(",", "."))
    # Comma thousands: 55,000 or 115,000.0000
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", u_ns):
        return float(u_ns.replace(",", ""))
    # Plain number with optional decimal
    if re.fullmatch(r"-?\d+(,\d{1,2})?", u_ns):
        return float(u_ns.replace(",", "."))
    if re.fullmatch(r"-?\d+(\.\d+)?", u_ns):
        return float(u_ns)
    return None


def _fix_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Vietnamese formatted number/currency/quantity strings into float."""
    for c in df.columns:
        if df[c].dtype != object:
            continue
        allow_u = bool(_QTY_UNIT_COL.search(c))
        if _SKIP_NUM_COL.search(c) and not allow_u:
            continue
        nn = df[c].dropna()
        if len(nn) < 3:
            continue
        sample = nn.iloc[:500]
        parsed_sample = [_parse_vn_num(v, allow_unit=allow_u) for v in sample]
        rate = sum(x is not None for x in parsed_sample) / len(sample)
        if rate >= 0.85:
            df[c] = df[c].map(lambda v: _parse_vn_num(v, allow_unit=allow_u))
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


_DATE_COL_PAT = re.compile(
    r"(^|_)(ngay|date|thoi_gian|time|created|han_dung|han_su_dung|het_han|han_tra|ngay_het_han|expiry|exp|hsd|sinh_nhat)($|_)"
)


def _parse_vn_date(v, hint_year: Optional[int] = None) -> Optional[str]:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        if pd.isna(v):
            return None
        if v.hour or v.minute or v.second:
            return v.strftime("%Y-%m-%d %H:%M:%S")
        return v.strftime("%Y-%m-%d")
    if isinstance(v, date) and not isinstance(v, datetime):
        return v.isoformat()
    # Excel serial date number (e.g. 45658 -> 2025-01-01)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        fv = float(v)
        if 30000 <= fv <= 60000:
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(fv, unit="D")
            return ts.strftime("%Y-%m-%d")
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "nat", "none", "null", "-", "—"):
        return None
    # Numeric string Excel serial date
    if re.fullmatch(r"\d{5}(\.0+)?", s):
        fv = float(s)
        if 30000 <= fv <= 60000:
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(fv, unit="D")
            return ts.strftime("%Y-%m-%d")
    # YYYY-MM-DD or YYYY/MM/DD (with optional time)
    m = re.fullmatch(r"(19\d{2}|20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?:\s+(\d{1,2}:\d{2}(?::\d{2})?))?", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            tpart = f" {m.group(4)}" if m.group(4) and m.group(4) not in ("00:00", "00:00:00") else ""
            return f"{y:04d}-{mo:02d}-{d:02d}{tpart}"
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY or DD/MM/YY
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](19\d{2}|20\d{2}|\d{2})(?:\s+(\d{1,2}:\d{2}(?::\d{2})?))?", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        if mo > 12 and d <= 12:
            d, mo = mo, d
        if 1 <= mo <= 12 and 1 <= d <= 31:
            tpart = f" {m.group(4)}" if m.group(4) and m.group(4) not in ("00:00", "00:00:00") else ""
            return f"{y:04d}-{mo:02d}-{d:02d}{tpart}"
    # Vietnamese '1 thg 6 2025' or 'Ngày 01 tháng 02 năm 2025'
    u = unaccent(s).lower()
    m = re.search(r"(\d{1,2})\s*(?:thg|thang)\s*(\d{1,2})(?:[\s,/-]*(?:nam)?\s*(20\d{2}|\d{2}))?", u)
    if m:
        d, mo = int(m.group(1)), int(m.group(2))
        y = int(m.group(3)) if m.group(3) else hint_year
        if y is not None:
            if y < 100:
                y += 2000
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{y:04d}-{mo:02d}-{d:02d}"
    # Yearless '1/1' or '15/06' when hint_year is known from sheet/file name (Rulebook §8)
    if hint_year is not None:
        m = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})(?:\s+(\d{1,2}:\d{2}(?::\d{2})?))?", s)
        if m:
            d, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                return f"{hint_year:04d}-{mo:02d}-{d:02d}"
    return None


def _fix_dates(df: pd.DataFrame, hint_year: Optional[int] = None) -> pd.DataFrame:
    for c in df.columns:
        is_dt_dtype = pd.api.types.is_datetime64_any_dtype(df[c])
        if not is_dt_dtype and not _DATE_COL_PAT.search(c):
            continue
        nn = df[c].dropna()
        if len(nn) == 0:
            continue
        sample = nn.iloc[:400]
        parsed_sample = [_parse_vn_date(v, hint_year) for v in sample]
        rate = sum(x is not None for x in parsed_sample) / len(sample)
        if rate >= 0.75:
            df[c] = df[c].map(lambda v: _parse_vn_date(v, hint_year))
    return df


_ORDER_ID_COL_PAT = re.compile(
    r"^(ma_don|ma_don_hang|order_id|so_hd|ma_hd|ma_hoa_don|hoa_don|hd|ma_phieu|so_phieu|ma_giao_dich|ma_chung_tu|so_chung_tu|so_ct|ma_bill|stt)$"
)
_FFILL_HEADER_PAT = re.compile(
    r"^(ma_don|ma_don_hang|order_id|so_hd|ma_hd|ma_hoa_don|hoa_don|hd|ma_phieu|so_phieu|ma_giao_dich|ma_chung_tu|so_chung_tu|so_ct|ma_bill|stt|"
    r"ngay|ngay_ban|ngay_dat|ngay_dat_hang|ngay_hd|ngay_lap|ngay_giao_dich|ngay_chung_tu|thoi_gian|gio|"
    r"trang_thai|trang_thai_don|trang_thai_don_hang|trang_thai_xu_ly|status|tinh_trang|ket_qua|ghi_chu_tt|ghi_chu|"
    r"ten_khach|ten_khach_hang|khach_hang|khach_hang_ten|khach|ten_kh|ma_kh|ma_khach|doi_tac|ben_mua|"
    r"kenh|kenh_ban|chi_nhanh|cua_hang)$"
)
_ITEM_INDICATOR_PAT = re.compile(
    r"ten_san_pham|ten_sp|ten_hang|ten_mon|ten_thuoc|san_pham|product|mat_hang|hang_hoa|(^|_)hang($|_)|noi_dung|"
    r"ma_sp|ma_san_pham|sku|ma_hang|ma_thuoc|ma_mon|thanh_tien|tri_gia|(^|_)tt($|_)|tong_tien|amount|so_luong|(^|_)sl($|_)"
)


def _fix_continuation_and_dupes(df: pd.DataFrame) -> pd.DataFrame:
    """Rulebook §8:
    1. Continuation lines of multi-item orders leave order_id / date / status / customer blank -> ffill header columns.
    2. Exact duplicate rows across all cells within the same order -> drop_duplicates.
    """
    if df.empty or len(df) < 2:
        return df
    oid_cols = [c for c in df.columns if _ORDER_ID_COL_PAT.search(c)]
    item_cols = [c for c in df.columns if _ITEM_INDICATOR_PAT.search(c) and c not in oid_cols]
    if oid_cols and item_cols:
        oid = oid_cols[0]
        has_item = df[item_cols].notna().any(axis=1)
        oid_null_rate = float(df.loc[has_item, oid].isna().mean()) if has_item.any() else 0.0
        # Case A: order_id itself is blank on continuation lines (e.g. biz_103, biz_121, biz_122, dev_004, dev_005, dev_011)
        if 0.02 <= oid_null_rate <= 0.90 and pd.notna(df[oid].iloc[0]):
            dt_ff = [c for c in df.columns if c == oid or re.search(r"^(ngay|thoi_gian|gio)", c)]
            df.loc[has_item, dt_ff] = df.loc[has_item, dt_ff].ffill()
            other_ff = [c for c in df.columns if _FFILL_HEADER_PAT.search(c) and c not in dt_ff]
            if other_ff:
                df.loc[has_item, other_ff] = df.loc[has_item].groupby(oid, sort=False)[other_ff].ffill()
        # Case B: order_id is present on every line, but date/customer/channel are blank on continuation/voucher lines (e.g. dev_013)
        elif oid_null_rate < 0.02:
            ff_cols = [c for c in df.columns if _FFILL_HEADER_PAT.search(c) and c != oid and c != "ghi_chu"]
            for c in ff_cols:
                nr = float(df[c].isna().mean())
                if 0.02 <= nr <= 0.85:
                    df[c] = df.groupby(oid, sort=False)[c].ffill().bfill()
        # Format integer-valued float STT / order_id as clean integer string
        if oid == "stt" and pd.api.types.is_numeric_dtype(df[oid]):
            df[oid] = df[oid].map(lambda x: str(int(round(x))) if pd.notna(x) else None)
        # Rulebook §8: exact duplicate rows within the same order (all cells identical) -> count once
        non_stt_oids = [c for c in oid_cols if c != "stt"]
        if non_stt_oids:
            df = df.drop_duplicates().reset_index(drop=True)
    return df


def _clean_df(df: pd.DataFrame, hint_year: Optional[int] = None) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = dedupe([norm_name(c) for c in df.columns])
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
            df[c] = df[c].replace("", None)
    df = _drop_summary_rows(df)
    df = _fix_dates(df, hint_year=hint_year)
    df = _fix_numeric_strings(df)
    df = _fix_continuation_and_dupes(df)
    return df.reset_index(drop=True)


def _extract_hint_year(stem: str, sheet: str = "") -> Optional[int]:
    m = re.search(r"\b(202\d)\b", f"{stem} {sheet}")
    return int(m.group(1)) if m else None


_SHARD_SHEET_PAT = re.compile(
    r"^(sheet\d*|t\d{1,2}[_.\s-]*202\d|thang[_.\s-]*\d{1,2}[_.\s-]*202\d|c_\d{1,2}[_.\s-]*202\d|\d{1,2}[_.\s-]+202\d|202\d[_.\s-]+\d{1,2})$",
    re.I,
)


def _strip_shard_suffix(name: str) -> str:
    # Strip trailing _thang01_2025, _t1_2025, _2025, _1, _2
    s = re.sub(r"_(thang_?\d{1,2}_?|t\d{1,2}_?)?202\d$", "", name)
    s = re.sub(r"_(thang_?\d{1,2}|t\d{1,2})$", "", s)
    s = re.sub(r"_[1-9]\d?$", "", s)
    return s or name


def _canonical_table_name(file_stem: str, sheet_name: str, num_sheets: int) -> str:
    base = norm_name(file_stem)
    sh = norm_name(sheet_name) if sheet_name else ""
    base_clean = _strip_shard_suffix(base)
    sh_clean = _strip_shard_suffix(sh) if sh else ""
    if base_clean in ("du_lieu_ban_hang", "bao_cao_ban_hang", "du_lieu_xuat", "quan_ly", "quanly", "dulieubanhang", "dulieuxuat", "baocaobanhang", "ban_hang_tong_hop"):
        if sh_clean and not _SHARD_SHEET_PAT.match(sh):
            return sh_clean
    if num_sheets <= 1 or not sh or _SHARD_SHEET_PAT.match(sh) or _SHARD_SHEET_PAT.match(sheet_name.strip()):
        return base_clean
    if sh_clean == base_clean:
        return base_clean
    return f"{base_clean}__{sh_clean}"


def _register(con, name: str, df: pd.DataFrame, tables: list[str]) -> None:
    if name in tables:
        name = f"{name}_{len(tables)}"
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: (_parse_vn_date(v) or str(v)) if isinstance(v, (pd.Timestamp, datetime, date)) else v)
    con.register("_tmp_df", df)
    con.execute(f"create or replace table {qid(name)} as select * from _tmp_df")
    con.unregister("_tmp_df")
    tables.append(name)


def load_business(biz_dir: str | Path, con) -> dict:
    """Load every data file into DuckDB tables, auto-unioning sharded sheets/files. Never writes to biz_dir."""
    biz_dir = Path(biz_dir)
    tables: list[str] = []
    readme = ""
    # Collect (canonical_name, df) chunks so sharded files/sheets with identical schema are unioned
    chunks: list[tuple[str, pd.DataFrame]] = []

    for f in sorted(biz_dir.rglob("*")):
        if not f.is_file() or f.name.startswith("~$") or f.name.startswith("."):
            continue
        ext = f.suffix.lower()
        try:
            if f.name.lower() == "readme.md":
                readme = f.read_text(encoding="utf-8", errors="ignore")
            elif ext in (".csv", ".tsv", ".txt"):
                try:
                    df = pd.read_csv(f, sep=None, engine="python", encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(f, sep=None, engine="python", encoding="latin-1")
                hy = _extract_hint_year(f.stem)
                cdf = _clean_df(df, hint_year=hy)
                if not cdf.empty:
                    chunks.append((_canonical_table_name(f.stem, "", 1), cdf))
            elif ext in (".xlsx", ".xlsm", ".xls"):
                xl = pd.ExcelFile(f)
                n_sh = len(xl.sheet_names)
                for sh in xl.sheet_names:
                    raw_head = xl.parse(sh, header=None, nrows=15)
                    if raw_head.empty:
                        continue
                    h = _detect_header_row(raw_head)
                    df = xl.parse(sh, header=h)
                    hy = _extract_hint_year(f.stem, sh)
                    cdf = _clean_df(df, hint_year=hy)
                    if not cdf.empty:
                        chunks.append((_canonical_table_name(f.stem, sh, n_sh), cdf))
            elif ext in (".sqlite", ".db", ".sqlite3"):
                sq = sqlite3.connect(f"file:{f}?mode=ro", uri=True)
                names = [r[0] for r in sq.execute(
                    "select name from sqlite_master where type in ('table','view') and name not like 'sqlite_%'")]
                for t in names:
                    df = pd.read_sql_query(f'select * from "{t}"', sq)
                    cdf = _clean_df(df, hint_year=_extract_hint_year(t))
                    if not cdf.empty:
                        chunks.append((_strip_shard_suffix(norm_name(t)), cdf))
                sq.close()
        except Exception as e:  # one bad file must not kill the business
            print(f"[prep] failed to load {f.name}: {e}")

    # Group chunks by (canonical_name, column_set) to union sharded tables safely
    grouped: dict[tuple[str, tuple[str, ...]], list[pd.DataFrame]] = {}
    order_keys: list[tuple[str, tuple[str, ...]]] = []
    for cname, cdf in chunks:
        key = (cname, tuple(cdf.columns))
        if key not in grouped:
            grouped[key] = []
            order_keys.append(key)
        grouped[key].append(cdf)

    for cname, cols_key in order_keys:
        dfs = grouped[(cname, cols_key)]
        merged = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        if len(dfs) > 1:
            merged = _fix_continuation_and_dupes(merged)
        _register(con, cname, merged, tables)

    return {"tables": tables, "readme": readme}


# ----------------------------------------------------------------------------
# 2. Profiling
# ----------------------------------------------------------------------------
@dataclass
class ColProfile:
    table: str
    col: str
    dtype: str
    n: int
    null_rate: float
    n_distinct: int
    samples: list
    date_rate: float = 0.0
    num_rate: float = 0.0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    unique_ratio: float = 0.0


def profile_table(con, table: str, sample_n: int = 20000) -> list[ColProfile]:
    n_total = con.execute(f"select count(*) from {qid(table)}").fetchone()[0]
    df = con.execute(f"select * from {qid(table)} using sample {sample_n} rows").df() \
        if n_total > sample_n else con.execute(f"select * from {qid(table)}").df()
    out = []
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        date_rate = num_rate = 0.0
        vmin = vmax = None
        if len(nn):
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                num_rate = 1.0
                vmin, vmax = float(nn.min()), float(nn.max())
            elif pd.api.types.is_datetime64_any_dtype(s):
                date_rate = 1.0
            else:
                as_str = nn.astype(str)
                num = pd.to_numeric(as_str.str.replace(r"[,\s]", "", regex=True), errors="coerce")
                num_rate = float(num.notna().mean())
                if num_rate > 0.9:
                    vmin, vmax = float(num.min()), float(num.max())
                if num_rate < 0.9 or as_str.str.contains(r"[-/]").mean() > 0.9:
                    d = pd.to_datetime(as_str, errors="coerce", format="mixed")
                    date_rate = float(d.notna().mean())
        n_dist = int(nn.nunique()) if len(nn) else 0
        out.append(ColProfile(
            table=table, col=c, dtype=str(s.dtype), n=n_total,
            null_rate=float(s.isna().mean()), n_distinct=n_dist,
            samples=[str(x) for x in nn.drop_duplicates().head(5).tolist()],
            date_rate=date_rate, num_rate=num_rate, vmin=vmin, vmax=vmax,
            unique_ratio=(n_dist / len(nn)) if len(nn) else 0.0,
        ))
    return out


# ----------------------------------------------------------------------------
# 3. Role mapping
# ----------------------------------------------------------------------------
# role -> regexes on the normalized column name; earlier pattern = higher priority
ROLE_KEYWORDS: dict[str, list[str]] = {
    "date":        [r"^ngay_?ban", r"^ngay_?dat", r"^ngay_?hd", r"^ngay_?giao_?dich", r"^ngay_?lap", r"^ngay_?mua",
                    r"^ngay_?chung_?tu", r"^order_date", r"^sale_date", r"^ngay_?thang$", r"^ngay$", r"^thoi_gian$",
                    r"ngay_?ban", r"ngay_?dat", r"ngay", r"date", r"thoi_gian", r"time", r"created"],
    "order_id":    [r"^ma_don", r"^order_id$", r"^so_hd$", r"^ma_hd$", r"^ma_hoa_don$", r"^hoa_don$", r"^hd$",
                    r"^ma_phieu$", r"^so_phieu$", r"^ma_giao_dich$", r"^ma_chung_tu$", r"^so_chung_tu$", r"^so_ct$",
                    r"^ma_bill$", r"ma_don", r"order_id", r"so_hd", r"ma_hd", r"hoa_don", r"invoice", r"bill",
                    r"receipt", r"ma_phieu", r"^stt$"],
    "product":     [r"^ten_san_pham$", r"^ten_sp$", r"^ten_hang$", r"^ten_mon$", r"^ten_thuoc$", r"^san_pham$",
                    r"^san_pham__ten_goi$", r"^hang_hoa__ten_goi$", r"^ten_goi$", r"^hang_ban$",
                    r"^mat_hang$", r"^hang_hoa$", r"^noi_dung$", r"^hang$", r"^lam_gi$", r"ten_san_pham", r"ten_sp",
                    r"ten_hang", r"ten_mon", r"ten_thuoc", r"product_name", r"product", r"mat_hang", r"hang_hoa", r"item"],
    "product_id":  [r"^ma_sp$", r"^ma_san_pham", r"^sku$", r"^product_id$", r"^ma_hang", r"^ma_mat_hang$", r"^ma_h_hoa$",
                    r"^ma_hh$", r"^id_sp$", r"^ma_thuoc$", r"^ma_mon$", r"ma_sp", r"ma_san_pham", r"sku", r"product_id",
                    r"ma_hang", r"ma_thuoc", r"ma_mon"],
    "customer":    [r"^ten_khach", r"^khach_hang$", r"^khach_hang_ten$", r"^khach_hang_mua$", r"^khach_hang__ten_goi$",
                    r"^khach_le_si$", r"^ten_kh$", r"^khach$", r"^kh$", r"^doi_tac$", r"^ben_mua$",
                    r"^ma_kh$", r"^ma_khach", r"^so_kh$", r"^customer", r"ten_khach", r"khach_hang", r"customer", r"khach"],
    "qty":         [r"so_luong_quy_doi", r"sl_quy_doi", r"slquydoi", r"quy_ra_dv_le", r"sl_quy_ra_le", r"sl_dv_co_ban",
                    r"so_luong_theo_don_vi_le", r"so_luong_le", r"sl_le", r"sl_co_ban", r"sl_dvt_co_ban",
                    r"^so_luong$", r"^so_luong_san_pham$", r"^so_luong_hang$", r"^sl_ban$", r"^sl$", r"^s_l$",
                    r"so_luong", r"(^|_)sl($|_)", r"qty", r"quantity"],
    "unit_price":  [r"^don_gia$", r"^don_gia_d$", r"^don_gia_ban$", r"^gia_ban$", r"^gia_ban_le$", r"^gia_le$",
                    r"^gia_1_dv$", r"^gia_don_vi$", r"^dg$", r"^gia$",
                    r"don_gia", r"gia_ban", r"unit_price", r"price"],
    "line_amount": [r"^thanh_tien$", r"^thanh_tien_d$", r"^tri_gia$", r"^t_tien$", r"^tien_ban$", r"^tt$", r"^cong$",
                    r"^line_amount$", r"^line_total$", r"^amount$", r"^doanh_thu$", r"^tien_hang$", r"^gia_tri_hang$",
                    r"^tien$", r"^tong_tien$", r"^tong$", r"thanh_tien", r"tri_gia", r"line_amount", r"line_total",
                    r"amount", r"doanh_thu", r"revenue", r"total"],
    "discount":    [r"^giam_gia_dong$", r"^chiet_khau_dong$", r"^giam_d$", r"^tien_giam$", r"^ck$", r"^chiet_khau$",
                    r"^giam_gia$", r"giam_gia_dong", r"chiet_khau", r"giam_gia", r"discount", r"(^|_)ck($|_)"],
    "voucher":     [r"voucher", r"coupon", r"khuyen_mai", r"ma_khuyen_mai"],
    "status":      [r"^trang_thai", r"^tinh_trang_don$", r"^tinh_trang$", r"^status$", r"^ket_qua$", r"^xu_ly$",
                    r"^ghi_chu_tt$", r"trang_thai", r"status", r"tinh_trang", r"ket_qua", r"^ghi_chu$", r"^note$"],
    "cost":        [r"^gia_von_luc_ban$", r"^gia_von$", r"^gia_nhap$", r"^don_gia_nhap$",
                    r"gia_von", r"gia_nhap", r"don_gia_nhap", r"cost"],
    "vat":         [r"vat", r"thue"],
    "delivery":    [r"phi_van_chuyen", r"phi_giao", r"van_chuyen", r"ship", r"delivery"],
    "branch":      [r"chi_nhanh", r"branch", r"cua_hang", r"store"],
    "expiry":      [r"han_dung", r"han_su_dung", r"ngay_het_han", r"het_han", r"expiry", r"(^|_)exp($|_)", r"(^|_)hsd($|_)"],
    "stock":       [r"ton_cuoi_ky", r"ton_cuoi", r"ton_thuc", r"ton_kho", r"so_luong_ton", r"stock", r"(^|_)ton($|_)"],
}
NUMERIC_ROLES = {"qty", "unit_price", "line_amount", "discount", "voucher", "cost", "vat", "delivery", "stock"}
DATE_ROLES = {"date", "expiry"}
PRECONVERTED_QTY_PAT = re.compile(
    r"so_luong_quy_doi|sl_quy_doi|slquydoi|quy_ra_dv_le|sl_quy_ra_le|so_luong_theo_don_vi_le|so_luong_le|sl_le|sl_co_ban|sl_dvt_co_ban"
)

CANCEL_PAT = re.compile(r"(^|_)(huy|da_huy|bi_huy|cancel|cancelled|canceled|void|da_xoa|deleted|that_bai|failed|khong_giao_duoc|giao_that_bai)($|_)", re.I)
RETURN_COL_PAT = re.compile(r"tra_hang|tra_lai|hoan_tra|hoan_hang|hoan_tien|return|refund", re.I)
RETURN_VAL_PAT = re.compile(r"(^|_)(tra_hang|(?<!kiem_)tra_lai|tra_lai_hang|hang_tra_ve|tra_ve|khach_mang_tra|khach_tra_lai|hoan_tra|hoan_hang|hoan_tien|doi_tra|return|returned|refund)($|_)", re.I)


def guess_roles_table(profiles: list[ColProfile], allowed_fact_cols: Optional[set[str]] = None) -> dict[str, list[str]]:
    """{role: [col,...]} ordered by pattern priority then column order."""
    m: dict[str, list[tuple[int, int, str]]] = {}
    for pos, p in enumerate(profiles):
        for role, pats in ROLE_KEYWORDS.items():
            if allowed_fact_cols is not None and role in ("qty", "line_amount", "discount"):
                if p.col not in allowed_fact_cols:
                    continue
            idx = next((i for i, pt in enumerate(pats) if re.search(pt, p.col)), None)
            if idx is None:
                continue
            if role == "date" and re.search(r"sinh|dob|birth|vao_lam|dang_ky|ap_dung", p.col):
                continue
            if role == "vat" and re.search(r"ma_so_thue|mst", p.col):
                continue
            if role == "product" and (p.num_rate >= 0.8 or re.search(r"(^|_)(ma|id|code|sku)_", p.col)):
                continue
            if role == "customer" and (p.num_rate >= 0.8 or re.search(r"id_cua_hang|khach_tra|khach_da_tra|so_khach|trang_thai|ghi_chu|diem|loai|nhom", p.col)):
                continue
            if role == "status" and re.search(r"_kh$|_hddt$|khach_hang__|san_pham__", p.col):
                continue
            if role in NUMERIC_ROLES and p.num_rate < 0.85:
                continue
            if role in DATE_ROLES and p.date_rate < 0.75:
                continue
            if role in ("order_id", "product_id") and p.num_rate >= 0.9 and p.dtype.startswith("float") and p.col != "stt":
                continue
            if role == "product" and p.num_rate >= 0.8:
                continue
            m.setdefault(role, []).append((idx, pos, p.col))
    roles = {r: [c for _, _, c in sorted(v)] for r, v in m.items()}
    # an expiry column is not a transaction date
    exp = set(roles.get("expiry", []))
    if "date" in roles:
        roles["date"] = [c for c in roles["date"] if c not in exp]
        if not roles["date"]:
            del roles["date"]
    return roles


def validate_roles(profiles: list[ColProfile], roles: dict[str, list[str]]) -> list[str]:
    issues = []
    by_col = {p.col: p for p in profiles}
    for role, cs in roles.items():
        for c in cs:
            p = by_col.get(c)
            if p is None:
                issues.append(f"{c}: column missing")
                continue
            if role == "date" and re.search(r"sinh|dob|birth|vao_lam|dang_ky|ap_dung", p.col):
                continue
            if role == "vat" and re.search(r"ma_so_thue|mst", p.col):
                continue
            if role == "product" and (p.num_rate >= 0.8 or re.search(r"(^|_)(ma|id|code|sku)_", p.col)):
                continue
            if role == "customer" and (p.num_rate >= 0.8 or re.search(r"id_cua_hang|khach_tra|khach_da_tra|so_khach|trang_thai|ghi_chu|diem|loai|nhom", p.col)):
                continue
            if role == "status" and re.search(r"_kh$|_hddt$|khach_hang__|san_pham__", p.col):
                continue
            if role in NUMERIC_ROLES and p.num_rate < 0.85:
                issues.append(f"{c}: role {role} but only {p.num_rate:.0%} numeric")
            if role in DATE_ROLES and p.date_rate < 0.75:
                issues.append(f"{c}: role {role} but only {p.date_rate:.0%} dates")
    return issues


def refine_roles(profiles: list[ColProfile], roles: dict, readme: str,
                 llm: Optional[Callable[[str], str]] = None) -> dict:
    """Optional LLM refinement (temp 0). Accepted only if it validates."""
    if llm is None:
        return roles
    brief = [{"col": p.col, "dtype": p.dtype, "null": round(p.null_rate, 2),
              "distinct": p.n_distinct, "samples": p.samples[:3]} for p in profiles]
    prompt = (
        "Map columns of a Vietnamese small-business sales table to roles.\n"
        f"Roles: {', '.join(ROLE_KEYWORDS)}.\n"
        'Return ONLY JSON {"role": ["col", ...]}. Use only listed columns; omit a role if unsure.\n\n'
        f"README:\n{readme[:2500]}\n\nHeuristic guess:\n{json.dumps(roles, ensure_ascii=False)}\n\n"
        f"Columns:\n{json.dumps(brief, ensure_ascii=False)}"
    )
    try:
        txt = llm(prompt)
        txt = txt[txt.index("{"): txt.rindex("}") + 1]
        proposed = json.loads(txt)
    except Exception:
        return roles
    valid = {p.col for p in profiles}
    cand = {r: [c for c in cs if c in valid] for r, cs in proposed.items()
            if r in ROLE_KEYWORDS and isinstance(cs, list)}
    cand = {r: cs for r, cs in cand.items() if cs}
    if validate_roles(profiles, cand):
        return roles
    merged = dict(roles)
    merged.update(cand)
    return merged


# ----------------------------------------------------------------------------
# 4. Fact table + joins -> flat `base`
# ----------------------------------------------------------------------------
def _cols(con, table: str) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in con.execute(f"describe {qid(table)}").fetchall()]


def _kind(duck_type: str) -> str:
    t = duck_type.upper()
    if any(k in t for k in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "HUGEINT")):
        return "number"
    if t.startswith("TIMESTAMP") or t == "DATE":
        return "time"
    if t == "BOOLEAN":
        return "boolean"
    return "string"


_NON_FACT_TABLE_PAT = re.compile(
    r"nhap|phieu_nhap|chi_tiet_nhap|mua_hang|ton_kho|the_kho|kiem_ke|lo_hang|lo_thuoc|"
    r"thanh_toan|so_quy|phieu_thu|phieu_chi|chi_phi|cong_no|so_no|so_gop|bao_hanh|thu_may_cu|"
    r"bang_gia|lich_su_gia|quy_doi|don_vi|danh_muc|khach_hang|san_pham|hang_hoa|nhan_vien|danh_sach_tho|"
    r"khuyen_mai|giao_hang|phieu_giao|bao_cao_ca|ghi_tat",
    re.I,
)


def choose_fact_table(con, tables: list[str], profs: dict[str, list[ColProfile]]) -> str:
    best, best_score = tables[0], -1e9
    for t in tables:
        roles = guess_roles_table(profs[t])
        n = profs[t][0].n if profs[t] else 0
        score = 0.0
        for r, w in (("qty", 3), ("line_amount", 3), ("unit_price", 2), ("product", 2), ("product_id", 1), ("date", 1), ("order_id", 1)):
            if r in roles:
                score += w
        if RETURN_COL_PAT.search(t):
            score -= 8
        if _NON_FACT_TABLE_PAT.search(t) and not re.search(r"ban_hang|hoa_don|don_hang|chi_tiet_don|chi_tiet_ban", t):
            score -= 6
        if re.search(r"chi_tiet|ban_hang|so_ban_hang|sobanhang", t) and not re.search(r"nhap", t):
            score += 3
        score += min(n, 1_000_000) / 1_000_000.0 * 2
        if score > best_score:
            best, best_score = t, score
    return best


_JOIN_KEY_GROUPS = [
    re.compile(r"^(ma_don|ma_don_hang|order_id|so_hd|ma_hd|ma_hoa_don|hoa_don|hd|ma_phieu|so_phieu|ma_giao_dich|ma_chung_tu|so_chung_tu|so_ct|ma_bill)$"),
    re.compile(r"^(ma_sp|ma_san_pham|sku|product_id|ma_hang|ma_h_hoa|id_sp|ma_thuoc|ma_mon|ma_vach|ma_vach_ean|ma_barcode)$"),
    re.compile(r"^(ma_kh|ma_khach|ma_khach_hang|ma_khach_mua|ma_kh_mua|ma_nguoi_mua|so_kh|customer_id|id_kh)$"),
    re.compile(r"^(ma_nv|ma_nhan_vien|nv_ban|nhan_vien_id)$"),
    re.compile(r"^(ma_cn|ma_chi_nhanh|branch_id)$"),
    re.compile(r"^(ten_san_pham|ten_sp|ten_hang|ten_mon|ten_thuoc|san_pham|mat_hang|hang_hoa|hang)$"),
]

_EXCLUDE_JOIN_TABLE_PAT = re.compile(
    r"nhap|the_kho|ton_kho|thanh_toan|so_quy|phieu_thu|phieu_chi|chi_phi|so_no|so_gop|bao_hanh|thu_may_cu|"
    r"bang_gia|lich_su_gia|quy_doi|don_vi|khuyen_mai|giao_hang|phieu_giao|bao_cao_ca|lo_hang|lo_thuoc",
    re.I,
)


def _same_key_group(c1: str, c2: str) -> bool:
    if c1 == c2:
        return True
    for g in _JOIN_KEY_GROUPS:
        if g.search(c1) and g.search(c2):
            return True
    return False


def _join_candidates(con, left: str, right: str, rprof: list[ColProfile]) -> list[tuple[str, str]]:
    """Key pairs (lk, rk) where `right` is unique on `rk` and covers >= 70% of `left.lk` values."""
    lcols = {c: k for c, k in _cols(con, left)}
    out: list[tuple[int, str, str]] = []
    for p in rprof:
        if p.n_distinct < 2 or p.unique_ratio < 0.98 or p.null_rate > 0.2:
            continue
        if p.dtype.startswith("float"):
            continue
        for lk, ltype in lcols.items():
            if _kind(ltype) == "time" or (_kind(ltype) == "number" and p.dtype.startswith("float")):
                continue
            same_grp = _same_key_group(lk, p.col)
            if not same_grp:
                # Allow cross-name join when both columns look like ID/code/key columns
                if _kind(ltype) != "string" or not (re.search(r"(^|_)(ma|id|so|sku|code|kh|sp|hh)", lk) and re.search(r"(^|_)(ma|id|so|sku|code|kh|sp|hh)", p.col)):
                    continue
            try:
                cov = con.execute(f"""
                    select avg((v in (select {qid(p.col)} from {qid(right)} where {qid(p.col)} is not null))::int) from
                      (select distinct {qid(lk)} v from {qid(left)} where {qid(lk)} is not null limit 50000)
                """).fetchone()[0]
            except Exception:
                cov = None
            if cov is not None and cov >= 0.7:
                prio = 0 if lk == p.col else (1 if same_grp else 2)
                out.append((prio, lk, p.col))
    out.sort(key=lambda x: x[0])
    return [(lk, rk) for _, lk, rk in out]


def build_base(con, fact: str, others: list[str], profs: dict[str, list[ColProfile]], notes: list[str]) -> list[str]:
    """CREATE TABLE base = fact LEFT JOIN each joinable dimension/header table (2 passes)."""
    fact_col_list = [c for c, _ in _cols(con, fact)]
    cols = [f"f.{qid(c)}" for c in fact_col_list]
    joins, joined = [], []
    available = {fact: "f"}

    # Special handling for abbreviation table (e.g. `ghi_tat` in dev_011, dev_014)
    pending = []
    for t in others:
        if re.search(r"ghi_tat", t):
            tcols = [c for c, _ in _cols(con, t)]
            if "ghi_tat" in tcols and "ten_hang" in tcols and "hang" in fact_col_list:
                a = "j_gt"
                joins.append(
                    f"left join (select * from {qid(t)} qualify row_number() over (partition by lower(ghi_tat)) = 1) {a} "
                    f"on lower(f.\"hang\") = lower({a}.\"ghi_tat\") or lower(f.\"hang\") = lower({a}.\"ten_hang\")"
                )
                cols.append(f"coalesce({a}.\"ten_hang\", f.\"hang\") as \"ten_hang\"")
                for c in tcols:
                    if c not in ("ghi_tat", "ten_hang") and c not in fact_col_list:
                        cols.append(f"{a}.{qid(c)} as {qid(c)}")
                notes.append(f"joined abbreviation table {t} onto hang -> ten_hang")
                joined.append(t)
            continue
        if RETURN_COL_PAT.search(t) or _EXCLUDE_JOIN_TABLE_PAT.search(t):
            continue
        pending.append(t)

    for _pass in range(2):
        for t in list(pending):
            for left_t, alias in list(available.items()):
                cands = _join_candidates(con, left_t, t, profs[t])
                if not cands:
                    continue
                lk, rk = cands[0]
                a = f"j{len(joined)}"
                sub = f"(select * from {qid(t)} where {qid(rk)} is not null qualify row_number() over (partition by {qid(rk)}) = 1)"
                joins.append(f"left join {sub} {a} on {alias}.{qid(lk)} = {a}.{qid(rk)}")
                existing = {c.split(" as ")[-1].split(".")[-1].strip('"') for c in cols}
                is_order_hdr = bool(re.search(r"don_hang|hoa_don", t) and not re.search(r"khach|san_pham|hang_hoa", t))
                for c, _ in _cols(con, t):
                    if c == rk:
                        continue
                    force_prefix = (c == "ten_goi") or (not is_order_hdr and bool(re.search(r"^(trang_thai|tinh_trang|ghi_chu|note|ngay_sinh|sinh_nhat|id_cua_hang)", c)))
                    name = f"{t}__{c}" if (c in existing or force_prefix) else c
                    cols.append(f"{a}.{qid(c)} as {qid(name)}")
                    existing.add(name)
                notes.append(f"joined {t} on {lk}={rk}")
                joined.append(t)
                available[t] = a
                pending.remove(t)
                break
    con.execute(f"create or replace table base as select {', '.join(cols)} from {qid(fact)} f {' '.join(joins)}")
    return joined


# ----------------------------------------------------------------------------
# 5. Unit conversion & Views
# ----------------------------------------------------------------------------
@dataclass
class RuleOptions:
    """Rulebook defaults baked into lines_valid. Override per split if rulebook differs."""
    exclude_cancelled: bool = True
    exclude_orders_with_return: bool = True
    subtract_line_discount: bool = True
    subtract_order_voucher: bool = False

    @classmethod
    def from_rulebook(cls, text: str) -> "RuleOptions":
        opts = cls()
        if not text:
            return opts
        u = unaccent(text).lower()
        if re.search(r"bao gom (ca )?don (bi )?huy|tinh ca don huy", u):
            opts.exclude_cancelled = False
        if re.search(r"khong loai (bo )?don (co )?(tra|hoan)|giu nguyen don co tra hang", u):
            opts.exclude_orders_with_return = False
        if re.search(r"tru voucher cap don vao doanh thu dong", u):
            opts.subtract_order_voucher = True
        return opts


def _build_unit_conversion_table(con, raw_tables: list[str], readme: str, notes: list[str]) -> bool:
    """Build DuckDB table `_unit_conv(prod_key, unit_key, factor)` from conversion tables and README bullets."""
    records: dict[tuple[str, str], float] = {}

    # 1. Parse README unit conversion bullets:
    #    "- Coca Cola 330ml: 1 lốc = 6 lon"
    #    "- **Hàng theo bó**: Ống PVC ..., 1 bó = 10 cây"
    if readme:
        for line in readme.splitlines():
            m = re.match(r"^\s*-\s+([^:*]+?):\s*1\s+(.+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+\S+", line)
            if m:
                pk = unaccent(m.group(1)).lower().strip()
                uk = unaccent(m.group(2)).lower().strip()
                fac = float(m.group(3).replace(",", "."))
                if pk and uk and fac > 0:
                    records[(pk, uk)] = fac
            m2 = re.search(r"\*\*Hàng theo\s+([^*:]+)\*\*:\s*(.+?)\s+bán theo\s+\1,\s*1\s+\1\s*=\s*(\d+(?:[.,]\d+)?)", line, re.I)
            if m2:
                uk = unaccent(m2.group(1)).lower().strip()
                fac = float(m2.group(3).replace(",", "."))
                for pname in m2.group(2).split(","):
                    pk = unaccent(pname).lower().strip()
                    if pk and uk and fac > 0:
                        records[(pk, uk)] = fac

    # 2. Parse conversion tables in DuckDB (e.g. danhsachquydoi, danhmuc, danhsachdonvitinh, don_vi_sp, banhang_donviquydoi)
    conv_tables = [t for t in raw_tables if re.search(r"quy_doi|quydoi|don_vi|donvi|danh_muc|danhmuc", t)]
    for t in conv_tables:
        df = con.execute(f"select * from {qid(t)}").df()
        if df.empty:
            continue
        cols = list(df.columns)
        p_cols = [c for c in cols if re.search(r"ten_sp|ten_san_pham|ten_hang|san_pham|mat_hang|hang_hoa|ma_sp|ma_hang|sku|id_sp", c)]
        u_cols = [c for c in cols if re.search(r"don_vi_lon|dvt_lon|don_vi_quy_doi|don_vi_ban|don_vi_tinh|don_vi|dvt|unit", c)
                  and not re.search(r"co_ban|nho|le|goc", c)]
        f_cols = [c for c in cols if re.search(r"he_so|so_luong_quy_doi|ty_le|quy_doi|sl_quy_doi|gia_tri", c)]
        if not p_cols or not f_cols:
            continue
        fc = f_cols[0]
        uc = u_cols[0] if u_cols else None
        # First pass: direct numeric factors and "1 thùng = 4 bịch" text formulas
        raw_rows = []
        base_units: dict[tuple[str, str], float] = {}
        for _, r in df.iterrows():
            pkeys = [unaccent(str(r[pc])).lower().strip() for pc in p_cols if pd.notna(r[pc]) and str(r[pc]).strip()]
            if not pkeys or pd.isna(r[fc]):
                continue
            fval_str = str(r[fc]).strip()
            uname = unaccent(str(r[uc])).lower().strip() if uc and pd.notna(r[uc]) else ""
            # Pattern A: "1 thùng = 4 bịch" or "1 thùng = 24 lon; 1 lốc = 6 lon"
            if "=" in fval_str:
                for part in re.split(r"[;,]", fval_str):
                    m = re.search(r"1\s+([^\d=]+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+([^\d=;,\s]+)", part)
                    if m:
                        u_big = unaccent(m.group(1)).lower().strip()
                        fac = float(m.group(2).replace(",", "."))
                        for pk in pkeys:
                            records[(pk, u_big)] = fac
                continue
            # Pattern B: numeric factor (e.g. 24) or chained factor (e.g. "15 vỉ" where 1 vỉ = 12 viên)
            num = _parse_vn_num(r[fc], allow_unit=False)
            if num is not None and num > 0 and uname:
                for pk in pkeys:
                    records[(pk, uname)] = float(num)
                    base_units[(pk, uname)] = float(num)
            elif uname:
                raw_rows.append((pkeys, uname, fval_str))
        # Second pass: resolve chained factors like "15 vỉ" -> 15 * factor("vỉ")
        for pkeys, uname, fval_str in raw_rows:
            m = re.match(r"^(\d+(?:[.,]\d+)?)\s+(.+)$", unaccent(fval_str).lower().strip())
            if m:
                mult = float(m.group(1).replace(",", "."))
                sub_u = m.group(2).strip()
                for pk in pkeys:
                    sub_fac = base_units.get((pk, sub_u), 1.0)
                    records[(pk, uname)] = mult * sub_fac

    if not records:
        return False
    conv_df = pd.DataFrame([(pk, uk, fac) for (pk, uk), fac in records.items()], columns=["prod_key", "unit_key", "factor"])
    con.register("_tmp_conv", conv_df)
    con.execute("create or replace table _unit_conv as select * from _tmp_conv")
    con.unregister("_tmp_conv")
    notes.append(f"built _unit_conv table with {len(conv_df)} product-unit conversion rules")
    return True


def _num(c: str) -> str:
    return f"try_cast({qid(c)} as double)"


def _date_expr(c: str) -> str:
    v = f"cast({qid(c)} as varchar)"
    return (f"coalesce(try_cast({qid(c)} as date), "
            f"try_strptime({v}, '%d/%m/%Y')::date, try_strptime({v}, '%d-%m-%Y')::date, "
            f"try_strptime({v}, '%Y/%m/%d')::date, try_strptime({v}, '%d/%m/%Y %H:%M:%S')::date, "
            f"try_strptime({v}, '%d/%m/%Y %H:%M')::date, try_strptime({v}, '%d/%m/%y')::date)")


def derive_amount(con, roles: dict[str, str], roles_l: dict[str, list[str]], opts: RuleOptions,
                  profs: dict[str, ColProfile], notes: list[str]) -> str:
    q, p, d = roles.get("qty"), roles.get("unit_price"), roles.get("discount")
    cands = roles_l.get("line_amount", [roles["line_amount"]] if "line_amount" in roles else [])
    la = cands[0] if cands else None

    # If multiple line_amount candidates exist and qty & unit_price are available, pick the candidate matching qty*price
    if len(cands) > 1 and q and p:
        best_c, best_match = la, -1.0
        for c in cands:
            try:
                m = con.execute(
                    f"select avg((abs({_num(c)} - {_num(q)}*{_num(p)}) <= greatest(1.0, 0.25*abs({_num(q)}*{_num(p)})))::int) "
                    f"from base where {_num(c)} is not null and {_num(q)} is not null and {_num(p)} is not null"
                ).fetchone()[0] or 0.0
            except Exception:
                m = 0.0
            if m > best_match:
                best_c, best_match = c, m
        if best_c and best_c != la and best_match >= 0.5:
            la = best_c
            roles["line_amount"] = la

    disc = "0"
    if d and opts.subtract_line_discount and not re.search(r"hd|don|bill|phieu|bot", d):
        if re.search(r"pct|percent|phan_tram|ti_le|%", d) and q and p:
            disc = f"coalesce({_num(q)}*{_num(p)}*{_num(d)}/100.0, 0)"
        else:
            disc = f"coalesce({_num(d)}, 0)"

    if la and q and p and disc != "0":
        # Check ONLY rows where discount > 0 to see if `la` is gross (needs discount subtracted) or already net
        r = con.execute(f"""
            select count(*),
                   avg((abs({_num(la)} - ({_num(q)}*{_num(p)} - {disc})) < 1.0)::int),
                   avg((abs({_num(la)} - {_num(q)}*{_num(p)}) < 1.0)::int)
            from base where {disc} > 0""").fetchone()
        n_disc, net_on_disc, gross_on_disc = (r[0] or 0), (r[1] or 0.0), (r[2] or 0.0)
        if n_disc >= 5 and gross_on_disc >= 0.85 and net_on_disc < 0.15:
            notes.append(f"amount = {la} - {d} (line amount is gross on {n_disc} discounted rows)")
            return f"({_num(la)} - {disc})"
        notes.append(f"amount = {la} (already net of line discount)")
        return _num(la)
    if la:
        notes.append(f"amount = {la}")
        return _num(la)
    if q and p:
        notes.append(f"amount = {q}*{p}" + (f" - {d}" if disc != "0" else ""))
        return f"({_num(q)}*{_num(p)} - {disc})"
    notes.append("WARNING: no amount column or qty*price found; amount is NULL")
    return "null::double"


def detect_status(con, col: str) -> tuple[list[str], list[str]]:
    vals = [r[0] for r in con.execute(
        f"select distinct {qid(col)} from base where {qid(col)} is not null limit 300").fetchall()]
    cancel = [v for v in vals if CANCEL_PAT.search(norm_name(str(v)))]
    ret = [v for v in vals if RETURN_VAL_PAT.search(norm_name(str(v)))]
    return cancel, ret


def _build_auxiliary_views(con, raw_tables: list[str], roles: dict[str, str], notes: list[str]) -> list[str]:
    """Build optional helper views `inventory_now`, `debts_open`, `cost_view` when supporting tables/columns exist."""
    aux: list[str] = []
    # 1. inventory_now
    inv_tables = [t for t in raw_tables if re.search(r"ton_kho|tonkho|kiem_ke", t)]
    if not inv_tables:
        inv_tables = [t for t in raw_tables if re.search(r"san_pham|hang_hoa|danh_muc", t)
                      and any(re.search(r"ton_cuoi|ton_thuc|ton_kho|so_luong_ton|(^|_)ton($|_)", c) for c, _ in _cols(con, t))]
    if inv_tables:
        it = inv_tables[0]
        try:
            con.execute(f"create or replace view inventory_now as select * from {qid(it)}")
            aux.append(f"inventory_now (from {it})")
        except Exception:
            pass

    # 2. debts_open
    debt_tables = [t for t in raw_tables if re.search(r"cong_no|so_no|sono", t)]
    if debt_tables:
        dt = debt_tables[0]
        try:
            con.execute(f"create or replace view debts_open as select * from {qid(dt)}")
            aux.append(f"debts_open (from {dt})")
        except Exception:
            pass

    # 3. cost_view
    if "cost" in roles:
        try:
            con.execute("create or replace view cost_view as select *, (amount - coalesce(qty, 1) * unit_cost) as line_profit from lines_valid where unit_cost is not null")
            aux.append("cost_view (lines_valid with unit_cost and line_profit)")
        except Exception:
            pass
    if aux:
        notes.append("auxiliary views: " + ", ".join(aux))
    return aux


def build_views(con, roles: dict[str, str], roles_l: dict[str, list[str]], profs: dict[str, ColProfile],
                return_tables: list[str], has_unit_conv: bool, opts: RuleOptions, notes: list[str]) -> None:
    base_cols = [c for c, _ in _cols(con, "base")]
    dt_col = roles.get("date")
    dt = _date_expr(dt_col) if dt_col else "null::date"
    amt = derive_amount(con, roles, roles_l, opts, profs, notes)

    # Detect cancelled / returned values across status and note columns
    canc_clauses, ret_terms = [], []
    status_candidates = list(dict.fromkeys(([roles["status"]] if "status" in roles else []) +
                                           [c for c in base_cols if re.search(r"trang_thai|status|tinh_trang|ket_qua|ghi_chu|note", c)]))
    for st_col in status_candidates:
        c_vals, r_vals = detect_status(con, st_col)
        if c_vals:
            canc_clauses.append(f"({qid(st_col)} in ({','.join(qlit(v) for v in c_vals)}))")
            notes.append(f"status col {st_col}: cancelled values={c_vals}")
        if r_vals:
            ret_terms.append(f"({qid(st_col)} in ({','.join(qlit(v) for v in r_vals)}))")
            notes.append(f"status col {st_col}: returned values={r_vals}")

    canc = " or ".join(canc_clauses) if canc_clauses else "false"

    for c in base_cols:
        if c in roles.values() or not RETURN_COL_PAT.search(c):
            continue
        pr = profs.get(c)
        if pr is not None and pr.num_rate >= 0.85:
            ret_terms.append(f"(coalesce({_num(c)}, 0) > 0)")
            notes.append(f"returned-line flag from numeric column {c} > 0")
        elif pr is not None:
            ret_terms.append(f"(lower(cast({qid(c)} as varchar)) in ('1','true','co','có','x','yes','y'))")
            notes.append(f"returned-line flag from text column {c}")

    # Non-merchandise rows (Voucher, Dùng <n> điểm, Phí ship) vs Service fee (Phí phục vụ)
    prod_col = roles.get("product")
    if prod_col:
        p_lower = f"lower(trim(cast({qid(prod_col)} as varchar)))"
        non_item_expr = (
            f"regexp_matches({p_lower}, "
            f"'^(voucher|chiet khau|chiết khấu|giam gia|giảm giá|khuyen mai|khuyến mãi|dung .* diem|dùng .* điểm|"
            f"phi ship|phí ship|phi van chuyen|phí vận chuyển|phi giao hang|phí giao hàng)$')"
        )
        srv_fee_expr = f"({p_lower} in ('phí phục vụ', 'phi phuc vu', 'phụ thu', 'phu thu'))"
    else:
        non_item_expr = "false"
        srv_fee_expr = "false"

    # Negative qty or negative amount on merchandise rows indicates a returned line (Rulebook §1 & biz_105/117/dev_004)
    q_col = roles.get("qty")
    if q_col:
        n_neg_q = con.execute(f"select count(*) from base where {_num(q_col)} < 0 and not ({non_item_expr})").fetchone()[0]
        if n_neg_q > 0:
            ret_terms.append(f"({_num(q_col)} < 0 and not ({non_item_expr}))")
            notes.append(f"returned-line flag from negative qty ({q_col} < 0: {n_neg_q} rows)")
    n_neg_amt = con.execute(f"select count(*) from base where ({amt}) < 0 and not ({non_item_expr})").fetchone()[0]
    if n_neg_amt > 0:
        ret_terms.append(f"(({amt}) < 0 and not ({non_item_expr}))")
        notes.append(f"returned-line flag from negative amount ({n_neg_amt} rows)")

    ret_line = " or ".join(ret_terms) if ret_terms else "false"

    oid = roles.get("order_id")
    if oid:
        part_key = f"{dt}, {qid(oid)}" if oid == "stt" and dt_col else qid(oid)
        canc_ord = f"bool_or({canc}) over (partition by {part_key})"
        ret_ord = f"bool_or({ret_line}) over (partition by {part_key})"
        for rt in return_tables:
            rt_cols = [c for c, _ in _cols(con, rt)]
            match_oid = next((c for c in rt_cols if _same_key_group(oid, c)), None)
            if match_oid:
                ret_ord = f"({ret_ord} or {qid(oid)} in (select {qid(match_oid)} from {qid(rt)} where {qid(match_oid)} is not null))"
                notes.append(f"returned orders also taken from table {rt}.{match_oid}")
    else:
        canc_ord, ret_ord = canc, ret_line

    # Base-unit quantity conversion if `qty` is not already pre-converted and `_unit_conv` + unit column exist
    qty_expr = None
    if q_col:
        unit_cols = [c for c in base_cols if re.search(r"^(don_vi_tinh|don_vi_ban|don_vi|dvt|unit|donvi)$", c)]
        if has_unit_conv and unit_cols and not PRECONVERTED_QTY_PAT.search(q_col):
            u_col = unit_cols[0]
            p_keys = [roles[r] for r in ("product", "product_id") if r in roles]
            if p_keys:
                pk_col = p_keys[0]
                qty_expr = (
                    f"({_num(q_col)} * coalesce((select max(uc.factor) from _unit_conv uc "
                    f"where uc.prod_key = norm_vn(base.{qid(pk_col)}) "
                    f"and uc.unit_key = norm_vn(base.{qid(u_col)})), 1.0))"
                )
                notes.append(f"qty = {q_col} * _unit_conv factor (matched on {pk_col}, {u_col})")
        if qty_expr is None:
            qty_expr = _num(q_col)
            notes.append(f"qty = {q_col}")

    std = [f"{dt} as dt", f"{amt} as amount"]
    taken = {"dt", "amount", "is_cancelled", "is_returned_order", "is_service_fee", "is_non_item"}
    if qty_expr is not None:
        std.append(f"{qty_expr} as qty")
        taken.add("qty")
    for role, name in (("order_id", "order_id"), ("product", "product"),
                       ("product_id", "product_id"), ("customer", "customer"), ("branch", "branch"),
                       ("cost", "unit_cost")):
        if role in roles:
            std.append((f"{_num(roles[role])}" if role == "cost" else qid(roles[role])) + f" as {name}")
            taken.add(name)
    std += [
        f"{canc_ord} as is_cancelled",
        f"{ret_ord} as is_returned_order",
        f"{srv_fee_expr} as is_service_fee",
        f"{non_item_expr} as is_non_item",
    ]
    orig = [f"base.{qid(c)} as {qid('raw_' + c if c in taken else c)}" for c in base_cols]

    con.execute(f"create or replace view lines_all as select {', '.join(std + orig)} from base")
    where = ["dt is not null", "not coalesce(is_non_item, false)"]
    if opts.exclude_cancelled:
        where.append("not coalesce(is_cancelled, false)")
    if opts.exclude_orders_with_return:
        where.append("not coalesce(is_returned_order, false)")
    con.execute(f"create or replace view lines_valid as select * from lines_all where {' and '.join(where)}")
    if oid:
        q_agg = "sum(qty) as qty" if "qty" in roles else "null::double as qty"
        con.execute(f"""create or replace view orders_valid as
            select order_id, min(dt) as dt, sum(amount) as revenue, {q_agg}, count(*) as n_lines
            from lines_valid group by order_id""")


# ----------------------------------------------------------------------------
# 6. Business context
# ----------------------------------------------------------------------------
@dataclass
class BizContext:
    biz_id: str
    src_dir: str
    readme: str
    work_dir: Path
    cube_db: Path
    sql_db: Path
    raw_tables: list[str]
    fact_table: str = ""
    roles: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    lines_cols: list = field(default_factory=list)       # [(col, kind)] of lines_valid
    raw_cols: dict = field(default_factory=dict)          # table -> [(col, kind)]
    aux_views: list[str] = field(default_factory=list)
    schema_text: str = ""
    ok: bool = True


def _schema_text(con, ctx: BizContext, profs: dict[str, list[ColProfile]]) -> str:
    lines = [f"# Business {ctx.biz_id}", ""]
    if ctx.readme.strip():
        lines += ["## README", ctx.readme.strip()[:4500], ""]
    lines.append("## Raw tables (original data; sharded sheets/files auto-unioned; names normalized to snake_case)")
    for t in ctx.raw_tables:
        ps = profs.get(t, [])
        n = ps[0].n if ps else 0
        lines.append(f"### {t}  ({n} rows)")
        for p in ps[:50]:
            lines.append(f"- {p.col} [{p.dtype}] e.g. {', '.join(p.samples[:3])[:80]}")
        if len(ps) > 50:
            lines.append(f"- ... {len(ps) - 50} more columns")
    lines += [
        "", "## Prepared views (use these first)",
        f"fact table: {ctx.fact_table}; joined into flat table `base`.",
        "`lines_all`   : every line, standardized columns + flags is_cancelled, is_returned_order, is_service_fee, is_non_item",
        "`lines_valid` : rulebook-default valid lines (excludes cancelled orders, orders with any returned item, and non-item Voucher/shipping lines)",
        "`orders_valid`: one row per valid order (order_id, dt, revenue, qty, n_lines)" if "order_id" in ctx.roles else "",
        "standardized columns: dt (date), amount (VND, net of line discount), qty (converted to base unit when conversion rules exist), "
        "order_id, product, product_id, customer, branch, unit_cost (if cost column exists), is_service_fee (filter WHERE NOT is_service_fee when ranking products), + all original columns of `base`",
        f"detected roles: {json.dumps(ctx.roles, ensure_ascii=False)}",
    ]
    if ctx.aux_views:
        lines.append("auxiliary views: " + ", ".join(ctx.aux_views))
    lines.append("")
    if ctx.notes:
        lines.append("## Prep notes")
        lines += [f"- {n}" for n in ctx.notes]
    return "\n".join(x for x in lines if x is not None)


def prepare_business(biz_id: str, src_dir: str | Path, work_dir: str | Path,
                     llm: Optional[Callable[[str], str]] = None,
                     opts: Optional[RuleOptions] = None) -> BizContext:
    opts = opts or RuleOptions()
    work = Path(work_dir) / biz_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    cube_db, sql_db = work / "cube.duckdb", work / "sql.duckdb"
    con = duckdb.connect(str(cube_db))
    # Register strip_accents UDF in DuckDB if not built-in, or use unaccent helper via macro
    info = load_business(src_dir, con)
    ctx = BizContext(biz_id=biz_id, src_dir=str(src_dir), readme=info["readme"], work_dir=work,
                     cube_db=cube_db, sql_db=sql_db, raw_tables=info["tables"])
    if not ctx.raw_tables:
        ctx.ok = False
        ctx.schema_text = f"# Business {biz_id}\nNo tables could be loaded."
        con.close()
        shutil.copy(cube_db, sql_db)
        return ctx

    raw_profs = {t: profile_table(con, t) for t in ctx.raw_tables}
    ctx.raw_cols = {t: [(c, _kind(k)) for c, k in _cols(con, t)] for t in ctx.raw_tables}
    profs_for_schema = raw_profs
    try:
        # Ensure strip_accents works in DuckDB for Vietnamese characters including đ/Đ
        con.execute("create or replace macro norm_vn(s) as strip_accents(replace(replace(lower(trim(cast(s as varchar))), 'đ', 'd'), 'Đ', 'd'))")
        ctx.fact_table = choose_fact_table(con, ctx.raw_tables, raw_profs)
        fact_cols = {c for c, _ in _cols(con, ctx.fact_table)}
        others = [t for t in ctx.raw_tables if t != ctx.fact_table]
        build_base(con, ctx.fact_table, others, raw_profs, ctx.notes)
        base_prof = profile_table(con, "base")
        roles_l = refine_roles(base_prof, guess_roles_table(base_prof, allowed_fact_cols=fact_cols), ctx.readme, llm)
        issues = validate_roles(base_prof, roles_l)
        if issues:
            ctx.notes.append("role issues: " + "; ".join(issues[:6]))
        ctx.roles = {r: cs[0] for r, cs in roles_l.items() if cs}
        ctx.notes.append("all candidates per role: " + json.dumps(roles_l, ensure_ascii=False))
        ret_tables = [t for t in others if RETURN_COL_PAT.search(t)]
        has_uconv = _build_unit_conversion_table(con, ctx.raw_tables, ctx.readme, ctx.notes)
        build_views(con, ctx.roles, roles_l, {p.col: p for p in base_prof}, ret_tables, has_uconv, opts, ctx.notes)
        ctx.aux_views = _build_auxiliary_views(con, ctx.raw_tables, ctx.roles, ctx.notes)
        ctx.lines_cols = [(c, _kind(k)) for c, k in _cols(con, "lines_valid")]
    except Exception as e:  # keep raw tables usable via SQL even if the pipeline failed
        ctx.ok = False
        ctx.notes.append(f"PREP FAILED: {type(e).__name__}: {e}. Only raw tables are available.")
        print(f"[prep] {biz_id} failed: {e}")
    ctx.schema_text = _schema_text(con, ctx, profs_for_schema)
    con.close()
    shutil.copy(cube_db, sql_db)
    return ctx


if __name__ == "__main__":
    import sys
    c = prepare_business(Path(sys.argv[1]).name, sys.argv[1], "/tmp/dm_work")
    print(c.schema_text)
