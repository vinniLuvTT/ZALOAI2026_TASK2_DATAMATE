"""Business package loader with README parser, multi-file/multi-sheet combiner, and disk cache."""

from __future__ import annotations

import datetime as dt
import os
import pickle
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.excel_fast import read_xlsx_dataframes
from src.temporal import parse_flexible_date
from src.value_parser import extract_year_from_name, parse_vn_number


@dataclass
class UnitConversionRule:
    product_name: str
    large_unit: str
    factor: float
    base_unit: str


@dataclass
class BusinessPackage:
    split: str
    business_id: str
    data_dir: Path
    readme_text: str
    readme_end_date: Optional[dt.date] = None
    inline_unit_rules: List[UnitConversionRule] = field(default_factory=list)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    sheet_tables: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)


def parse_readme_metadata(
    readme_text: str,
) -> Tuple[Optional[dt.date], List[UnitConversionRule]]:
    """Extract `readme_end_date` and inline unit conversion rules from `README.md`."""
    end_date: Optional[dt.date] = None
    m_end = re.search(
        r"Dữ liệu có đến hết\s+(\d{1,2}/\d{1,2}/\d{4})", readme_text, flags=re.IGNORECASE
    )
    if m_end:
        end_date = parse_flexible_date(m_end.group(1))

    unit_rules: List[UnitConversionRule] = []
    # Pattern: "  - Coca Cola 330ml: 1 lốc = 6 lon"
    for line in readme_text.splitlines():
        m_u = re.match(
            r"^\s*-\s+(.+?):\s*1\s+(.+?)\s*=\s*(\d+(?:[.,]\d+)?)\s+(.+?)\s*$", line
        )
        if m_u:
            pname = m_u.group(1).strip()
            l_unit = m_u.group(2).strip()
            factor = float(m_u.group(3).replace(",", "."))
            b_unit = m_u.group(4).strip()
            unit_rules.append(
                UnitConversionRule(
                    product_name=pname,
                    large_unit=l_unit,
                    factor=factor,
                    base_unit=b_unit,
                )
            )
    # Also check dev_004 "Ống PVC Bình Minh D32 cây 4m, ... bán theo bó, 1 bó = 10 cây"
    if "1 bó = 10 cây" in readme_text:
        for pvc in (
            "Ống PVC Bình Minh D32 cây 4m",
            "Ống PVC Bình Minh D50 cây 4m",
            "Ống PVC Bình Minh D90 cây 4m",
        ):
            unit_rules.append(
                UnitConversionRule(
                    product_name=pvc, large_unit="bó", factor=10.0, base_unit="cây"
                )
            )

    return end_date, unit_rules


def _read_csv_robust(csv_path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=enc, dtype=object)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception:
            continue
    return pd.DataFrame()


def _read_sqlite_ro(sqlite_path: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    uri = f"file:{sqlite_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        t_names = [r[0] for r in cur.fetchall()]
        for t in t_names:
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
            out[t] = df
    finally:
        conn.close()
    return out


def _LogicalBaseName(filename: str) -> str:
    """Group multi-part files into a logical table name.

    Examples:
    - 'SoBanHang_1.xlsx', 'SoBanHang_2026.xlsx' -> 'SoBanHang'
    - 'DanhSachHoaDonChiTiet_1.xlsx', 'DanhSachHoaDonChiTiet_2.xlsx' -> 'DanhSachHoaDonChiTiet'
    - 'BanHang_T1_2025.xlsx' .. 'BanHang_T5_2026.xlsx' -> 'BanHang'
    - 'BanHang_Thang01_2025.xlsx' -> 'BanHang'
    - 'BanHang_2025.xlsx', 'BanHang_2026.xlsx' -> 'BanHang'
    - 'DonHang_Lo_1.xlsx', 'DonHang_Lo_2.xlsx' -> 'DonHang_Lo'
    """
    stem = Path(filename).stem
    # BanHang_T1_2025 or BanHang_Thang01_2025
    stem = re.sub(r"_T(?:hang)?\d{1,2}_202\d$", "", stem, flags=re.IGNORECASE)
    # _2025, _2026, _1, _2
    stem = re.sub(r"_(?:202\d|\d+)$", "", stem)
    return stem


def load_business_package(
    split_dir: Path | str,
    business_id: str,
    cache_dir: Optional[Path | str] = "artifacts/cache",
    use_cache: bool = True,
) -> BusinessPackage:
    """Load a business data package with optional pickle caching."""
    split_path = Path(split_dir)
    split_name = split_path.name
    biz_dir = split_path / "data" / business_id
    if not biz_dir.exists():
        raise FileNotFoundError(f"Business directory not found: {biz_dir}")

    if use_cache and cache_dir:
        cpath = Path(cache_dir) / f"pkg_{split_name}_{business_id}.pkl"
        if cpath.exists():
            try:
                with open(cpath, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass

    readme_path = biz_dir / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    end_date, unit_rules = parse_readme_metadata(readme_text)

    pkg = BusinessPackage(
        split=split_name,
        business_id=business_id,
        data_dir=biz_dir,
        readme_text=readme_text,
        readme_end_date=end_date,
        inline_unit_rules=unit_rules,
    )

    grouped_parts: Dict[str, List[pd.DataFrame]] = {}

    for fname in sorted(os.listdir(biz_dir)):
        if fname == "README.md" or fname.startswith("."):
            continue
        fpath = biz_dir / fname
        if not fpath.is_file():
            continue

        file_year = extract_year_from_name(fname)
        logical_base = _LogicalBaseName(fname)

        if fname.endswith(".csv"):
            df = _read_csv_robust(fpath)
            df["_source_file"] = fname
            df["_sheet_name"] = ""
            df["_sheet_year"] = file_year
            grouped_parts.setdefault(logical_base, []).append(df)

        elif fname.endswith(".sqlite"):
            sql_tables = _read_sqlite_ro(fpath)
            for tname, df in sql_tables.items():
                df["_source_file"] = fname
                df["_sheet_name"] = tname
                df["_sheet_year"] = None
                pkg.tables[tname] = df

        elif fname.endswith(".xlsx"):
            sheet_dfs = read_xlsx_dataframes(fpath)
            pkg.sheet_tables[fname] = sheet_dfs

            # Check if this workbook is a multi-entity container (like DuLieuBanHang.xlsx,
            # DuLieuXuat.xlsx, Bao_cao_ban_hang.xlsx, QuanLy.xlsx, SoNo.xlsx, BanHang_TongHop_*.xlsx)
            # OR a monthly-partitioned single-entity table (where sheet names are months like 'T01_2025', 'Thang 02 2025', '04.2025')
            snames = list(sheet_dfs.keys())
            if len(snames) == 1:
                s0 = snames[0]
                df = sheet_dfs[s0].copy()
                s_year = extract_year_from_name(s0) or file_year
                df["_source_file"] = fname
                df["_sheet_name"] = s0
                df["_sheet_year"] = s_year
                grouped_parts.setdefault(logical_base, []).append(df)
            else:
                for sname, sdf in sheet_dfs.items():
                    if sdf.empty:
                        continue
                    df = sdf.copy()
                    s_year = extract_year_from_name(sname) or file_year
                    df["_source_file"] = fname
                    df["_sheet_name"] = sname
                    df["_sheet_year"] = s_year

                    # Is this a monthly sheet (e.g. 'Thang 02 2025', 'T04_2025', '04.2025', 'T7-2025')?
                    is_month_sheet = bool(
                        re.match(
                            r"^(?:tháng|thang|t)?\s*\d{1,2}[\s._-]+202\d$",
                            sname.strip(),
                            flags=re.IGNORECASE,
                        )
                    )
                    if is_month_sheet or sname == "Sổ bán hàng":
                        grouped_parts.setdefault(logical_base, []).append(df)
                    else:
                        grouped_parts.setdefault(sname, []).append(df)

    for tname, df_list in grouped_parts.items():
        if len(df_list) == 1:
            pkg.tables[tname] = df_list[0]
        else:
            pkg.tables[tname] = pd.concat(df_list, ignore_index=True)

    if use_cache and cache_dir:
        cpath = Path(cache_dir) / f"pkg_{split_name}_{business_id}.pkl"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cpath, "wb") as f:
                pickle.dump(pkg, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return pkg
