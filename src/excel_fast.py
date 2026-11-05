"""High-performance C-backed (lxml) XLSX reader and table header detector.

Reads multi-sheet Excel workbooks ~10x faster than `openpyxl` by parsing
the OpenXML zip streams directly with `lxml.etree`, while preserving exact
column positions, sheet names, banner-skipping, and merged header handling.
"""

from __future__ import annotations

import posixpath
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lxml import etree
import pandas as pd


NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col_letters_to_index(cell_ref: str) -> int:
    """Convert Excel cell reference like 'A1', 'BC23' to 0-based column index."""
    idx = 0
    for ch in cell_ref:
        if "A" <= ch <= "Z":
            idx = idx * 26 + (ord(ch) - ord("A") + 1)
        elif "a" <= ch <= "z":
            idx = idx * 26 + (ord(ch) - ord("a") + 1)
        else:
            break
    return max(0, idx - 1)


def read_xlsx_raw_sheets(xlsx_path: Path | str) -> Dict[str, List[List[Any]]]:
    """Read all sheets of an XLSX file into `{sheet_name: list_of_rows}`."""
    path = Path(xlsx_path)
    sheets_out: Dict[str, List[List[Any]]] = {}

    with zipfile.ZipFile(path, "r") as zf:
        namelist = set(zf.namelist())

        # 1. Shared strings
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in namelist:
            ss_root = etree.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.findall(f"{NS_MAIN}si"):
                shared_strings.append("".join(si.itertext()))

        # 2. Relationships for workbook sheets
        rel_map: Dict[str, str] = {}
        rels_path = "xl/_rels/workbook.xml.rels"
        if rels_path in namelist:
            rels_root = etree.fromstring(zf.read(rels_path))
            for rel in rels_root:
                rid = rel.get("Id")
                target = rel.get("Target")
                if rid and target:
                    if target.startswith("/"):
                        norm_target = target.lstrip("/")
                    else:
                        norm_target = posixpath.normpath(posixpath.join("xl", target))
                    rel_map[rid] = norm_target

        # 3. Sheet list in workbook order
        sheet_entries: List[Tuple[str, str]] = []
        if "xl/workbook.xml" in namelist:
            wb_root = etree.fromstring(zf.read("xl/workbook.xml"))
            sheets_el = wb_root.find(f"{NS_MAIN}sheets")
            if sheets_el is not None:
                for sh in sheets_el.findall(f"{NS_MAIN}sheet"):
                    sname = sh.get("name") or "Sheet"
                    rid = sh.get(f"{NS_REL}id")
                    target = rel_map.get(rid or "", "")
                    if target in namelist:
                        sheet_entries.append((sname, target))

        if not sheet_entries:
            for name in sorted(namelist):
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                    sheet_entries.append((Path(name).stem, name))

        # 4. Parse each worksheet XML
        for sname, target in sheet_entries:
            ws_root = etree.fromstring(zf.read(target))
            sheet_data = ws_root.find(f"{NS_MAIN}sheetData")
            if sheet_data is None:
                sheets_out[sname] = []
                continue

            rows_list: List[List[Any]] = []
            for row_el in sheet_data:
                row_vals: List[Any] = []
                for c_el in row_el:
                    ref = c_el.get("r")
                    col_idx = _col_letters_to_index(ref) if ref else len(row_vals)
                    while len(row_vals) <= col_idx:
                        row_vals.append(None)

                    ctype = c_el.get("t")
                    val: Any = None
                    if ctype == "inlineStr":
                        is_el = c_el.find(f"{NS_MAIN}is")
                        if is_el is not None:
                            val = "".join(is_el.itertext())
                    else:
                        v_el = c_el.find(f"{NS_MAIN}v")
                        if v_el is not None and v_el.text is not None:
                            raw_txt = v_el.text
                            if ctype == "s":
                                s_idx = int(raw_txt)
                                val = (
                                    shared_strings[s_idx]
                                    if 0 <= s_idx < len(shared_strings)
                                    else raw_txt
                                )
                            elif ctype == "b":
                                val = raw_txt.strip() in ("1", "true", "True")
                            elif ctype in ("str", "e"):
                                val = raw_txt
                            else:
                                # Numeric ('n' or None)
                                try:
                                    if "." in raw_txt or "e" in raw_txt or "E" in raw_txt:
                                        fval = float(raw_txt)
                                        val = int(fval) if fval.is_integer() else fval
                                    else:
                                        val = int(raw_txt)
                                except ValueError:
                                    val = raw_txt
                    row_vals[col_idx] = val
                rows_list.append(row_vals)
            sheets_out[sname] = rows_list

    return sheets_out


def detect_header_row(rows: List[List[Any]]) -> Tuple[int, List[str]]:
    """Detect the true table header row in the first 10 rows of a sheet.

    Skips:
    - Empty rows
    - Single-cell report title banners (e.g., 'DANH SÁCH HÓA ĐƠN', 'SỔ CHI TIẾT BÁN HÀNG')
    - Report metadata rows (e.g., 'Kỳ báo cáo: toàn bộ | Kênh bán: Tất cả', 'Trạng thái đơn hàng | Đặt hàng (Đã duyệt)')
    - Sparse group header rows (e.g., ('Hàng', None, None, 'Tiền', None, None, 'Đơn'))
    """
    for idx, r in enumerate(rows[:10]):
        vals = list(r)
        while vals and (vals[-1] is None or str(vals[-1]).strip() == ""):
            vals.pop()
        if len(vals) < 2:
            continue
        # Skip sparse group header rows that contain internal Nones
        if any(v is None or str(v).strip() == "" for v in vals):
            continue
        if all(isinstance(v, str) for v in vals):
            if len(vals) == 2 and (
                "Đặt hàng" in vals[1]
                or "Kênh bán:" in vals[1]
                or "Từ ngày" in vals[0]
            ):
                continue
            return idx, [v.strip() for v in vals]
    if rows:
        return 0, [str(c or f"col_{i}").strip() for i, c in enumerate(rows[0])]
    return 0, []


def rows_to_dataframe(rows: List[List[Any]]) -> pd.DataFrame:
    """Convert raw worksheet rows into a clean pandas DataFrame using `detect_header_row`."""
    if not rows:
        return pd.DataFrame()
    h_idx, headers = detect_header_row(rows)
    if not headers:
        return pd.DataFrame()
    ncols = len(headers)
    data_rows: List[List[Any]] = []
    for r in rows[h_idx + 1 :]:
        if not r or all(c is None or str(c).strip() == "" for c in r):
            continue
        if len(r) < ncols:
            padded = list(r) + [None] * (ncols - len(r))
        else:
            padded = list(r[:ncols])
        data_rows.append(padded)
    return pd.DataFrame(data_rows, columns=headers)


def read_xlsx_dataframes(xlsx_path: Path | str) -> Dict[str, pd.DataFrame]:
    """Read all sheets of an XLSX file into `{sheet_name: pd.DataFrame}`."""
    raw_sheets = read_xlsx_raw_sheets(xlsx_path)
    return {sname: rows_to_dataframe(rows) for sname, rows in raw_sheets.items()}
