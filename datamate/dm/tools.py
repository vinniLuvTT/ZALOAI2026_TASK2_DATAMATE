"""
Agent tools for ONE question on ONE business.

ToolBox holds state (final answer, trace). `ToolBox.functions()` returns plain Python
callables with type hints + docstrings; the same callables are given to Deep Agents
(wrapped as LangChain tools) and to the fallback loop (JSON schema is derived from them).
"""
from __future__ import annotations

import difflib
import json
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any, Callable, Optional

import duckdb

from . import timeutil
from .config import Config
from .cube_server import CubeServer
from .prep import BizContext, unaccent
from .verify import coerce

_FORBIDDEN = re.compile(
    r"\b(attach|detach|copy|install|load|export|import|create|insert|update|delete|drop|alter|call|pragma|"
    r"read_csv|read_parquet|read_json|read_text|glob|set|vacuum|checkpoint)\b", re.I)


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if v != v:
            return None
        r = round(v, 4)
        return int(r) if abs(r - round(r)) < 1e-9 else r
    if isinstance(v, (date,)):
        return v.isoformat()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _table_text(cols: list[str], rows: list[list[Any]], total: int, max_rows: int) -> str:
    out = [" | ".join(cols)]
    for r in rows[:max_rows]:
        out.append(" | ".join("" if c is None else str(_fmt(c)) for c in r))
    if total > max_rows:
        out.append(f"... ({total} rows total, showing {max_rows}; add LIMIT / aggregate)")
    else:
        out.append(f"({total} rows)")
    return "\n".join(out)


class SQLRunner:
    """Read-only DuckDB over the sql.duckdb copy; one shared connection, a cursor per call."""

    def __init__(self, db_path: Path, timeout_s: int):
        self.timeout_s = timeout_s
        try:
            self.con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})
        except Exception:
            self.con = duckdb.connect(str(db_path), read_only=True)

    def run(self, sql: str, max_rows: int) -> tuple[list[str], list[list[Any]], int]:
        cur = self.con.cursor()
        timer = threading.Timer(self.timeout_s, cur.interrupt)
        timer.start()
        try:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(max_rows + 1)
            total = len(rows)
            if total > max_rows:
                total = max_rows + 1 + len(cur.fetchall())
            return cols, [list(r) for r in rows[:max_rows]], total
        finally:
            timer.cancel()
            cur.close()


class ToolBox:
    def __init__(self, cfg: Config, ctx: BizContext, rulebook: str, calendar: Optional[timeutil.Calendar],
                 cube: Optional[CubeServer], cube_catalog: str, sql: SQLRunner, shared: dict):
        self.cfg, self.ctx, self.rulebook, self.cal = cfg, ctx, rulebook, calendar
        self.cube, self.cube_catalog, self.sql = cube, cube_catalog, sql
        self.shared = shared            # per-business cache (entity lists)
        self.final: Optional[dict] = None
        self.trace: list[dict] = []
        self.sql_history: list[str] = []
        self._unanswerable_warned: bool = False

    # ------------------------------------------------------------------ helpers
    def _log(self, tool: str, args: dict, result: str) -> str:
        self.trace.append({"tool": tool, "args": args, "result": result[:600]})
        return result[:6000]

    # ------------------------------------------------------------------ schema / rulebook
    def get_schema(self, table: str = "") -> str:
        """Return the data schema (tables, columns with sample values, prepared views, detected roles).
        Pass `table` (a raw table name) to see only that table; empty = everything."""
        txt = self.ctx.schema_text
        if table:
            m = re.search(rf"### {re.escape(table)}\b.*?(?=\n### |\n## |\Z)", txt, re.S)
            txt = m.group(0) if m else f"No raw table named {table!r}. Tables: {self.ctx.raw_tables}"
        return self._log("get_schema", {"table": table}, txt)

    def get_rulebook(self, offset: int = 0) -> str:
        """Return the rulebook (default meanings of terms like revenue, 'bán chạy', week, cost, Tết).
        Long rulebooks are paged: pass the offset printed at the end of the previous page."""
        size = 6000
        chunk = self.rulebook[offset: offset + size]
        if offset + size < len(self.rulebook):
            chunk += f"\n[... more; call get_rulebook(offset={offset + size})]"
        return self._log("get_rulebook", {"offset": offset}, chunk or "(rulebook empty)")

    def find_columns(self, keyword: str) -> str:
        """Search ALL tables for column names containing a keyword (accent-insensitive, e.g. 'gia von',
        'cost', 'han dung'). Use this before answering 'unanswerable' to confirm data is really missing."""
        kw = re.sub(r"[^a-z0-9]+", "_", unaccent(keyword).lower()).strip("_")
        words = [w for w in kw.split("_") if w]
        hits = []
        for t, cols in self.ctx.raw_cols.items():
            for c, k in cols:
                if kw and (kw in c or (words and all(w in c for w in words))):
                    hits.append(f"{t}.{c} [{k}]")
        res = "\n".join(hits[:40]) if hits else f"No column matches {keyword!r} in any table."
        return self._log("find_columns", {"keyword": keyword}, res)

    # ------------------------------------------------------------------ entities
    def _entities(self, column: str) -> list[tuple[str, int, str]]:
        key = f"ent:{column}"
        if key in self.shared:
            return self.shared[key]
        res: list[tuple[str, int, str]] = []
        cols = [c for c, _ in self.ctx.lines_cols]
        if column in cols:
            code = ", product_id" if column == "product" and "product_id" in cols else ""
            try:
                _, rows, _ = self.sql.run(
                    f'select {column}{code}, count(*) n from lines_all where {column} is not null group by all', 100000)
                for r in rows:
                    res.append((str(r[0]), int(r[-1]), str(r[1]) if code else ""))
            except Exception:
                pass
        self.shared[key] = res
        return res

    def find_entity(self, text: str, column: str = "product") -> str:
        """Fuzzy-match a name mentioned in the question (may be abbreviated / no accents / typo) to the exact
        spellings stored in the data. column: product | customer | branch. Use the returned exact value in filters."""
        ents = self._entities(column)
        if not ents:
            return self._log("find_entity", {"text": text, "column": column},
                             f"No `{column}` column in lines_all. Use find_columns / get_schema.")
        q = re.sub(r"\s+", " ", unaccent(text).lower()).strip()
        qt = set(q.split())
        scored = []
        for name, n, code in ents:
            u = re.sub(r"\s+", " ", unaccent(name).lower()).strip()
            ut = set(u.split())
            ratio = difflib.SequenceMatcher(None, q, u).ratio()
            contain = len(qt & ut) / max(len(qt), 1)
            sub = 1.0 if q and (q in u or u in q) else 0.0
            scored.append((max(ratio, 0.6 * contain + 0.4 * sub, sub * 0.95), name, n, code))
        scored.sort(key=lambda x: (-x[0], -x[2]))
        lines = [f"{name!r}" + (f" (code {code})" if code else "") + f" rows={n} score={s:.2f}"
                 for s, name, n, code in scored[:6]]
        return self._log("find_entity", {"text": text, "column": column}, "\n".join(lines))

    # ------------------------------------------------------------------ querying
    def sql_query(self, sql: str) -> str:
        """Run a read-only DuckDB SELECT. Prefer view `lines_valid` (valid sales lines: columns dt, amount, qty,
        order_id, product, product_id, customer, branch, ...) or `lines_all` (adds is_cancelled,
        is_returned_order) or any raw table (see get_schema). Returns up to ~40 rows. Use ORDER BY with a
        deterministic tie-break (metric DESC, then product_id/code ASC)."""
        s = sql.strip().rstrip(";")
        if ";" in s or not re.match(r"^(select|with|describe|show|summarize)\b", s, re.I) or _FORBIDDEN.search(s):
            return self._log("sql_query", {"sql": sql}, "SQL error: only a single read-only SELECT/WITH is allowed.")
        try:
            cols, rows, total = self.sql.run(s, self.cfg.max_rows)
        except Exception as e:
            return self._log("sql_query", {"sql": sql}, f"SQL error: {str(e)[:400]}")
        self.sql_history.append(s)
        return self._log("sql_query", {"sql": sql}, _table_text(cols, rows, total, self.cfg.max_rows))

    def cube_query(self, measures: list[str], dimensions: Optional[list[str]] = None,
                   time_dimension: Optional[str] = None, date_from: Optional[str] = None,
                   date_to: Optional[str] = None, granularity: Optional[str] = None,
                   filters: Optional[list[dict]] = None, order: Optional[dict] = None,
                   limit: int = 50) -> str:
        """Query the Cube semantic layer (cube `sales` = rulebook-valid lines; raw_<table> cubes = raw data).
        measures e.g. ["sales.revenue","sales.quantity"]; dimensions e.g. ["sales.product"];
        time_dimension e.g. "sales.dt" with date_from/date_to as 'YYYY-MM-DD' (inclusive) and optional
        granularity day|week|month|quarter|year; filters e.g.
        [{"member":"sales.product","operator":"equals","values":["Trà sữa"]}] (operators: equals, notEquals,
        contains, gt, gte, lt, lte, set, notSet); order e.g. {"sales.revenue":"desc"}. If it errors, use sql_query."""
        if self.cube is None:
            return self._log("cube_query", {}, "Cube is not available; use sql_query instead.")
        q: dict[str, Any] = {"measures": measures, "limit": int(limit)}
        if dimensions:
            q["dimensions"] = dimensions
        if time_dimension:
            td: dict[str, Any] = {"dimension": time_dimension}
            if date_from and date_to:
                td["dateRange"] = [date_from, date_to]
            if granularity:
                td["granularity"] = granularity
            q["timeDimensions"] = [td]
        if filters:
            q["filters"] = filters
        if order:
            q["order"] = [[k, v] for k, v in order.items()]
        res = self.cube.query(q)
        if "error" in res:
            return self._log("cube_query", q, f"Cube error: {res['error']} (fall back to sql_query)")
        data = res["data"]
        if not data:
            return self._log("cube_query", q, "(0 rows)")
        cols = list(data[0].keys())
        rows = []
        for d in data:
            row = []
            for c in cols:
                v = d[c]
                if c in measures and isinstance(v, str):
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                row.append(v)
            rows.append(row)
        return self._log("cube_query", q, _table_text(cols, rows, len(rows), self.cfg.max_rows))

    # ------------------------------------------------------------------ calendar
    def tet_window(self, year: int) -> str:
        """Tết window for the Tết that falls in solar `year`: 23 tháng Chạp (previous lunar year) to mùng 10
        tháng Giêng, as solar dates."""
        if not self.cal:
            return self._log("tet_window", {"year": year}, "Calendar not available.")
        w = self.cal.tet_window(year)
        res = f"{w[0].isoformat()} .. {w[1].isoformat()}" if w else f"No Tết window for {year} in calendar."
        return self._log("tet_window", {"year": year}, res)

    def lunar_to_solar(self, solar_year: int, lunar_month: int, lunar_day: int) -> str:
        """Solar date of a lunar date, e.g. mùng 1 tháng Giêng = (year,1,1); rằm tháng 8 = (year,8,15);
        for lunar months 11-12 the solar date just before the Tết of `solar_year`."""
        if not self.cal:
            return self._log("lunar_to_solar", {}, "Calendar not available.")
        d = self.cal.solar_from_lunar(solar_year, lunar_month, lunar_day)
        return self._log("lunar_to_solar", {"y": solar_year, "m": lunar_month, "d": lunar_day},
                         d.isoformat() if d else "not found")

    def solar_to_lunar(self, solar_date: str) -> str:
        """Lunar (month, day, is_leap) of a solar date 'YYYY-MM-DD'."""
        if not self.cal:
            return self._log("solar_to_lunar", {}, "Calendar not available.")
        r = self.cal.solar_to_lunar(timeutil._d(solar_date))
        return self._log("solar_to_lunar", {"d": solar_date}, f"lunar month {r[0]}, day {r[1]}, leap={r[2]}" if r else "not found")

    def shift_range_years(self, start: str, end: str, years: int = -1) -> str:
        """Shift an inclusive date range by whole years (years=-1 -> 'cùng kỳ năm ngoái')."""
        s, e = timeutil.shift_years(timeutil._d(start), years), timeutil.shift_years(timeutil._d(end), years)
        return self._log("shift_range_years", {"start": start, "end": end, "years": years},
                         f"{s.isoformat()} .. {e.isoformat()}")

    # ------------------------------------------------------------------ final answer
    def _check_premature_unanswerable(self, reason: str) -> Optional[str]:
        if self._unanswerable_warned:
            return None
        r_norm = unaccent(reason or "").lower()
        concept_patterns = [
            (r"gia von|loi nhuan|lai gop|cost|profit|gia nhap", r"gia_von|gia_nhap|don_gia_nhap|cost"),
            (r"han dung|het han|han su dung|hsd|expiry|lo hang|so lo", r"han_dung|han_su_dung|het_han|hsd|expiry|so_lo|ma_lo"),
            (r"ton kho|so luong ton|stock|inventory", r"ton_cuoi|ton_thuc|ton_kho|so_luong_ton|(^|_)ton($|_)"),
            (r"cong no|con no|chua tra|debt", r"con_no|cong_no|chua_tra|so_no|(^|_)no($|_)"),
            (r"chi phi|van hanh|expense", r"chi_phi|tien_chi|so_chi_phi"),
        ]
        found = []
        for r_pat, c_pat in concept_patterns:
            if re.search(r_pat, r_norm):
                for t, cols in self.ctx.raw_cols.items():
                    if re.search(c_pat, t):
                        found.append(f"table `{t}`")
                    for c, _ in cols:
                        if re.search(c_pat, c):
                            found.append(f"`{t}.{c}`")
        if found:
            self._unanswerable_warned = True
            uniq = list(dict.fromkeys(found))[:8]
            return (
                f"HOLD ON: You tried to submit unanswerable ({reason!r}), but this business HAS matching schema elements: "
                f"{', '.join(uniq)}. Query them with sql_query / get_schema first, or call submit_answer('unanswerable', ...) "
                f"again ONLY if you verified those columns cannot answer the question."
            )
        called_tools = {t["tool"] for t in self.trace}
        if not (called_tools & {"find_columns", "get_schema"}):
            self._unanswerable_warned = True
            return (
                "HOLD ON: Call find_columns or get_schema to verify the required column/table is truly missing before "
                "submitting unanswerable. If you already verified, call submit_answer('unanswerable', ...) again."
            )
        return None

    def submit_answer(self, answer_type: str, answer: Any = None, reason: str = "") -> str:
        """FINAL STEP. Submit the answer. answer_type: number | percent | string | date | boolean | list | table |
        unanswerable. number = plain number (VND integer, no units); percent = e.g. -12.45; string = exact name as
        stored in data; date = YYYY-MM-DD (or YYYY-MM for a month); list = JSON array in the requested order;
        table = {"columns":[...],"rows":[[...]]}; unanswerable = data cannot support the question (give `reason`)."""
        if str(answer_type).strip().lower() == "unanswerable":
            warn = self._check_premature_unanswerable(reason or str(answer or ""))
            if warn:
                return self._log("submit_answer", {"type": answer_type, "reason": reason}, warn)
        try:
            self.final = coerce(answer_type, answer, reason)
        except Exception as e:
            return self._log("submit_answer", {"type": answer_type}, f"INVALID: {e}. Fix and call submit_answer again.")
        return self._log("submit_answer", {"type": answer_type, "answer": answer}, "Answer recorded. Stop now.")

    # ------------------------------------------------------------------
    def functions(self) -> list[Callable]:
        fns = [self.get_schema, self.get_rulebook, self.find_columns, self.find_entity, self.sql_query]
        if self.cube is not None:
            fns.append(self.cube_query)
        fns += [self.tet_window, self.lunar_to_solar, self.solar_to_lunar, self.shift_range_years, self.submit_answer]
        return fns
