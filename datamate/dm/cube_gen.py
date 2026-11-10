"""
Generate Cube YAML models from a prepared BizContext. Pure templating - the LLM never writes YAML.

Cubes produced
  sales        over view `lines_valid`  (rulebook-valid lines)
               measures: revenue, quantity, order_count, line_count, avg_unit_price
               dimensions: every column of lines_valid, `dt` is the time dimension
  raw_<table>  over each raw table (count + sum of numeric non-id columns, all columns as dimensions)
Returns a catalog text that is shown to the agent so it knows the exact member names.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from .prep import BizContext

_ID_LIKE = re.compile(r"(^|_)(id|ma|code|sku|stt|so_hd|ma_)($|_)|^ma_|_id$|_code$")
_MAX_DIMS = 60


def _ident(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    return s if re.match(r"[a-zA-Z_]", s) else "c_" + s


def _q(col: str) -> str:
    return '{CUBE}."%s"' % col.replace('"', '""')


def _dim(col: str, kind: str) -> dict:
    d = {"name": _ident(col), "sql": _q(col)}
    if kind == "time":
        d["type"] = "time"
    elif kind == "number":
        d["type"] = "number"
    elif kind == "boolean":
        d["type"] = "boolean"
    else:
        d["type"] = "string"
    return d


def build_models(ctx: BizContext) -> tuple[list[dict], str]:
    cubes: list[dict] = []
    catalog: list[str] = []

    if ctx.ok and ctx.lines_cols:
        cols = dict(ctx.lines_cols)
        dims = [_dim(c, k) for c, k in ctx.lines_cols[:_MAX_DIMS]]
        # dt as a proper time dimension (view column is DATE)
        measures = [
            {"name": "revenue", "sql": _q("amount"), "type": "sum"},
            {"name": "line_count", "type": "count"},
        ]
        if "qty" in cols:
            measures.append({"name": "quantity", "sql": _q("qty"), "type": "sum"})
        if "order_id" in cols:
            measures.append({"name": "order_count", "sql": _q("order_id"), "type": "count_distinct"})
        if "product" in cols:
            measures.append({"name": "product_count", "sql": _q("product"), "type": "count_distinct"})
        if "customer" in cols:
            measures.append({"name": "customer_count", "sql": _q("customer"), "type": "count_distinct"})
        cubes.append({"name": "sales", "sql_table": "lines_valid", "dimensions": dims, "measures": measures})
        catalog.append("cube `sales` (valid lines: no cancelled orders, no orders with returned items; amount after line discount)")
        catalog.append("  measures: " + ", ".join(f"sales.{m['name']}" for m in measures))
        catalog.append("  time dimension: sales.dt")
        catalog.append("  dimensions: " + ", ".join(f"sales.{d['name']}" for d in dims))

    for t in ctx.raw_tables:
        cols = ctx.raw_cols.get(t, [])[:_MAX_DIMS]
        if not cols:
            continue
        name = "raw_" + _ident(t)
        dims = [_dim(c, k) for c, k in cols]
        measures = [{"name": "row_count", "type": "count"}]
        for c, k in cols:
            if k == "number" and not _ID_LIKE.search(c):
                measures.append({"name": "sum_" + _ident(c), "sql": _q(c), "type": "sum"})
                measures.append({"name": "max_" + _ident(c), "sql": _q(c), "type": "max"})
                measures.append({"name": "min_" + _ident(c), "sql": _q(c), "type": "min"})
        cubes.append({"name": name, "sql_table": f'"{t}"', "dimensions": dims, "measures": measures[:40]})
        catalog.append(f"cube `{name}` (raw table {t}, no filtering)")
        catalog.append("  measures: " + ", ".join(f"{name}.{m['name']}" for m in measures[:40]))
        catalog.append("  dimensions: " + ", ".join(f"{name}.{d['name']}" for d in dims))
        for d in dims:
            if d["type"] == "time":
                catalog.append(f"  time dimension: {name}.{d['name']}")
    return cubes, "\n".join(catalog)


def write_model(ctx: BizContext, model_dir: Path) -> str:
    model_dir.mkdir(parents=True, exist_ok=True)
    for old in model_dir.glob("*.yml"):
        old.unlink()
    cubes, catalog = build_models(ctx)
    (model_dir / "model.yml").write_text(
        yaml.safe_dump({"cubes": cubes}, allow_unicode=True, sort_keys=False, width=200), encoding="utf-8")
    return catalog
