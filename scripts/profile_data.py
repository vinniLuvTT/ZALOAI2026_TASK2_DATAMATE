#!/usr/bin/env python3
"""Profile all train and public_test businesses in parallel and build data catalogs."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loader import load_business_package
from src.rule_engine import RulebookEngine


def _profile_one(args: Tuple[str, str, str, str]) -> Dict[str, Any]:
    split_dir, biz_id, cache_dir, catalog_dir = args
    pkg = load_business_package(split_dir, biz_id, cache_dir=cache_dir, use_cache=True)
    re_eng = RulebookEngine()
    files = sorted(os.listdir(pkg.data_dir))
    domains = re_eng.detect_domains(pkg.readme_text, files)

    sources = []
    for tname, df in pkg.tables.items():
        cols_info = []
        for col in df.columns:
            if col.startswith("_"):
                continue
            s = df[col]
            null_cnt = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum())
            null_ratio = round(null_cnt / len(df), 4) if len(df) else 0.0
            non_null = s.dropna()
            non_null = non_null[non_null.astype(str).str.strip() != ""]
            examples = [str(x) for x in non_null.head(3).tolist()]
            cols_info.append(
                {
                    "name": col,
                    "null_ratio": null_ratio,
                    "examples": examples,
                }
            )
        sources.append(
            {
                "table": tname,
                "row_count": int(len(df)),
                "column_count": len(cols_info),
                "columns": cols_info,
            }
        )

    cat = {
        "split": pkg.split,
        "business_id": biz_id,
        "domains": domains,
        "readme_end_date": pkg.readme_end_date.isoformat() if pkg.readme_end_date else None,
        "inline_unit_rules_count": len(pkg.inline_unit_rules),
        "files": [f for f in files if f != "README.md"],
        "sources": sources,
    }

    out_path = Path(catalog_dir) / pkg.split / f"{biz_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cat, f, ensure_ascii=False, indent=2)

    return cat


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DataMate datasets.")
    parser.add_argument("--splits", nargs="+", default=["train", "public_test"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", default="artifacts/cache")
    parser.add_argument("--catalog-dir", default="artifacts/data_catalog")
    args = parser.parse_args()

    t0 = time.time()
    tasks: List[Tuple[str, str, str, str]] = []
    for sp in args.splits:
        data_root = Path(sp) / "data"
        if not data_root.exists():
            continue
        for biz_id in sorted(os.listdir(data_root)):
            if (data_root / biz_id).is_dir():
                tasks.append((sp, biz_id, args.cache_dir, args.catalog_dir))

    with mp.Pool(processes=min(args.workers, max(1, len(tasks)))) as pool:
        results = pool.map(_profile_one, tasks)

    elapsed = round(time.time() - t0, 2)
    summary = {
        "total_businesses_profiled": len(results),
        "elapsed_seconds": elapsed,
        "by_split": {},
    }
    for sp in args.splits:
        sp_res = [r for r in results if r["split"] == sp]
        summary["by_split"][sp] = {
            "count": len(sp_res),
            "total_tables": sum(len(r["sources"]) for r in sp_res),
            "total_rows": sum(sum(s["row_count"] for s in r["sources"]) for r in sp_res),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
