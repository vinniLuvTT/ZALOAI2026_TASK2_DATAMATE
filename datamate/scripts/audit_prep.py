#!/usr/bin/env python
"""
Audit `prepare_business` across all businesses in one or more split directories.
Usage:
  python scripts/audit_prep.py ../train/data ../public_test/data
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dm.prep import prepare_business  # noqa: E402


def audit_split(split_dir: Path, work_root: Path) -> tuple[int, int]:
    if (split_dir / "README.md").exists() or any(split_dir.glob("*.xlsx")) or any(split_dir.glob("*.csv")) or any(split_dir.glob("*.sqlite")):
        biz_dirs = [split_dir]
    else:
        biz_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    ok_count = 0
    print(f"=== Auditing {split_dir} ({len(biz_dirs)} businesses) ===", flush=True)
    for b in biz_dirs:
        t0 = time.time()
        ctx = prepare_business(b.name, b, work_root)
        dt = time.time() - t0
        con = duckdb.connect(str(ctx.sql_db), read_only=True)
        try:
            la = con.execute("select count(*) from lines_all").fetchone()[0]
            lv, rev, min_dt, max_dt = con.execute(
                "select count(*), coalesce(sum(amount), 0), min(dt), max(dt) from lines_valid"
            ).fetchone()
            q_sum = con.execute("select coalesce(sum(qty), 0) from lines_valid").fetchone()[0] if "qty" in ctx.roles else None
        except Exception as e:
            la, lv, rev, min_dt, max_dt, q_sum = 0, 0, 0, None, None, f"ERR:{e}"
        con.close()
        passed = bool(ctx.ok and lv > 0 and rev > 0 and min_dt is not None)
        if passed:
            ok_count += 1
        status = "PASS" if passed else "FAIL"
        q_str = f"{q_sum:,.0f}" if isinstance(q_sum, (int, float)) else str(q_sum)
        print(
            f"[{status}] {b.name:8s} ({dt:4.1f}s) | fact={ctx.fact_table:24s} | "
            f"la={la:7d} lv={lv:7d} | rev={rev:16,.0f} | qty={q_str:>10s} | "
            f"dt={min_dt}..{max_dt} | roles={ctx.roles}",
            flush=True,
        )
        if not passed:
            for n in ctx.notes:
                print(f"   note: {n}", flush=True)
    print(f"Summary for {split_dir}: {ok_count}/{len(biz_dirs)} passed\n", flush=True)
    return ok_count, len(biz_dirs)


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or [Path("../train/data"), Path("../public_test/data")]
    work = Path("/tmp/dm_audit_work")
    total_ok, total_n = 0, 0
    for d in dirs:
        if d.exists():
            ok, n = audit_split(d, work)
            total_ok += ok
            total_n += n
    print(f"OVERALL: {total_ok}/{total_n} businesses passed")
    if total_ok < total_n:
        sys.exit(1)


if __name__ == "__main__":
    main()
