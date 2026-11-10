#!/usr/bin/env python
"""
DataMate CLI

  python run.py predict --split /data --out /result
  python run.py eval    --split train_dir --out out_eval [--businesses biz_001,biz_002] [--limit 30] [--harvest]
  python run.py bank    --train train_dir --out fewshot_bank.jsonl
  python run.py prep    --biz train_dir/data/biz_001           # inspect schema / roles / notes
  python run.py calendar --file train_dir/calendar.csv         # check Tết windows parse
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dm.config import Config


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict")
    p.add_argument("--split", default="/data")
    p.add_argument("--out", default="/result")

    e = sub.add_parser("eval")
    e.add_argument("--split", required=True)
    e.add_argument("--out", default="out_eval")
    e.add_argument("--businesses", default="")
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--harvest", action="store_true", help="store SQL of correct runs into the few-shot bank")
    e.add_argument("--no-bank", action="store_true")
    e.add_argument("--penalty", type=float, default=None)

    b = sub.add_parser("bank")
    b.add_argument("--train", required=True)
    b.add_argument("--out", default="fewshot_bank.jsonl")

    q = sub.add_parser("prep")
    q.add_argument("--biz", required=True)

    c = sub.add_parser("calendar")
    c.add_argument("--file", required=True)

    a = ap.parse_args()
    cfg = Config()

    if a.cmd == "prep":
        from dm.prep import prepare_business
        ctx = prepare_business(Path(a.biz).name, a.biz, cfg.work_dir)
        print(ctx.schema_text)
        return

    if a.cmd == "calendar":
        from dm.timeutil import Calendar
        cal = Calendar.load(a.file)
        if cal:
            for y in cal.years():
                w = cal.tet_window(y)
                if w:
                    print(y, w[0], "..", w[1])
        return

    if a.cmd == "bank":
        from dm.fewshot import FewShotBank
        root = Path(a.train)
        bank = FewShotBank()
        bank.add_train(root / "questions.jsonl", root / "answers.jsonl")
        bank.save(a.out)
        print(f"bank: {len(bank.items)} items -> {a.out}")
        return

    from dm.fewshot import FewShotBank
    from dm.pipeline import Runner, Split

    default_bank = Path(cfg.fewshot_bank) if cfg.fewshot_bank else cfg.project_root / "fewshot_bank.jsonl"

    if a.cmd == "predict":
        cfg.out_dir = Path(a.out)
        split = Split(a.split)
        bank = FewShotBank(default_bank) if default_bank.exists() else None
        Runner(cfg, split, cfg.out_dir, bank).run()
        return

    # ---- eval -----------------------------------------------------------
    from dm import scoring
    penalty = a.penalty if a.penalty is not None else cfg.wrong_penalty
    out = Path(a.out)
    split = Split(a.split)
    if not split.answers:
        sys.exit("eval needs answers.jsonl in the split")
    bank = None if a.no_bank else (FewShotBank(default_bank) if default_bank.exists() else None)
    biz = [x for x in a.businesses.split(",") if x]
    runner = Runner(cfg, split, out, bank, exclude_same_business_fewshot=True)
    recs = runner.run(businesses=biz or None, limit=a.limit or None)
    # include previously-run records when resuming
    if not recs and (out / "logs").exists():
        recs = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "logs").glob("*.json")]
    refs = {k: v for k, v in split.answers.items() if k in {r["id"] for r in recs}}
    rep = scoring.summarize(recs, refs, penalty)
    rep["calibration(threshold,PA,correct,wrong)"] = scoring.calibrate(recs, refs, penalty)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_business",)}, ensure_ascii=False, indent=2))
    if a.harvest:
        hb = FewShotBank(default_bank) if default_bank.exists() else FewShotBank(default_bank)
        n = 0
        for r in recs:
            ref = refs.get(r["id"])
            if ref and scoring.match(r["final"], ref) == "correct" and r.get("sql"):
                hb.add_harvest(r["id"], r["business_id"], r["question"], ref["type"], ref.get("answer"), r["sql"])
                n += 1
        hb.save(default_bank)
        print(f"harvested {n} correct SQL traces into {default_bank}")


if __name__ == "__main__":
    main()
