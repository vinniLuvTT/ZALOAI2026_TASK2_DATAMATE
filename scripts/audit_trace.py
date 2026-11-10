"""
scripts/audit_trace.py
Công cụ kiểm toán tự động file trace (VD: artifacts/trace_public.jsonl hoặc trace_val.jsonl)
theo từng doanh nghiệp (business_id) để phát hiện:
1. Doanh nghiệp có tỷ lệ ABSTAIN / lỗi thực thi bất thường (như trường hợp dev_012 trước đây).
2. Các câu hỏi vẫn rơi về intent tổng quát (revenue_total, order_count) dù có tín hiệu ràng buộc.
3. Các câu hỏi có nguy cơ sai kiểu dữ liệu (WRONG_TYPE).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.guard import ConstraintGuard


def main() -> None:
    parser = argparse.ArgumentParser(description="Kiểm toán chất lượng trace theo từng doanh nghiệp")
    parser.add_argument("--trace", type=Path, default=Path("artifacts/trace_public.jsonl"))
    args = parser.parse_args()

    if not args.trace.exists():
        print(f"Không tìm thấy file trace: {args.trace}")
        return

    guard = ConstraintGuard()
    records = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    by_biz = defaultdict(list)
    suspicious_records = []

    for r in records:
        biz_id = r.get("business_id", "unknown")
        by_biz[biz_id].append(r)
        q = r.get("question", "")
        verdict = guard.verify_plan(q, r.get("plan") or r)
        pred_type = r.get("type") or r.get("pred_type")
        if pred_type not in ("abstain", "unanswerable") and verdict.action != "ALLOW":
            suspicious_records.append((biz_id, r.get("id"), q, r.get("intent"), verdict.reason_code))

    print(f"=== BÁO CÁO KIỂM TOÁN TRACE ({len(records)} bản ghi) ===")
    print(f"{'Business':<12} | {'Total':>5} | {'Abstain':>7} | {'Unans':>5} | {'GenericIntent':>13} | {'UniqueAns':>9}")
    print("-" * 68)

    for biz_id in sorted(by_biz.keys()):
        items = by_biz[biz_id]
        type_counts = Counter((x.get("type") or x.get("pred_type") or "unknown") for x in items)
        generic_cnt = sum(
            1
            for x in items
            if str(x.get("intent") or "").lower() in guard.GENERIC_INTENTS
            and (x.get("type") or x.get("pred_type")) not in ("abstain", "unanswerable")
        )
        ans_set = {json.dumps(x.get("answer"), ensure_ascii=False, sort_keys=True) for x in items}
        print(
            f"{biz_id:<12} | {len(items):>5} | {type_counts['abstain']:>7} | "
            f"{type_counts['unanswerable']:>5} | {generic_cnt:>13} | {len(ans_set):>9}"
        )

    print(f"\n[!] Số câu trả lời có nguy cơ bỏ rơi ràng buộc / sai kiểu dữ liệu: {len(suspicious_records)}")
    for biz_id, qid, q, intent, reason in suspicious_records[:15]:
        print(f"  - [{biz_id} | {qid}] intent={intent} | lỗi={reason} | Q: {q}")


if __name__ == "__main__":
    main()
