"""
scripts/split_train_val.py
Tách tập Train (30 doanh nghiệp) thành 22 doanh nghiệp Dev và 8 doanh nghiệp Held-out Validation
theo phương pháp phân tầng (Stratified Business-Level Split) dựa trên định dạng file (.csv, .xlsx, .sqlite)
và nhóm câu hỏi, đồng thời trích xuất mẫu 70 câu Public Test (5 câu x 14 doanh nghiệp) để kiểm định độc lập.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def detect_business_stratum(biz_dir: Path) -> str:
    """Xác định tầng (stratum) của doanh nghiệp dựa trên định dạng dữ liệu vật lý."""
    if not biz_dir.exists():
        return "unknown"
    exts = set()
    for p in biz_dir.rglob("*"):
        if p.is_file():
            suf = p.suffix.lower()
            if suf in {".csv", ".xlsx", ".xls", ".sqlite", ".db", ".sqlite3"}:
                exts.add(".sqlite" if suf in {".sqlite", ".db", ".sqlite3"} else (".xlsx" if suf in {".xlsx", ".xls"} else ".csv"))
    if not exts:
        return "other"
    return "+".join(sorted(exts))


def stratified_business_split(
    biz_ids: List[str],
    train_dir: Path,
    val_count: int = 8,
    seed: int = 42,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Chia danh sách business_id theo tầng định dạng dữ liệu để tránh lệch phân phối."""
    rng = random.Random(seed)
    strata: Dict[str, List[str]] = defaultdict(list)
    biz_stratum_map: Dict[str, str] = {}

    for biz_id in sorted(biz_ids):
        candidates = [
            train_dir / biz_id,
            train_dir / "businesses" / biz_id,
            train_dir / "data" / biz_id,
        ]
        biz_path = next((p for p in candidates if p.exists()), train_dir / biz_id)
        stratum = detect_business_stratum(biz_path)
        strata[stratum].append(biz_id)
        biz_stratum_map[biz_id] = stratum

    total = len(biz_ids)
    val_biz: List[str] = []
    remaining_pool: List[str] = []

    for stratum, members in sorted(strata.items()):
        shuffled = list(members)
        rng.shuffle(shuffled)
        quota = max(1, round(len(shuffled) * val_count / total)) if len(shuffled) > 1 else 0
        val_biz.extend(shuffled[:quota])
        remaining_pool.extend(shuffled[quota:])

    rng.shuffle(remaining_pool)
    while len(val_biz) < val_count and remaining_pool:
        val_biz.append(remaining_pool.pop(0))
    while len(val_biz) > val_count:
        remaining_pool.append(val_biz.pop())

    val_set = set(val_biz)
    dev_biz = sorted([b for b in biz_ids if b not in val_set])
    return dev_biz, sorted(val_biz), biz_stratum_map


def sample_public_audit_questions(
    public_questions_file: Path,
    per_business: int = 5,
    seed: int = 42,
) -> List[dict]:
    """Lấy mẫu đều qua 14 doanh nghiệp public_test, ưu tiên phủ đa dạng tín hiệu ràng buộc."""
    if not public_questions_file.exists():
        return []
    rng = random.Random(seed)
    questions = [
        json.loads(line)
        for line in public_questions_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_biz: Dict[str, List[dict]] = defaultdict(list)
    for q in questions:
        by_biz[q["business_id"]].append(q)

    sampled: List[dict] = []
    for biz_id in sorted(by_biz.keys()):
        pool = list(by_biz[biz_id])
        rng.shuffle(pool)
        sampled.extend(pool[:per_business])
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified Business-Level Split cho DataMate")
    parser.add_argument("--train-dir", type=Path, default=Path("train"))
    parser.add_argument("--public-dir", type=Path, default=Path("public_test"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--val-businesses", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    questions_path = args.train_dir / "questions.jsonl"
    answers_path = args.train_dir / "answers.jsonl"

    if not questions_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file câu hỏi train tại: {questions_path}")

    questions = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    answers_by_id: Dict[str, dict] = {}
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                answers_by_id[rec["id"]] = rec

    all_biz = sorted({q["business_id"] for q in questions})
    dev_biz, val_biz, stratum_map = stratified_business_split(
        all_biz, args.train_dir, val_count=args.val_businesses, seed=args.seed
    )

    dev_set: Set[str] = set(dev_biz)
    val_set: Set[str] = set(val_biz)
    assert not (dev_set & val_set), "Lỗi: Tập Dev và Val bị trùng lặp business_id!"

    for split_name, target_set in [("train_dev", dev_set), ("train_val", val_set)]:
        out_dir = args.artifacts_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        split_q = [q for q in questions if q["business_id"] in target_set]
        split_a = [answers_by_id[q["id"]] for q in split_q if q["id"] in answers_by_id]

        (out_dir / "questions.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in split_q) + "\n",
            encoding="utf-8",
        )
        if split_a:
            (out_dir / "answers.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in split_a) + "\n",
                encoding="utf-8",
            )

    manifest = {
        "seed": args.seed,
        "total_businesses": len(all_biz),
        "dev_businesses": dev_biz,
        "val_businesses": val_biz,
        "strata": stratum_map,
    }
    (args.artifacts_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    pub_q_path = args.public_dir / "questions.jsonl"
    sampled_pub = sample_public_audit_questions(pub_q_path, per_business=5, seed=args.seed)
    if sampled_pub:
        (args.artifacts_dir / "public_audit_sample.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in sampled_pub) + "\n",
            encoding="utf-8",
        )

    print(f"[OK] Đã tách {len(dev_biz)} Dev businesses và {len(val_biz)} Held-out Val businesses.")
    print(f"[OK] Manifest lưu tại: {args.artifacts_dir / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
