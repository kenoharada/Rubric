#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List


def read_csv_dict(path: Path, columns: List[str]) -> List[dict]:
    rows: List[dict] = []
    with path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append({k: v for k, v in zip(columns, row)})
    return rows


def dist(rows: List[dict], key: str) -> Dict[str, float]:
    c = Counter(r[key] for r in rows)
    n = len(rows)
    return {k: v / n for k, v in sorted(c.items())}


def tvd(d1: Dict[str, float], d2: Dict[str, float]) -> float:
    keys = sorted(set(d1) | set(d2))
    return 0.5 * sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


def main() -> None:
    parser = argparse.ArgumentParser(description="ets3のtrain/testサンプル分布差を確認")
    parser.add_argument("--root", default=".")
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out_dir", default="investigation/artifacts")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = root / "codes" / "dataset" / "ETS" / "data" / "text" / "index-training.csv"
    test_path = root / "codes" / "dataset" / "ETS" / "data" / "text" / "index-test.csv"

    train_rows = read_csv_dict(train_path, ["answer_id", "question_id", "nationality", "quality"])
    test_rows = read_csv_dict(test_path, ["answer_id", "question_id", "quality"])

    random.seed(args.seed)
    random.shuffle(train_rows)
    train_sample = train_rows[: args.train_size]

    # utils.prepare_ets_dataset() と同じく、seedはここで再初期化しない
    random.shuffle(test_rows)
    test_sample = test_rows[: args.test_size]

    full_train_quality = dist(train_rows, "quality")
    sample_train_quality = dist(train_sample, "quality")
    full_test_quality = dist(test_rows, "quality")
    sample_test_quality = dist(test_sample, "quality")

    full_train_qid = dist(train_rows, "question_id")
    sample_train_qid = dist(train_sample, "question_id")
    full_test_qid = dist(test_rows, "question_id")
    sample_test_qid = dist(test_sample, "question_id")

    result = {
        "seed": args.seed,
        "train_size": args.train_size,
        "test_size": args.test_size,
        "quality_distribution": {
            "full_train": full_train_quality,
            "sample_train": sample_train_quality,
            "full_test": full_test_quality,
            "sample_test": sample_test_quality,
            "tvd_train_sample_vs_test_sample": tvd(sample_train_quality, sample_test_quality),
            "tvd_full_train_vs_full_test": tvd(full_train_quality, full_test_quality),
        },
        "question_distribution": {
            "full_train": full_train_qid,
            "sample_train": sample_train_qid,
            "full_test": full_test_qid,
            "sample_test": sample_test_qid,
            "tvd_train_sample_vs_test_sample": tvd(sample_train_qid, sample_test_qid),
            "tvd_full_train_vs_full_test": tvd(full_train_qid, full_test_qid),
        },
    }

    out_json = out_dir / "ets3_split_distribution.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
