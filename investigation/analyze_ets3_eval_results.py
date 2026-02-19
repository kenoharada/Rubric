#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import cohen_kappa_score


def parse_rating(response: str) -> Optional[int]:
    if not isinstance(response, str):
        return None
    after = re.search(r"Rating:\s*(.*)", response)
    if not after:
        return None
    nums = re.findall(r"\d+", after.group(1))
    if not nums:
        return None
    return int(nums[-1])


def rating_to_quality(rating: Optional[int]) -> Optional[str]:
    mapping = {3: "high", 2: "medium", 1: "low"}
    return mapping.get(rating)


def quality_to_score(quality: Optional[str]) -> Optional[int]:
    mapping = {"high": 3, "medium": 2, "low": 1}
    return mapping.get(quality)


def md5_text(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def analyze_results_file(path: Path) -> dict:
    y_true: List[int] = []
    y_pred: List[int] = []
    parse_none = 0
    out_of_range = 0
    true_counter: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()

    for line in path.read_text().splitlines():
        data = json.loads(line)
        true_quality = data["annotated_score"]
        true_counter[true_quality] += 1

        rating = parse_rating(data["model_response"])
        if rating is None:
            parse_none += 1
            continue
        if rating not in (1, 2, 3):
            out_of_range += 1
            continue

        pred_quality = rating_to_quality(rating)
        pred_counter[pred_quality] += 1
        y_true.append(quality_to_score(true_quality))
        y_pred.append(quality_to_score(pred_quality))

    n = sum(true_counter.values())
    valid = len(y_true)
    qwk = None if valid == 0 else cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return {
        "n_samples": n,
        "valid_pairs": valid,
        "valid_ratio": 0.0 if n == 0 else valid / n,
        "parse_none": parse_none,
        "out_of_range": out_of_range,
        "qwk_from_results": qwk,
        "true_label_counts": dict(true_counter),
        "pred_label_counts": dict(pred_counter),
    }


def collect_runs(root: Path, dataset: str) -> List[dict]:
    rows: List[dict] = []
    base = root / "codes" / "evaluation_results" / dataset
    for model_dir in sorted(base.glob("*")):
        if not model_dir.is_dir():
            continue
        for run_dir in sorted(model_dir.glob("*")):
            results_path = run_dir / "results.jsonl"
            rubric_path = run_dir / "rubric.txt"
            qwk_path = run_dir / "qwk.txt"
            if not (results_path.exists() and rubric_path.exists() and qwk_path.exists()):
                continue
            analyzed = analyze_results_file(results_path)
            analyzed.update(
                {
                    "model": model_dir.name,
                    "run_name": run_dir.name,
                    "run_path": str(run_dir),
                    "rubric_hash": md5_text(rubric_path),
                    "qwk_reported": float(qwk_path.read_text().strip()),
                }
            )
            rows.append(analyzed)
    return rows


def make_noise_summary(rows: List[dict]) -> List[dict]:
    by_hash: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["rubric_hash"]].append(row)

    summary: List[dict] = []
    for rubric_hash, group in by_hash.items():
        qwks = [r["qwk_reported"] for r in group]
        summary.append(
            {
                "rubric_hash": rubric_hash,
                "count_runs": len(group),
                "qwk_min": min(qwks),
                "qwk_max": max(qwks),
                "qwk_range": max(qwks) - min(qwks),
                "runs": [r["run_path"] for r in group],
            }
        )
    summary.sort(key=lambda x: (-x["qwk_range"], -x["count_runs"]))
    return summary


def find_baseline_vs_optimized(rows: List[dict]) -> List[dict]:
    index = {(r["model"], r["run_name"]): r for r in rows}
    models = sorted({r["model"] for r in rows})
    seeds = ["expert", "simplest"]
    compared: List[dict] = []
    for model in models:
        for seed in seeds:
            no_key = (model, f"zero_shot_no_{seed}")
            opt_key = (model, f"zero_shot_base_{seed}_True_train100_iteration5_top3_bs4-8-12_mc1")
            if no_key not in index or opt_key not in index:
                continue
            baseline = index[no_key]
            optimized = index[opt_key]
            compared.append(
                {
                    "model": model,
                    "seed_prompt": seed,
                    "baseline_run": baseline["run_path"],
                    "optimized_run": optimized["run_path"],
                    "same_rubric_hash": baseline["rubric_hash"] == optimized["rubric_hash"],
                    "baseline_rubric_hash": baseline["rubric_hash"],
                    "optimized_rubric_hash": optimized["rubric_hash"],
                    "baseline_qwk": baseline["qwk_reported"],
                    "optimized_qwk": optimized["qwk_reported"],
                    "optimized_minus_baseline": optimized["qwk_reported"] - baseline["qwk_reported"],
                    "baseline_pred_counts": baseline["pred_label_counts"],
                    "optimized_pred_counts": optimized["pred_label_counts"],
                }
            )
    return compared


def main() -> None:
    parser = argparse.ArgumentParser(description="ets3評価結果のノイズと分布を集計")
    parser.add_argument("--dataset", default="ets3")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="investigation/artifacts")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(root, args.dataset)
    noise_summary = make_noise_summary(rows)
    baseline_vs_optimized = find_baseline_vs_optimized(rows)

    summary = {
        "dataset": args.dataset,
        "total_eval_runs": len(rows),
        "total_unique_rubrics": len({r["rubric_hash"] for r in rows}),
        "max_qwk_range_among_same_rubric": max((r["qwk_range"] for r in noise_summary), default=0.0),
    }

    payload = {
        "summary": summary,
        "runs": rows,
        "same_rubric_noise": noise_summary,
        "baseline_vs_optimized_zero_shot_true": baseline_vs_optimized,
    }

    out_json = out_dir / f"eval_analysis_{args.dataset}.json"
    out_csv = out_dir / f"baseline_vs_optimized_{args.dataset}.csv"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    fieldnames = [
        "model",
        "seed_prompt",
        "same_rubric_hash",
        "baseline_rubric_hash",
        "optimized_rubric_hash",
        "baseline_qwk",
        "optimized_qwk",
        "optimized_minus_baseline",
        "baseline_run",
        "optimized_run",
        "baseline_pred_counts",
        "optimized_pred_counts",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in baseline_vs_optimized:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {out_json}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
