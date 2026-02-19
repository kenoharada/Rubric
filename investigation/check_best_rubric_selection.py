#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_train_qwk(path: Path) -> List[float]:
    values: List[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(float(line))
    return values


def parse_top_candidate_qwk(path: Path) -> Dict[int, Dict[int, float]]:
    pattern = re.compile(r"Step (\d+), Top (\d+) Candidate, Training QWK: ([\-0-9\.eE]+)")
    step_to_top_qwk: Dict[int, Dict[int, float]] = {}
    for line in path.read_text().splitlines():
        m = pattern.search(line)
        if not m:
            continue
        step = int(m.group(1))
        topk = int(m.group(2))
        qwk = float(m.group(3))
        step_to_top_qwk.setdefault(step, {})[topk] = qwk
    return step_to_top_qwk


def analyze_run(run_dir: Path) -> dict:
    train_qwk_path = run_dir / "train_qwk.txt"
    train_score_path = run_dir / "train_score_top_candidates.txt"
    best_rubric_path = run_dir / "best_rubric.txt"
    initial_rubric_path = run_dir / "initial_rubric.txt"

    train_qwk = read_train_qwk(train_qwk_path)
    best_qwk = max(train_qwk)
    best_idx = train_qwk.index(best_qwk)  # 0=initial, 1..=step

    step_to_top_qwk = parse_top_candidate_qwk(train_score_path)
    if best_idx == 0:
        expected_path = initial_rubric_path
        top1_qwk = best_qwk
        top3_qwk = None
    else:
        expected_path = run_dir / "best_rubric" / f"step_{best_idx}_top1.txt"
        top1_qwk = step_to_top_qwk.get(best_idx, {}).get(1)
        top3_qwk = step_to_top_qwk.get(best_idx, {}).get(3)

    actual_hash = md5_file(best_rubric_path)
    expected_hash = md5_file(expected_path) if expected_path.exists() else None

    matched_step_files: List[str] = []
    for step_file in sorted((run_dir / "best_rubric").glob("step_*_top*.txt")):
        if md5_file(step_file) == actual_hash:
            matched_step_files.append(step_file.name)

    return {
        "run": str(run_dir),
        "best_step_index_in_train_qwk": best_idx,
        "best_train_qwk": best_qwk,
        "expected_best_rubric_path": str(expected_path),
        "expected_hash": expected_hash,
        "actual_best_rubric_path": str(best_rubric_path),
        "actual_hash": actual_hash,
        "is_expected_best_saved": expected_hash == actual_hash,
        "actual_hash_matches_step_files": matched_step_files,
        "top1_qwk_at_best_step": top1_qwk,
        "top3_qwk_at_best_step": top3_qwk,
        "top1_minus_top3_gap_at_best_step": None
        if (top1_qwk is None or top3_qwk is None)
        else (top1_qwk - top3_qwk),
    }


def discover_runs(root: Path, dataset: str) -> List[Path]:
    runs: List[Path] = []
    base = root / "codes" / "optimization_results" / dataset
    if not base.exists():
        return runs
    for run_dir in sorted(base.glob("*/*")):
        if not run_dir.is_dir():
            continue
        required = [
            run_dir / "train_qwk.txt",
            run_dir / "train_score_top_candidates.txt",
            run_dir / "best_rubric.txt",
            run_dir / "initial_rubric.txt",
            run_dir / "best_rubric",
        ]
        if all(p.exists() for p in required):
            runs.append(run_dir)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="best_rubric保存の整合性チェック")
    parser.add_argument("--dataset", default="ets3")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out_dir", default="investigation/artifacts")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(root, args.dataset)
    results = [analyze_run(run_dir) for run_dir in runs]

    mismatch = [r for r in results if not r["is_expected_best_saved"]]
    gaps = [
        r["top1_minus_top3_gap_at_best_step"]
        for r in mismatch
        if r["top1_minus_top3_gap_at_best_step"] is not None
    ]
    summary = {
        "dataset": args.dataset,
        "total_runs": len(results),
        "mismatch_runs": len(mismatch),
        "mismatch_ratio": 0.0 if not results else len(mismatch) / len(results),
        "avg_top1_minus_top3_gap_at_best_step": None if not gaps else sum(gaps) / len(gaps),
    }

    out_json = out_dir / f"best_rubric_selection_{args.dataset}.json"
    out_csv = out_dir / f"best_rubric_selection_{args.dataset}.csv"

    payload = {"summary": summary, "runs": results}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    fieldnames = [
        "run",
        "best_step_index_in_train_qwk",
        "best_train_qwk",
        "expected_best_rubric_path",
        "actual_best_rubric_path",
        "is_expected_best_saved",
        "actual_hash_matches_step_files",
        "top1_qwk_at_best_step",
        "top3_qwk_at_best_step",
        "top1_minus_top3_gap_at_best_step",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON: {out_json}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()

