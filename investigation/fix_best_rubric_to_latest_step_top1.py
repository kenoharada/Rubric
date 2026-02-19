#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TOP1_PATTERN = re.compile(r"^step_(\d+)_top1\.txt$")


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def find_latest_top1(best_rubric_dir: Path) -> Optional[Tuple[int, Path]]:
    candidates: List[Tuple[int, Path]] = []
    for p in best_rubric_dir.glob("step_*_top1.txt"):
        m = TOP1_PATTERN.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        candidates.append((step, p))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])


def iter_run_dirs(optimization_root: Path, dataset: Optional[str]) -> List[Path]:
    run_dirs: List[Path] = []
    dataset_dirs = [optimization_root / dataset] if dataset else sorted(optimization_root.glob("*"))
    for ds_dir in dataset_dirs:
        if not ds_dir.is_dir():
            continue
        for model_dir in sorted(ds_dir.glob("*")):
            if not model_dir.is_dir():
                continue
            for run_dir in sorted(model_dir.glob("*")):
                if not run_dir.is_dir():
                    continue
                if (run_dir / "best_rubric").is_dir():
                    run_dirs.append(run_dir)
    return run_dirs


def process_run(run_dir: Path, apply: bool) -> Dict:
    best_rubric_dir = run_dir / "best_rubric"
    best_rubric_txt = run_dir / "best_rubric.txt"

    latest = find_latest_top1(best_rubric_dir)
    if latest is None:
        return {
            "run_dir": str(run_dir),
            "status": "skipped_no_top1",
            "selected_step": None,
            "selected_top1_path": None,
            "best_rubric_txt": str(best_rubric_txt),
            "changed": False,
        }

    selected_step, selected_path = latest
    selected_bytes = selected_path.read_bytes()
    selected_hash = md5_bytes(selected_bytes)

    if best_rubric_txt.exists():
        current_bytes = best_rubric_txt.read_bytes()
        current_hash = md5_bytes(current_bytes)
    else:
        current_bytes = None
        current_hash = None

    changed = current_hash != selected_hash
    if apply and changed:
        best_rubric_txt.write_bytes(selected_bytes)

    return {
        "run_dir": str(run_dir),
        "status": "updated" if (apply and changed) else ("would_update" if changed else "unchanged"),
        "selected_step": selected_step,
        "selected_top1_path": str(selected_path),
        "best_rubric_txt": str(best_rubric_txt),
        "changed": changed,
        "current_hash": current_hash,
        "selected_hash": selected_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="best_rubric.txt を各runの最大stepの top1 に揃える",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default=None, help="例: ets3 (未指定で全dataset)")
    parser.add_argument(
        "--optimization_root",
        default="codes/optimization_results",
        help="optimization_results のルート",
    )
    parser.add_argument("--out_dir", default="investigation/artifacts")
    parser.add_argument("--apply", action="store_true", help="実際に書き換える")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    optimization_root = (root / args.optimization_root).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = iter_run_dirs(optimization_root, args.dataset)
    results = [process_run(run_dir, apply=args.apply) for run_dir in run_dirs]

    updated = [r for r in results if r["status"] == "updated"]
    would_update = [r for r in results if r["status"] == "would_update"]
    unchanged = [r for r in results if r["status"] == "unchanged"]
    skipped = [r for r in results if r["status"] == "skipped_no_top1"]

    summary = {
        "apply": args.apply,
        "dataset": args.dataset,
        "optimization_root": str(optimization_root),
        "total_runs": len(results),
        "updated_runs": len(updated),
        "would_update_runs": len(would_update),
        "unchanged_runs": len(unchanged),
        "skipped_runs": len(skipped),
    }

    suffix = args.dataset if args.dataset else "all"
    mode = "apply" if args.apply else "dryrun"
    out_json = out_dir / f"fix_best_rubric_latest_top1_{suffix}_{mode}.json"
    out_csv = out_dir / f"fix_best_rubric_latest_top1_{suffix}_{mode}.csv"

    payload = {"summary": summary, "results": results}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    fieldnames = [
        "run_dir",
        "status",
        "selected_step",
        "selected_top1_path",
        "best_rubric_txt",
        "changed",
        "current_hash",
        "selected_hash",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if updated:
        print("\nupdated runs:")
        for row in updated:
            print(f"- {row['run_dir']}  <- step_{row['selected_step']}_top1")
    elif would_update:
        print("\nwould update runs:")
        for row in would_update:
            print(f"- {row['run_dir']}  <- step_{row['selected_step']}_top1")
    print(f"\nJSON: {out_json}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()

