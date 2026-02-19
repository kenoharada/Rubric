#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


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
                if run_dir.is_dir():
                    run_dirs.append(run_dir)
    return run_dirs


def read_train_qwk(path: Path) -> List[float]:
    values: List[float] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        values.append(float(line))
    return values


def select_expected_rubric_path_by_train_qwk_max(run_dir: Path) -> Tuple[Optional[int], Optional[Path], Optional[str]]:
    train_qwk_path = run_dir / "train_qwk.txt"
    if not train_qwk_path.exists():
        return None, None, "missing_train_qwk"

    train_qwk = read_train_qwk(train_qwk_path)
    if not train_qwk:
        return None, None, "empty_train_qwk"

    # run全体を通して最大のQWKとなるstepを採用する。
    # 同値最大が複数ある場合は、最初の出現を採用する。
    # train_qwk[0]はstep0(initial_rubric), train_qwk[i]はstep_iのtop1に対応。
    best_step_index = train_qwk.index(max(train_qwk))
    if best_step_index == 0:
        source = run_dir / "initial_rubric.txt"
    else:
        source = run_dir / "best_rubric" / f"step_{best_step_index}_top1.txt"

    if not source.exists():
        return best_step_index, None, "missing_selected_source"

    return best_step_index, source, None


def process_run(run_dir: Path, apply: bool) -> Dict:
    best_rubric_txt = run_dir / "best_rubric.txt"
    selected_step, source_path, err = select_expected_rubric_path_by_train_qwk_max(run_dir)
    if err is not None:
        return {
            "run_dir": str(run_dir),
            "status": f"skipped_{err}",
            "selected_step": selected_step,
            "selected_source_path": str(source_path) if source_path else None,
            "best_rubric_txt": str(best_rubric_txt),
            "changed": False,
        }

    assert source_path is not None
    source_bytes = source_path.read_bytes()
    source_hash = md5_bytes(source_bytes)

    if best_rubric_txt.exists():
        current_hash = md5_bytes(best_rubric_txt.read_bytes())
    else:
        current_hash = None

    changed = current_hash != source_hash
    if apply and changed:
        best_rubric_txt.write_bytes(source_bytes)

    return {
        "run_dir": str(run_dir),
        "status": "updated" if (apply and changed) else ("would_update" if changed else "unchanged"),
        "selected_step": selected_step,
        "selected_source_path": str(source_path),
        "best_rubric_txt": str(best_rubric_txt),
        "changed": changed,
        "current_hash": current_hash,
        "selected_hash": source_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="best_rubric.txt を train_qwk 最大stepのrubricに揃える（実装の挙動ベース）",
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--dataset", default=None, help="例: ets3（未指定で全dataset）")
    parser.add_argument("--optimization_root", default="codes/optimization_results")
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
    skipped = [r for r in results if r["status"].startswith("skipped_")]

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
    out_json = out_dir / f"fix_best_rubric_by_train_qwk_max_{suffix}_{mode}.json"
    out_csv = out_dir / f"fix_best_rubric_by_train_qwk_max_{suffix}_{mode}.csv"

    payload = {"summary": summary, "results": results}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    fieldnames = [
        "run_dir",
        "status",
        "selected_step",
        "selected_source_path",
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
            print(f"- {row['run_dir']}  <- step={row['selected_step']}")
    elif would_update:
        print("\nwould update runs:")
        for row in would_update:
            print(f"- {row['run_dir']}  <- step={row['selected_step']}")
    if skipped:
        print("\nskipped runs:")
        for row in skipped:
            print(f"- {row['run_dir']}  ({row['status']})")
    print(f"\nJSON: {out_json}")
    print(f"CSV: {out_csv}")


if __name__ == "__main__":
    main()
