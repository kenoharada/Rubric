#!/usr/bin/env python3
"""Generate LaTeX tables comparing performance across different training sample sizes."""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from inference import quality_to_score_for_qwk, rating_to_quality, parse_rating
from sklearn.metrics import cohen_kappa_score


MODEL_LABELS = {
    "openai/gpt-5-mini": "GPT-5 mini",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen3-80B-A3B",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
}

DATASET_LABELS = {
    "asap_1": "ASAP",
    "ets3": "TOEFL11",
    "ASAP2": "ASAP 2.0",
}

KNOWN_VENDORS = ("openai", "google", "qwen")


def restore_model_name(model_dir_name: str) -> str:
    for vendor in KNOWN_VENDORS:
        prefix = vendor + "_"
        if model_dir_name.startswith(prefix):
            return model_dir_name.replace(prefix, vendor + "/", 1)
    return model_dir_name


def compute_qwk_from_jsonl(results_path: Path, dataset: str) -> Optional[float]:
    """Compute QWK from results.jsonl file."""
    if not results_path.exists():
        return None
    y_true, y_pred = [], []
    with results_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            true_quality = row.get("annotated_score")
            model_response = row.get("model_response")
            if model_response is None:
                continue
            rating = parse_rating(model_response)
            if rating is None:
                continue
            pred_quality = rating_to_quality(rating, dataset)
            true_score = quality_to_score_for_qwk(true_quality, dataset)
            pred_score = quality_to_score_for_qwk(pred_quality, dataset)
            if true_score is None or pred_score is None:
                continue
            y_true.append(true_score)
            y_pred.append(pred_score)
    if len(y_true) < 2:
        return None
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def load_metric(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def get_qwk(run_dir: Path, dataset: str) -> Optional[float]:
    """Get QWK from qwk.txt if available, otherwise compute from results.jsonl."""
    qwk = load_metric(run_dir / "qwk.txt")
    if qwk is not None:
        return qwk
    return compute_qwk_from_jsonl(run_dir / "results.jsonl", dataset)


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def highlight_best(value: Optional[float], best: Optional[float]) -> str:
    rendered = format_float(value)
    if value is None:
        return rendered
    if best is not None and abs(value - best) < 1e-9:
        return f"\\textbf{{{rendered}}}"
    return rendered


def build_table(
    data: Dict[str, Dict[str, Dict[int, Optional[float]]]],
    datasets: List[str],
    train_sizes: List[int],
    baselines: Dict[str, Dict[str, Optional[float]]],
) -> str:
    """Build a LaTeX table.

    data[dataset][model][train_size] = qwk
    baselines[dataset][model] = qwk (no optimization baseline)
    """
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\resizebox{\\columnwidth}{!}{%")

    # Columns: Dataset, LLM, Baseline, train_size_1, train_size_2, ...
    ncols = 2 + 1 + len(train_sizes)  # dataset + model + baseline + train_sizes
    col_spec = "ll" + "r" * (1 + len(train_sizes))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header = ["\\textbf{Dataset}", "\\textbf{LLM}", "\\textbf{Baseline}"]
    for ts in train_sizes:
        header.append(f"\\textbf{{$N={ts}$}}")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    model_order = [
        "google/gemini-3-flash-preview",
        "openai/gpt-5-mini",
        "qwen/qwen3-next-80b-a3b-instruct",
    ]

    for di, dataset in enumerate(datasets):
        if di > 0:
            lines.append("\\midrule")
        display_dataset = DATASET_LABELS.get(dataset, dataset)
        for mi, model in enumerate(model_order):
            if model not in data.get(dataset, {}):
                continue
            display_model = MODEL_LABELS.get(model, model)
            ds_cell = display_dataset if mi == 0 else ""

            # Find best QWK among train_sizes for this model
            qwk_values = [data[dataset][model].get(ts) for ts in train_sizes]
            valid_values = [v for v in qwk_values if v is not None]
            best_val = max(valid_values) if valid_values else None

            baseline_qwk = baselines.get(dataset, {}).get(model)
            cells = [ds_cell, display_model, format_float(baseline_qwk)]
            for ts in train_sizes:
                qwk = data[dataset][model].get(ts)
                cells.append(highlight_best(qwk, best_val))
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\caption{Effect of training sample size ($N$) on test QWK. "
                  "Baseline uses the human expert rubric without optimization. "
                  "Bold indicates the best QWK among training sizes for each model.}")
    lines.append("\\label{tab:train_size}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate train-size comparison tables.")
    parser.add_argument("--root", default="evaluation_results",
                        help="Root directory of evaluation results.")
    parser.add_argument("--dataset", action="append",
                        help="Limit to a dataset (can be repeated).")
    parser.add_argument("--train_sizes", type=int, nargs="+", default=[10, 20, 50, 100],
                        help="Training sizes to compare.")
    parser.add_argument("--output", help="Write LaTeX table to a file instead of stdout.")
    parser.add_argument(
        "--run_pattern",
        default="zero_shot_base_expert_True_train{train_size}_iteration5_top3_bs4-8-12_mc4",
        help="Run name pattern with {train_size} placeholder.",
    )
    parser.add_argument(
        "--baseline_pattern",
        default="zero_shot_no_expert",
        help="Baseline run name (no optimization).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    dataset_filter = set(args.dataset) if args.dataset else {"asap_1", "ASAP2"}
    train_sizes = sorted(args.train_sizes)

    # Collect data: data[dataset][model][train_size] = qwk
    data: Dict[str, Dict[str, Dict[int, Optional[float]]]] = {}
    baselines: Dict[str, Dict[str, Optional[float]]] = {}

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        if dataset not in dataset_filter:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = restore_model_name(model_dir.name)
            if model_name not in MODEL_LABELS:
                continue

            # Load baseline
            baseline_dir = model_dir / args.baseline_pattern
            if baseline_dir.exists():
                baseline_qwk = get_qwk(baseline_dir, dataset)
                baselines.setdefault(dataset, {})[model_name] = baseline_qwk
                print(f"Baseline: {dataset}/{model_name} -> QWK={format_float(baseline_qwk)}")

            # Load train-size results
            for ts in train_sizes:
                run_name = args.run_pattern.format(train_size=ts)
                run_dir = model_dir / run_name
                if not run_dir.exists():
                    print(f"  Not found: {run_dir}")
                    continue
                qwk = get_qwk(run_dir, dataset)
                data.setdefault(dataset, {}).setdefault(model_name, {})[ts] = qwk
                print(f"  {dataset}/{model_name}/train{ts} -> QWK={format_float(qwk)}")

    if not data:
        raise SystemExit("No results found.")

    # Report missing evaluations
    missing = []
    for dataset in dataset_filter:
        for model_dir_name in ["google_gemini-3-flash-preview", "openai_gpt-5-mini", "qwen_qwen3-next-80b-a3b-instruct"]:
            model_name = restore_model_name(model_dir_name)
            for ts in train_sizes:
                qwk = data.get(dataset, {}).get(model_name, {}).get(ts)
                if qwk is None:
                    run_name = args.run_pattern.format(train_size=ts)
                    opt_rubric = root.parent / "optimization_results" / dataset / model_dir_name / f"base_expert_True_train{ts}_iteration5_top3_bs4-8-12_mc4" / "best_rubric.txt"
                    has_rubric = opt_rubric.exists()
                    missing.append((dataset, model_name, ts, has_rubric))

    if missing:
        print(f"\n[WARNING] {len(missing)} missing evaluation(s):")
        for dataset, model, ts, has_rubric in missing:
            rubric_status = "rubric EXISTS (needs evaluate.py)" if has_rubric else "rubric MISSING (needs inference.py first)"
            print(f"  {dataset}/{model}/train{ts}: {rubric_status}")

    datasets = [d for d in ["asap_1", "ASAP2", "ets3"] if d in data]
    table = build_table(data, datasets, train_sizes, baselines)

    if args.output:
        Path(args.output).write_text(table + "\n")
        print(f"\nWritten to {args.output}")
    else:
        print("\n" + table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
