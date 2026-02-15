#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from inference import quality_to_score_for_qwk, rating_to_quality, parse_rating


# Map run_name (the directory name under each model) to a display label in the Method column.
# Example:
METHOD_LABELS = {
    "zero_shot_no_expert": "Human",
    # "few_shot_no_expert": "Human (few\_shot)",
    "zero_shot_base_expert_False_train100_iteration1_top3_bs4-8-12_mc1": "AutoCalibrate",
    # "few_shot_base_expert_False_train100_iteration1_top3_bs4-8-12_mc1": "AutoCalibrate (few\_shot)",
    "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc1": "\\textbf{Ours}",
    # "few_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc1": "\\textbf{Ours} (few\_shot)",
}

MODEL_LABELS = {
    "openai/gpt-5-mini": "GPT-5 mini",
    # "openai/gpt-4.1": "GPT-4.1",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen3-30B-A3B",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
}

DATASET_LABELS = {
    "asap_1": "ASAP P1",
    # "ets": "TOEFL11",
    "ets3": "TOEFL11",
    "ASAP2": "ASAP 2.0",
}
# METHOD_LABELS: Dict[str, str] = {
#     "zero_shot_no_expert": "zero_shot",
#     "few_shot_no_expert": "few_shot",
# }

KNOWN_VENDORS = (
    "openai",
    "google",
    "qwen",
    # "anthropic",
    # "meta",
    # "mistral",
)


def restore_model_name(model_dir_name: str) -> str:
    for vendor in KNOWN_VENDORS:
        prefix = vendor + "_"
        if model_dir_name.startswith(prefix):
            return model_dir_name.replace(prefix, vendor + "/", 1)
    return model_dir_name


def rankdata(values: Iterable[float]) -> List[float]:
    # Average ranks for ties, 1-based.
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])
    ranks = [0.0] * len(indexed)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in ry))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def macro_f1_score(y_true: List[int], y_pred: List[int]) -> Optional[float]:
    if not y_true:
        return None
    labels = sorted(set(y_true))
    f1_scores: List[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) if f1_scores else None


def load_spearman(results_path: Path, dataset: str) -> Optional[float]:
    y_true: List[float] = []
    y_pred: List[float] = []
    if not results_path.exists():
        return None
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
    return spearman_corr(y_true, y_pred)


def load_f1_mae(results_path: Path, dataset: str) -> Tuple[Optional[float], Optional[float]]:
    y_true: List[int] = []
    y_pred: List[int] = []
    if not results_path.exists():
        return None, None
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
            y_true.append(int(true_score))
            y_pred.append(int(pred_score))
    if not y_true:
        return None, None
    f1 = macro_f1_score(y_true, y_pred)
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
    return f1, mae


def load_metric(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def iter_runs(root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        for model_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
            model_name = restore_model_name(model_dir.name)
            for run_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                run_name = run_dir.name
                yield dataset, model_name, run_name, run_dir


def format_float(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def highlight_score(value: Optional[float], best: Optional[float], second: Optional[float]) -> str:
    rendered = format_float(value)
    if value is None:
        return rendered
    if best is not None and value == best:
        return f"\\textbf{{{rendered}}}"
    if second is not None and value == second:
        return f"\\underline{{{rendered}}}"
    return rendered


def best_and_second(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    return best_and_second_by_direction(values, higher_is_better=True)


def best_and_second_by_direction(
    values: List[Optional[float]], higher_is_better: bool
) -> Tuple[Optional[float], Optional[float]]:
    present = sorted({v for v in values if v is not None}, reverse=higher_is_better)
    if not present:
        return None, None
    if len(present) == 1:
        return present[0], None
    return present[0], present[1]


def build_table(
    rows: List[Tuple[str, str, Dict[str, Optional[float]], Dict[str, Tuple[Optional[float], Optional[float]]]]],
    dataset: str,
    metrics: List[str],
) -> str:
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    display_dataset = DATASET_LABELS.get(dataset, dataset)
    lines.append(f"\\caption{{Evaluation results for {display_dataset}}}")
    col_spec = "ll" + ("r" * len(metrics))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    header = ["LLM", "Rubric written by"]
    if "qwk" in metrics:
        header.append("QWK")
    if "accuracy" in metrics:
        header.append("Accuracy")
    if "spearman" in metrics:
        header.append("Spearman")
    if "f1_macro" in metrics:
        header.append("Macro-F1")
    if "mae" in metrics:
        header.append("MAE")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    last_model_name = None
    for model_name, method_label, metric_values, best_second in rows:
        if last_model_name is not None and model_name != last_model_name:
            lines.append("\\midrule")
        last_model_name = model_name
        display_model = MODEL_LABELS.get(model_name, model_name)
        row_cells = [display_model, method_label]
        for metric in metrics:
            value = metric_values.get(metric)
            best, second = best_second.get(metric, (None, None))
            row_cells.append(highlight_score(value, best, second))
        lines.append(" & ".join(row_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="evaluation_results", help="Root directory of evaluation results.")
    parser.add_argument("--dataset", action="append", help="Limit to a dataset (can be repeated).")
    parser.add_argument("--output", help="Write LaTeX tables to a file instead of stdout.")
    parser.add_argument(
        "--metrics",
        default="qwk,accuracy,spearman,f1_macro,mae",
        help="Comma-separated list of metrics to include: accuracy,qwk,spearman,f1_macro,mae",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    dataset_filter = set(args.dataset or ["asap_1", "ets3", "ASAP2"])
    grouped: Dict[str, Dict[str, Dict[str, Dict[str, Optional[float]]]]] = {}
    required_runs = list(METHOD_LABELS.keys())
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    allowed_metrics = {"accuracy", "qwk", "spearman", "f1_macro", "mae"}
    metric_directions = {"mae": False}
    unknown = [m for m in metrics if m not in allowed_metrics]
    if unknown:
        raise SystemExit(f"Unknown metrics: {', '.join(unknown)}")
    if not metrics:
        raise SystemExit("No metrics selected.")

    for dataset, model_name, run_name, run_dir in iter_runs(root):
        if dataset_filter and dataset not in dataset_filter:
            continue
        if run_name not in METHOD_LABELS:
            continue
        if model_name not in MODEL_LABELS:
            continue
        print(f"Processing: dataset={dataset}, model={model_name}, run={run_name}")
        resolved_run_dir = run_dir
        if not resolved_run_dir.exists():
            print(f"  Warning: run directory not found: {resolved_run_dir}")
            continue
        accuracy = load_metric(resolved_run_dir / "accuracy.txt")
        qwk = load_metric(resolved_run_dir / "qwk.txt")
        scoring_dataset = dataset
        spearman = load_spearman(
            resolved_run_dir / "results.jsonl",
            scoring_dataset,
        )
        f1_macro, mae = load_f1_mae(
            resolved_run_dir / "results.jsonl",
            scoring_dataset,
        )
        grouped.setdefault(dataset, {}).setdefault(model_name, {})[run_name] = {
            "accuracy": accuracy,
            "qwk": qwk,
            "spearman": spearman,
            "f1_macro": f1_macro,
            "mae": mae,
        }

    if not grouped:
        raise SystemExit("No runs found.")

    tables = []
    for dataset in sorted(grouped.keys()):
        rows_for_dataset: List[
            Tuple[str, str, Dict[str, Optional[float]], Dict[str, Tuple[Optional[float], Optional[float]]]]
        ] = []
        for model_name in sorted(grouped[dataset].keys()):
            runs = grouped[dataset][model_name]
            if any(r not in runs for r in required_runs):
                continue
            best_second: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
            for metric in metrics:
                values = [runs[r].get(metric) for r in required_runs]
                higher_is_better = metric_directions.get(metric, True)
                best, second = best_and_second_by_direction(values, higher_is_better)
                best_second[metric] = (best, second)
            for run_name in required_runs:
                method_label = METHOD_LABELS[run_name]
                metric_values = runs[run_name]
                rows_for_dataset.append(
                    (
                        model_name,
                        method_label,
                        metric_values,
                        dict(best_second),
                    )
                )
        if rows_for_dataset:
            tables.append(build_table(rows_for_dataset, dataset, metrics))

    if not tables:
        raise SystemExit("No runs found after filtering for required methods.")
    output_text = "\n\n".join(tables) + "\n"
    if args.output:
        Path(args.output).write_text(output_text)
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
