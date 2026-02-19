import argparse
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from inference import parse_rating, quality_to_rating, rating_to_quality


def _infer_dataset(rows):
    has_int = False
    has_quality = False
    max_rating = None
    for row in rows:
        raw = row.get("annotated_score")
        if isinstance(raw, int):
            has_int = True
        elif isinstance(raw, str) and raw.lower() in {"high", "medium", "low"}:
            has_quality = True
        if "model_response" in row:
            parsed = parse_rating(row["model_response"])
            if parsed is not None:
                max_rating = parsed if max_rating is None else max(max_rating, parsed)
    if has_int and not has_quality:
        return "asap"
    if has_quality:
        if max_rating is not None and max_rating <= 3:
            return "ets3"
        return "ets"
    return "ets"


def _normalize_annotated_rating(raw, dataset):
    if dataset.startswith("asap"):
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str) and raw.strip().isdigit():
            return int(raw.strip())
        return None
    if isinstance(raw, str):
        return quality_to_rating(raw.lower(), dataset)
    return None


def _is_correct(model_rating, annotated_rating):
    if model_rating is None or annotated_rating is None:
        return None
    if isinstance(annotated_rating, list):
        return model_rating in annotated_rating
    return model_rating == annotated_rating


def _label_for_plot(dataset, rating):
    if rating is None:
        return None
    if dataset.startswith("asap"):
        return str(rating)
    return rating_to_quality(rating, dataset)


def _save_plot(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_correctness(counts, out_path):
    labels = ["correct", "incorrect", "parse_failed", "unknown"]
    values = [counts.get(label, 0) for label in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#2ca02c", "#d62728", "#7f7f7f", "#9467bd"])
    ax.set_title("Correctness Counts")
    ax.set_ylabel("Count")
    _save_plot(fig, out_path)


def _plot_distribution(gold_labels, pred_labels, out_path):
    all_labels = sorted(set(gold_labels + pred_labels), key=lambda x: (len(x), x))
    gold_counts = Counter(gold_labels)
    pred_counts = Counter(pred_labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(all_labels))
    ax.bar([i - 0.2 for i in x], [gold_counts.get(l, 0) for l in all_labels], width=0.4, label="gold")
    ax.bar([i + 0.2 for i in x], [pred_counts.get(l, 0) for l in all_labels], width=0.4, label="pred")
    ax.set_xticks(list(x))
    ax.set_xticklabels(all_labels)
    ax.set_title("Rating Distribution")
    ax.set_ylabel("Count")
    ax.legend()
    _save_plot(fig, out_path)


def _plot_confusion(labels, gold_labels, pred_labels, out_path):
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for g, p in zip(gold_labels, pred_labels):
        if g in index and p in index:
            matrix[index[g]][index[p]] += 1
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues", origin="lower")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Confusion Matrix")
    for i, row in enumerate(matrix):
        for j, val in enumerate(row):
            ax.text(j, i, str(val), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    _save_plot(fig, out_path)


def _order_labels(labels):
    if all(label.isdigit() for label in labels):
        return [str(v) for v in sorted(int(l) for l in labels)]
    if set(labels) <= {"low", "medium", "high"}:
        order = ["low", "medium", "high"]
        return [label for label in order if label in labels]
    return sorted(labels)


def _process_file(input_path, dataset_override=None, output_dir=None):
    with open(input_path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    dataset = dataset_override or _infer_dataset(rows)
    out_dir = output_dir or os.path.dirname(input_path)
    os.makedirs(out_dir, exist_ok=True)

    parsed_rows = []
    counts = Counter()
    gold_plot = []
    pred_plot = []

    for row in rows:
        model_rating = parse_rating(row.get("model_response", "")) if row.get("model_response") else None
        raw_annotated = row.get("annotated_score")
        annotated_rating = _normalize_annotated_rating(raw_annotated, dataset)
        is_correct = _is_correct(model_rating, annotated_rating)

        if model_rating is None:
            label = "parse_failed"
        elif is_correct is True:
            label = "correct"
        elif is_correct is False:
            label = "incorrect"
        else:
            label = "unknown"
        counts[label] += 1

        parsed = dict(row)
        parsed["annotated_score_raw"] = raw_annotated
        parsed["annotated_score"] = annotated_rating
        parsed["model_rating"] = model_rating
        parsed["correctness_label"] = label
        parsed_rows.append(parsed)

        gold_label = _label_for_plot(dataset, annotated_rating if not isinstance(annotated_rating, list) else None)
        pred_label = _label_for_plot(dataset, model_rating)
        if dataset.startswith("asap"):
            if gold_label is not None and pred_label is not None:
                gold_plot.append(gold_label)
                pred_plot.append(pred_label)
        else:
            gold_quality = raw_annotated.lower() if isinstance(raw_annotated, str) else None
            pred_quality = rating_to_quality(model_rating, dataset) if model_rating is not None else None
            if gold_quality and pred_quality:
                gold_plot.append(gold_quality)
                pred_plot.append(pred_quality)

    parsed_path = os.path.join(out_dir, "results_parsed.jsonl")
    with open(parsed_path, "w") as f:
        for row in parsed_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    _plot_correctness(counts, os.path.join(out_dir, "correctness_counts.png"))
    if gold_plot and pred_plot:
        _plot_distribution(gold_plot, pred_plot, os.path.join(out_dir, "rating_distribution.png"))
        labels = _order_labels(sorted(set(gold_plot + pred_plot)))
        _plot_confusion(labels, gold_plot, pred_plot, os.path.join(out_dir, "confusion_matrix.png"))

    total = sum(counts.values())
    accuracy = counts.get("correct", 0) / total if total else 0.0
    return {
        "input": input_path,
        "output_dir": out_dir,
        "dataset": dataset,
        "parsed": len(parsed_rows),
        "accuracy": accuracy,
        "counts": dict(counts),
        "parsed_path": parsed_path,
    }


def _discover_results(root_dir):
    root = Path(root_dir)
    return sorted(str(p) for p in root.rglob("results.jsonl"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="*", help="Path(s) to results.jsonl")
    parser.add_argument("--dataset", default=None, help="Dataset name (asap_1, ets, ets3, etc.)")
    parser.add_argument("--output_dir", default=None, help="Output directory for parsed results and plots")
    parser.add_argument("--root", default=".", help="Root directory to search when no --input is given")
    args = parser.parse_args()

    inputs = args.input
    if not inputs:
        inputs = _discover_results(args.root)
        if not inputs:
            raise FileNotFoundError(f"No results.jsonl found under: {args.root}")

    summaries = []
    for input_path in inputs:
        summary = _process_file(
            input_path,
            dataset_override=args.dataset,
            output_dir=args.output_dir,
        )
        summaries.append(summary)
        print(
            f"{summary['input']} dataset={summary['dataset']} "
            f"parsed={summary['parsed']} accuracy={summary['accuracy']:.4f} "
            f"counts={summary['counts']}"
        )

    if len(summaries) > 1:
        total_parsed = sum(s["parsed"] for s in summaries)
        total_correct = sum(s["counts"].get("correct", 0) for s in summaries)
        overall_acc = total_correct / total_parsed if total_parsed else 0.0
        print(f"total_files={len(summaries)} total_parsed={total_parsed} overall_accuracy={overall_acc:.4f}")


if __name__ == "__main__":
    main()
