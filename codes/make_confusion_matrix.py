#!/usr/bin/env python3
"""Generate confusion matrix heatmaps: Human Expert Rubric vs Ours for Qwen3-80B-A3B."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import parse_rating, rating_to_quality, quality_to_score_for_qwk

HUMAN_RUN = "zero_shot_no_expert"
OURS_RUN = "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4"
MODEL_DIR = "qwen_qwen3-next-80b-a3b-instruct"
# MODEL_DIR = "openai_gpt-5-mini"

DATASETS = [
    {"key": "asap_1", "label": "ASAP P1", "scores": list(range(1, 7)), "tick_labels": None},
    {"key": "ASAP2", "label": "ASAP 2.0", "scores": list(range(1, 7)), "tick_labels": None},
    {"key": "ets3", "label": "TOEFL11", "scores": [1, 2, 3], "tick_labels": ["Low", "Med", "High"]},
]


def scoring_dataset_for_run(dataset: str, run_name: str) -> str:
    return dataset


def load_predictions(results_path: Path, dataset: str, run_name: str) -> Tuple[List[int], List[int]]:
    """Load true and predicted scores from a results.jsonl file."""
    scoring_ds = scoring_dataset_for_run(dataset, run_name)
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
            pred_quality = rating_to_quality(rating, scoring_ds)
            true_score = quality_to_score_for_qwk(true_quality, scoring_ds)
            pred_score = quality_to_score_for_qwk(pred_quality, scoring_ds)
            if true_score is None or pred_score is None:
                continue
            y_true.append(int(true_score))
            y_pred.append(int(pred_score))
    return y_true, y_pred


def build_confusion_matrix(y_true: List[int], y_pred: List[int], scores: List[int]) -> np.ndarray:
    n = len(scores)
    idx = {s: i for i, s in enumerate(scores)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti, pi = idx.get(t), idx.get(p)
        if ti is not None and pi is not None:
            cm[ti][pi] += 1
    return cm


def plot_heatmap(ax, cm, scores, tick_labels, title, show_ylabel=True, show_xlabel=True):
    """Plot a single confusion matrix heatmap on the given axes."""
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal", origin="lower")

    labels = tick_labels if tick_labels else [str(s) for s in scores]
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(scores)))
    ax.set_yticklabels(labels)

    if show_xlabel:
        ax.set_xlabel("Predicted Score")
    if show_ylabel:
        ax.set_ylabel("True Score")

    ax.set_title(title, fontsize=10)

    # Annotate cells with counts
    for i in range(len(scores)):
        for j in range(len(scores)):
            count = cm[i, j]
            if count > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                fontsize = 10 if len(scores) > 4 else 12
                ax.text(j, i, str(count), ha="center", va="center",
                        color=color, fontsize=fontsize)

    return im


def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix heatmaps.")
    parser.add_argument("--root", default="evaluation_results",
                        help="Root directory of evaluation results.")
    parser.add_argument("--output", default="../paper_latex/figures/confusion_matrices.pdf",
                        help="Output file path.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # Layout: 2 rows (Human Expert, Ours) x 3 columns (ASAP P1, ASAP 2.0, TOEFL11)
    fig = plt.figure(figsize=(10, 5.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[6, 6, 3.5],
                           hspace=0.4, wspace=0.35)

    methods = [
        (HUMAN_RUN, "Human Expert"),
        (OURS_RUN, "Ours"),
    ]

    last_im = None
    for row_idx, (run_name, method_label) in enumerate(methods):
        for col_idx, ds in enumerate(DATASETS):
            results_path = root / ds["key"] / MODEL_DIR / run_name / "results.jsonl"
            if not results_path.exists():
                alt_ds = scoring_dataset_for_run(ds["key"], run_name)
                if alt_ds != ds["key"]:
                    results_path = root / alt_ds / MODEL_DIR / run_name / "results.jsonl"

            if not results_path.exists():
                print(f"WARNING: {results_path} not found, skipping")
                continue

            y_true, y_pred = load_predictions(results_path, ds["key"], run_name)
            cm = build_confusion_matrix(y_true, y_pred, ds["scores"])

            title = f"{ds['label']} — {method_label}"
            ax = fig.add_subplot(gs[row_idx, col_idx])
            last_im = plot_heatmap(
                ax, cm, ds["scores"], ds.get("tick_labels"), title,
                show_ylabel=(col_idx == 0),
                show_xlabel=(row_idx == 1),
            )

            # Print summary
            total = cm.sum()
            correct = np.trace(cm)
            print(f"{ds['label']} / {method_label}: {correct}/{total} correct ({correct/total:.1%})")

    # Add colorbar
    if last_im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
        fig.colorbar(last_im, cax=cbar_ax, label="Row-normalized proportion")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
