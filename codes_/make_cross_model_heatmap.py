#!/usr/bin/env python3
"""Generate heatmap figure for cross-model rubric transferability."""
import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


KNOWN_VENDORS = ("openai", "google", "qwen")

MODEL_LABELS = {
    "openai/gpt-5-mini": "GPT-5 mini",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen3-80B-A3B",
}

MODEL_ORDER = [
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "qwen/qwen3-next-80b-a3b-instruct",
]

DATASET_LABELS = {
    "asap_1": "ASAP P1",
    "ASAP2": "ASAP 2.0",
    "ets3": "TOEFL11",
}

DATASET_ORDER = ["asap_1", "ASAP2", "ets3"]

OURS_EXPERT_RUN = "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc1"
HUMAN_RUN = "zero_shot_no_expert"
CROSS_OURS_EXPERT_PREFIX = "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc1_from_"


def model_to_dir(model: str) -> str:
    return model.replace("/", "_")


def load_metric(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def resolve_run_dir(root: Path, dataset: str, model_dir: str, run_name: str) -> Optional[Path]:
    primary = root / dataset / model_dir / run_name
    if primary.is_dir():
        return primary
    return None


def load_same_model_qwk(root: Path, dataset: str, model: str, run_name: str) -> Optional[float]:
    run_dir = resolve_run_dir(root, dataset, model_to_dir(model), run_name)
    if run_dir is None:
        return None
    return load_metric(run_dir / "qwk.txt")


def load_cross_model_qwk(cross_root: Path, dataset: str, eval_model: str, source_model: str) -> Optional[float]:
    run_name = CROSS_OURS_EXPERT_PREFIX + model_to_dir(source_model)
    run_dir = cross_root / dataset / model_to_dir(eval_model) / run_name
    if not run_dir.is_dir():
        return None
    return load_metric(run_dir / "qwk.txt")


def build_matrix(same_root: Path, cross_root: Path, dataset: str) -> np.ndarray:
    n = len(MODEL_ORDER)
    matrix = np.full((n, n), np.nan)
    for i, source in enumerate(MODEL_ORDER):
        for j, evaluator in enumerate(MODEL_ORDER):
            if source == evaluator:
                qwk = load_same_model_qwk(same_root, dataset, evaluator, OURS_EXPERT_RUN)
            else:
                qwk = load_cross_model_qwk(cross_root, dataset, evaluator, source)
            if qwk is not None:
                matrix[i, j] = qwk
    return matrix


def build_human_row(same_root: Path, dataset: str) -> np.ndarray:
    n = len(MODEL_ORDER)
    row = np.full(n, np.nan)
    for j, model in enumerate(MODEL_ORDER):
        qwk = load_same_model_qwk(same_root, dataset, model, HUMAN_RUN)
        if qwk is not None:
            row[j] = qwk
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cross-model heatmap.")
    parser.add_argument("--same-root", default="evaluation_results")
    parser.add_argument("--cross-root", default="evaluation_results_cross_model")
    parser.add_argument("--output", default="../paper_latex/figures/cross_model_heatmap.pdf")
    args = parser.parse_args()

    same_root = Path(args.same_root)
    cross_root = Path(args.cross_root)

    model_short = [MODEL_LABELS[m] for m in MODEL_ORDER]
    n_datasets = len(DATASET_ORDER)
    n_models = len(MODEL_ORDER)

    fig, axes = plt.subplots(1, n_datasets, figsize=(4.2 * n_datasets, 4.0))
    if n_datasets == 1:
        axes = [axes]

    # Compute global vmin/vmax across all datasets for consistent color scale
    all_vals = []
    matrices = {}
    human_rows = {}
    for dataset in DATASET_ORDER:
        m = build_matrix(same_root, cross_root, dataset)
        h = build_human_row(same_root, dataset)
        matrices[dataset] = m
        human_rows[dataset] = h
        all_vals.extend(m[~np.isnan(m)].tolist())
        all_vals.extend(h[~np.isnan(h)].tolist())

    vmin = min(all_vals) if all_vals else 0
    vmax = max(all_vals) if all_vals else 1
    # Add some padding
    vmin = max(vmin - 0.05, -0.1)
    vmax = min(vmax + 0.05, 1.0)

    cmap = plt.cm.RdYlGn

    for idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[idx]
        matrix = matrices[dataset]
        human_row = human_rows[dataset]

        # Combine: human row on top, then transfer matrix
        combined = np.vstack([human_row.reshape(1, -1), matrix])
        row_labels = ["Human"] + model_short

        im = ax.imshow(combined, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        # Add text annotations
        for i in range(combined.shape[0]):
            for j in range(combined.shape[1]):
                val = combined[i, j]
                if np.isnan(val):
                    continue
                # Determine text color based on background
                norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                text_color = "white" if norm_val < 0.35 or norm_val > 0.85 else "black"
                fontweight = "bold" if i > 0 and (i - 1) == j else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, color=text_color, fontweight=fontweight)

        # Draw box around diagonal cells (rows 1-3, matching column)
        for k in range(n_models):
            rect = plt.Rectangle((k - 0.5, k + 1 - 0.5), 1, 1,
                                 linewidth=2.5, edgecolor="black", facecolor="none")
            ax.add_patch(rect)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(model_short, fontsize=11, rotation=20, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=11)
        ax.set_title(DATASET_LABELS[dataset], fontsize=12, fontweight="bold")

        if idx == 0:
            ax.set_ylabel("Rubric Source", fontsize=11)
        ax.set_xlabel("Evaluator Model", fontsize=11)

        # Add horizontal line to separate Human row
        ax.axhline(y=0.5, color="black", linewidth=1.5)

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("QWK", fontsize=10)

    fig.suptitle("Cross-Model Rubric Transferability", fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.92, 1.0])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
