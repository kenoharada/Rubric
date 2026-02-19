#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import prepare_ets_dataset, prepare_asap_dataset


def _safe_numeric_series(values: Iterable) -> pd.Series:
    series = pd.Series(list(values))
    try:
        numeric = pd.to_numeric(series)
    except Exception:
        return series
    if numeric.notna().all():
        return numeric.astype(int)
    return series


def _sort_labels(labels: Iterable, dataset: str) -> List:
    labels_list = list(labels)
    if dataset in ("ets", "ets3"):
        order = ["low", "medium", "high"]
        if set(labels_list).issubset(order):
            return [label for label in order if label in labels_list]
    numeric = _safe_numeric_series(labels_list)
    if pd.api.types.is_numeric_dtype(numeric):
        return sorted(set(numeric.tolist()))
    return sorted(set(labels_list))


def _counts_by_label(values: Iterable, dataset: str) -> pd.Series:
    series = _safe_numeric_series(values)
    counts = series.value_counts(dropna=True)
    labels = _sort_labels(counts.index.tolist(), dataset)
    return counts.reindex(labels, fill_value=0)


def _align_counts(
    full_counts: pd.Series, sample_counts: pd.Series, dataset: str
) -> Tuple[List, pd.Series, pd.Series]:
    all_labels = set(full_counts.index.tolist()) | set(sample_counts.index.tolist())
    labels = _sort_labels(all_labels, dataset)
    return (
        labels,
        full_counts.reindex(labels, fill_value=0),
        sample_counts.reindex(labels, fill_value=0),
    )


def _total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.abs(p - q).sum()


def _plot_grouped_bars(df: pd.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
    labels = df.index.tolist()
    columns = df.columns.tolist()
    x = np.arange(len(labels))
    width = 0.8 / max(len(columns), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, col in enumerate(columns):
        offset = (idx - (len(columns) - 1) / 2) * width
        ax.bar(x + offset, df[col].values, width, label=col)
    ax.set_title(title)
    ax.set_xlabel("label")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_delta_bars(df: pd.DataFrame, title: str, ylabel: str, out_path: Path) -> None:
    labels = df.index.tolist()
    columns = df.columns.tolist()
    x = np.arange(len(labels))
    width = 0.8 / max(len(columns), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, col in enumerate(columns):
        offset = (idx - (len(columns) - 1) / 2) * width
        ax.bar(x + offset, df[col].values, width, label=col)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("label")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _numeric_summary(values: Iterable) -> dict:
    series = _safe_numeric_series(values)
    if not pd.api.types.is_numeric_dtype(series):
        return {}
    return {
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _prepare_train_test_data(dataset: str):
    if dataset in ("ets", "ets3"):
        train_data, _, test_data = prepare_ets_dataset()
        return train_data, test_data
    if dataset.startswith("asap_"):
        essay_set = int(dataset.split("_")[1])
        train_data, _, test_data = prepare_asap_dataset(essay_set)
        return train_data, test_data
    raise ValueError(f"Unknown dataset: {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check train sample score distribution.")
    parser.add_argument("--dataset", required=True, help="ets, ets3, or asap_{essay_set}")
    parser.add_argument("--train-size", type=int, required=True, help="Sample size from train_data.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling.")
    parser.add_argument("--output-dir", default="distribution_checks", help="Output directory.")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots.")
    args = parser.parse_args()

    train_data, test_data = _prepare_train_test_data(args.dataset)
    random.seed(args.seed)
    random.shuffle(train_data)
    sampled = train_data[: args.train_size]

    full_labels = [item["answer"] for item in train_data]
    sample_labels = [item["answer"] for item in sampled]
    test_labels = [item["answer"] for item in test_data]

    full_counts = _counts_by_label(full_labels, args.dataset)
    sample_counts = _counts_by_label(sample_labels, args.dataset)
    test_counts = _counts_by_label(test_labels, args.dataset)
    labels, full_counts, sample_counts = _align_counts(full_counts, sample_counts, args.dataset)
    labels, _, test_counts = _align_counts(full_counts, test_counts, args.dataset)

    full_ratio = full_counts / full_counts.sum()
    sample_ratio = sample_counts / sample_counts.sum()
    delta_ratio = sample_ratio - full_ratio
    tvd = _total_variation_distance(full_ratio.values, sample_ratio.values)
    test_ratio = test_counts / test_counts.sum()
    test_delta_ratio = test_ratio - full_ratio
    tvd_test = _total_variation_distance(full_ratio.values, test_ratio.values)

    summary = {
        "dataset": args.dataset,
        "train_size": len(train_data),
        "sample_size": len(sampled),
        "test_size": len(test_data),
        "seed": args.seed,
        "total_variation_distance": float(tvd),
        "test_total_variation_distance": float(tvd_test),
        "full_numeric_summary": _numeric_summary(full_labels),
        "sample_numeric_summary": _numeric_summary(sample_labels),
        "test_numeric_summary": _numeric_summary(test_labels),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.dataset}_train{args.train_size}_seed{args.seed}"

    table = pd.DataFrame(
        {
            "label": labels,
            "full_count": full_counts.values,
            "sample_count": sample_counts.values,
            "test_count": test_counts.values,
            "full_ratio": full_ratio.values,
            "sample_ratio": sample_ratio.values,
            "test_ratio": test_ratio.values,
            "delta_ratio": delta_ratio.values,
            "test_delta_ratio": test_delta_ratio.values,
        }
    )
    table.to_csv(output_dir / f"{prefix}_counts.csv", index=False)

    with open(output_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not args.no_plot:
        counts_df = pd.DataFrame(
            {
                "full": full_counts.values,
                "sample": sample_counts.values,
                "test": test_counts.values,
            },
            index=labels,
        )
        ratios_df = pd.DataFrame(
            {
                "full": full_ratio.values,
                "sample": sample_ratio.values,
                "test": test_ratio.values,
            },
            index=labels,
        )
        delta_df = pd.DataFrame(
            {
                "sample - full": delta_ratio.values,
                "test - full": test_delta_ratio.values,
            },
            index=labels,
        )
        _plot_grouped_bars(
            counts_df,
            "Train vs Sample Counts",
            "count",
            output_dir / f"{prefix}_counts.png",
        )
        _plot_grouped_bars(
            ratios_df,
            "Train vs Sample Ratios",
            "ratio",
            output_dir / f"{prefix}_ratios.png",
        )
        _plot_delta_bars(
            delta_df,
            "Sample - Train Ratio Delta",
            "delta ratio",
            output_dir / f"{prefix}_ratio_delta.png",
        )

    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
