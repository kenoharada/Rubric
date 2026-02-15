#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from inference import parse_rating, rating_to_quality, quality_to_score_for_qwk
from make_result_table import METHOD_LABELS, MODEL_LABELS, restore_model_name


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


def _counts_by_label(values: Iterable) -> pd.Series:
    series = _safe_numeric_series(values)
    return series.value_counts(dropna=True)


def _align_counts(counts_list: List[pd.Series], dataset: str) -> Tuple[List, List[pd.Series]]:
    all_labels = set()
    for counts in counts_list:
        all_labels.update(counts.index.tolist())
    sorted_labels = _sort_labels(all_labels, dataset)
    aligned = [counts.reindex(sorted_labels, fill_value=0) for counts in counts_list]
    return sorted_labels, aligned


def _total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * np.abs(p - q).sum()


def _strip_latex(text: str) -> str:
    text = text.replace("\\_", "_")
    text = text.replace("\\textbf{", "").replace("\\underline{", "")
    if text.endswith("}"):
        text = text[:-1]
    return text


def _display_label(run_name: str) -> str:
    label = METHOD_LABELS.get(run_name, run_name)
    return _strip_latex(label)


def _display_model(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


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


def _plot_delta_bars(df: pd.DataFrame, title: str, out_path: Path) -> None:
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
    ax.set_ylabel("delta (pred - true)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in labels])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _rankdata(values: Iterable[float]) -> List[float]:
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


def _spearman_corr(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    rx = _rankdata(x)
    ry = _rankdata(y)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = sum((a - mean_x) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_y) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _plot_metric_bars(df: pd.DataFrame, title: str, out_path: Path) -> None:
    labels = df.index.tolist()
    columns = df.columns.tolist()
    x = np.arange(len(labels))
    width = 0.8 / max(len(columns), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, col in enumerate(columns):
        offset = (idx - (len(columns) - 1) / 2) * width
        ax.bar(x + offset, df[col].values, width, label=col)
    ax.set_title(title)
    ax.set_xlabel("run")
    ax.set_ylabel("metric")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_confusion(
    matrix: np.ndarray,
    labels: List[int],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_yticklabels([str(l) for l in labels])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_scatter(
    y_true: List[int],
    y_pred: List[int],
    title: str,
    out_path: Path,
) -> None:
    if not y_true:
        return
    jitter = 0.1
    x = np.array(y_true) + np.random.uniform(-jitter, jitter, size=len(y_true))
    y = np.array(y_pred) + np.random.uniform(-jitter, jitter, size=len(y_pred))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(x, y, alpha=0.4, s=12)
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    ax.plot([min_v, max_v], [min_v, max_v], color="black", linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("true score")
    ax.set_ylabel("pred score")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_abs_diff_hist(
    y_true: List[int],
    y_pred: List[int],
    title: str,
    out_path: Path,
) -> None:
    if not y_true:
        return
    diffs = np.abs(np.array(y_pred) - np.array(y_true))
    bins = np.arange(diffs.max() + 2) - 0.5
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.hist(diffs, bins=bins, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("|pred - true|")
    ax.set_ylabel("count")
    ax.set_xticks(np.arange(diffs.max() + 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _unique_labels(labels: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for label in labels:
        if label not in seen:
            seen[label] = 1
            out.append(label)
        else:
            seen[label] += 1
            out.append(f"{label} ({seen[label]})")
    return out


def _safe_filename(text: str) -> str:
    return text.replace("/", "_").replace("\\", "_")


def _load_results(path: Path, dataset: str) -> Tuple[List, List, List[int], List[int], int]:
    true_labels = []
    pred_labels = []
    true_scores = []
    pred_scores = []
    invalid = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            true_label = row.get("annotated_score")
            true_labels.append(true_label)
            model_response = row.get("model_response")
            if not isinstance(model_response, str):
                invalid += 1
                continue
            rating = parse_rating(model_response)
            pred_label = rating_to_quality(rating, dataset) if rating is not None else None
            if pred_label is None:
                invalid += 1
                continue
            pred_labels.append(pred_label)
            true_score = quality_to_score_for_qwk(true_label, dataset)
            pred_score = quality_to_score_for_qwk(pred_label, dataset)
            if true_score is None or pred_score is None:
                invalid += 1
                continue
            true_scores.append(true_score)
            pred_scores.append(pred_score)
    return true_labels, pred_labels, true_scores, pred_scores, invalid


def _iter_results(root: Path) -> Iterable[Tuple[str, str, str, Path]]:
    for path in root.rglob("results.jsonl"):
        rel = path.relative_to(root)
        if len(rel.parts) < 3:
            continue
        dataset = rel.parts[0]
        model_dir = rel.parts[1]
        run_name = "/".join(rel.parts[2:-1])
        if 'few_shot' in run_name:
            continue
        yield dataset, model_dir, run_name, path


def _make_proportions(counts: pd.Series) -> pd.Series:
    total = counts.sum()
    if total == 0:
        return counts.astype(float)
    return counts / total


def compare_model_runs(
    dataset: str,
    model_dir: str,
    run_paths: Dict[str, Path],
    output_root: Path,
) -> List[Dict]:
    stats_rows = []
    model_name = restore_model_name(model_dir)
    model_label = _display_model(model_name)
    out_dir = output_root / dataset / model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    true_counts: Optional[pd.Series] = None
    run_counts: Dict[str, pd.Series] = {}
    run_proportions: Dict[str, pd.Series] = {}
    run_invalid: Dict[str, Tuple[int, int]] = {}
    run_scores: Dict[str, Tuple[List[int], List[int]]] = {}

    for run_name, path in sorted(run_paths.items()):
        true_labels, pred_labels, true_scores, pred_scores, invalid = _load_results(path, dataset)
        true_counts_run = _counts_by_label(true_labels)
        pred_counts = _counts_by_label(pred_labels)
        if true_counts is None:
            true_counts = true_counts_run
        run_counts[run_name] = pred_counts
        run_proportions[run_name] = _make_proportions(pred_counts)
        run_invalid[run_name] = (invalid, len(true_labels))
        run_scores[run_name] = (true_scores, pred_scores)

    if true_counts is None:
        return stats_rows

    counts_list = [true_counts] + [run_counts[name] for name in run_counts]
    labels, aligned = _align_counts(counts_list, dataset)

    aligned_true = aligned[0]
    aligned_runs = aligned[1:]
    true_prop = _make_proportions(aligned_true)

    run_labels = [_display_label(name) for name in run_counts]
    run_labels = _unique_labels(run_labels)

    counts_df = pd.DataFrame(
        {label: counts.values for label, counts in zip(run_labels, aligned_runs)},
        index=labels,
    )
    counts_df.insert(0, "True", aligned_true.values)

    proportions_df = pd.DataFrame(
        {label: _make_proportions(counts).values for label, counts in zip(run_labels, aligned_runs)},
        index=labels,
    )
    proportions_df.insert(0, "True", true_prop.values)

    delta_df = pd.DataFrame(
        {label: proportions_df[label].values - proportions_df["True"].values for label in run_labels},
        index=labels,
    )

    counts_df.to_csv(out_dir / "answer_counts.csv")
    proportions_df.to_csv(out_dir / "answer_proportions.csv")
    delta_df.to_csv(out_dir / "answer_delta.csv")

    _plot_grouped_bars(
        proportions_df,
        f"{model_label} ({dataset}) answer distribution",
        "proportion",
        out_dir / "answer_proportions.png",
    )
    _plot_delta_bars(
        delta_df,
        f"{model_label} ({dataset}) delta vs true",
        out_dir / "answer_delta.png",
    )

    metric_rows = []
    for run_name, run_label in zip(run_counts.keys(), run_labels):
        invalid, total = run_invalid[run_name]
        pred_prop = _make_proportions(run_counts[run_name]).reindex(labels, fill_value=0).values
        tvd = _total_variation_distance(pred_prop, true_prop.values)
        true_scores, pred_scores = run_scores[run_name]
        qwk = None
        spearman = None
        f1 = None
        if true_scores and pred_scores:
            try:
                from sklearn.metrics import cohen_kappa_score
                qwk = cohen_kappa_score(true_scores, pred_scores, weights="quadratic")
            except Exception:
                qwk = None
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(true_scores, pred_scores, average="macro")
            except Exception:
                f1 = None
            spearman = _spearman_corr(true_scores, pred_scores)
        stats_rows.append({
            "dataset": dataset,
            "model": model_name,
            "model_label": model_label,
            "run_name": run_name,
            "run_label": run_label,
            "invalid_count": invalid,
            "total_count": total,
            "invalid_rate": invalid / total if total else 0.0,
            "total_variation_distance": float(tvd),
            "qwk": qwk,
            "spearman": spearman,
            "f1_macro": f1,
        })
        metric_rows.append({
            "run": run_label,
            "qwk": qwk,
            "spearman": spearman,
            "f1_macro": f1,
        })

        if true_scores and pred_scores:
            score_labels = _sort_labels(set(true_scores + pred_scores), dataset)
            label_to_idx = {label: i for i, label in enumerate(score_labels)}
            matrix = np.zeros((len(score_labels), len(score_labels)), dtype=int)
            for t, p in zip(true_scores, pred_scores):
                matrix[label_to_idx[t], label_to_idx[p]] += 1
            _plot_confusion(
                matrix,
                score_labels,
                f"{model_label} ({dataset}) {run_label} confusion",
                out_dir / f"confusion_{_safe_filename(run_label)}.png",
            )
            _plot_scatter(
                true_scores,
                pred_scores,
                f"{model_label} ({dataset}) {run_label} true vs pred",
                out_dir / f"scatter_{_safe_filename(run_label)}.png",
            )
            _plot_abs_diff_hist(
                true_scores,
                pred_scores,
                f"{model_label} ({dataset}) {run_label} |pred-true|",
                out_dir / f"absdiff_hist_{_safe_filename(run_label)}.png",
            )

    pd.DataFrame(stats_rows).to_csv(out_dir / "run_stats.csv", index=False)
    if metric_rows:
        metrics_df = pd.DataFrame(metric_rows).set_index("run")
        metrics_df.to_csv(out_dir / "metrics.csv")
        metrics_df.to_csv(out_dir / "qwk_spearman.csv")
        _plot_metric_bars(
            metrics_df,
            f"{model_label} ({dataset}) metrics",
            out_dir / "metrics.png",
        )
        _plot_metric_bars(
            metrics_df.drop(columns=[c for c in ["f1_macro"] if c in metrics_df.columns]),
            f"{model_label} ({dataset}) QWK/Spearman",
            out_dir / "qwk_spearman.png",
        )
    return stats_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare model answer distributions vs true labels."
    )
    parser.add_argument("--results-root", default="evaluation_results")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--output-dir", default="analysis/model_distribution")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_root = Path(args.output_dir)

    model_filters = set(args.models or [])
    collected: Dict[str, Dict[str, Dict[str, Path]]] = {}

    for dataset, model_dir, run_name, path in _iter_results(results_root):
        if args.dataset != "all" and dataset != args.dataset:
            continue
        model_name = restore_model_name(model_dir)
        if model_filters and model_name not in model_filters and model_dir not in model_filters:
            continue
        collected.setdefault(dataset, {}).setdefault(model_dir, {})[run_name] = path

    all_stats: List[Dict] = []
    for dataset, model_map in sorted(collected.items()):
        for model_dir, run_paths in sorted(model_map.items()):
            all_stats.extend(compare_model_runs(dataset, model_dir, run_paths, output_root))

    if all_stats:
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_stats).to_csv(output_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
