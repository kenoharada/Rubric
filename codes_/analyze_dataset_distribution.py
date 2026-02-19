import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import prepare_asap_dataset, prepare_ets_dataset


def _safe_numeric_series(values):
    series = pd.Series(values)
    try:
        numeric = pd.to_numeric(series)
    except Exception:
        return series
    if numeric.notna().all():
        return numeric.astype(int)
    return series


def _counts_by_answer(data):
    answers = [d.get("answer") for d in data]
    series = _safe_numeric_series(answers)
    return series.value_counts().sort_index()


def _align_counts(counts_list):
    all_answers = set()
    for counts in counts_list:
        all_answers.update(counts.index.tolist())
    sorted_answers = sorted(all_answers)
    aligned = [counts.reindex(sorted_answers, fill_value=0) for counts in counts_list]
    return sorted_answers, aligned


def _basic_stats(data):
    answers = [d.get("answer") for d in data]
    series = _safe_numeric_series(answers)
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    return {
        "n": int(series.shape[0]),
        "unique_count": int(series.nunique(dropna=True)),
        "mean": float(numeric.mean()) if not numeric.empty else np.nan,
        "std": float(numeric.std(ddof=0)) if not numeric.empty else np.nan,
        "min": float(numeric.min()) if not numeric.empty else np.nan,
        "median": float(numeric.median()) if not numeric.empty else np.nan,
        "max": float(numeric.max()) if not numeric.empty else np.nan,
    }


def _total_variation_distance(p, q):
    return 0.5 * np.abs(p - q).sum()


def _plot_grouped_bars(df, title, ylabel, out_path):
    answers = df.index.tolist()
    splits = df.columns.tolist()
    x = np.arange(len(answers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, split in enumerate(splits):
        ax.bar(x + (idx - (len(splits) - 1) / 2) * width, df[split].values, width, label=split)
    ax.set_title(title)
    ax.set_xlabel("answer")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(a) for a in answers])
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def analyze_dataset(name, train_data, val_data, test_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    split_names = ["train", "val", "test"]
    counts_list = [
        _counts_by_answer(train_data),
        _counts_by_answer(val_data),
        _counts_by_answer(test_data),
    ]
    answers, aligned_counts = _align_counts(counts_list)

    counts_df = pd.DataFrame(
        {split: counts.values for split, counts in zip(split_names, aligned_counts)},
        index=answers,
    )
    proportions_df = counts_df.div(counts_df.sum(axis=0), axis=1)

    stats = {
        "train": _basic_stats(train_data),
        "val": _basic_stats(val_data),
        "test": _basic_stats(test_data),
    }
    stats_df = pd.DataFrame(stats).T

    counts_df.to_csv(os.path.join(out_dir, "answer_counts.csv"))
    proportions_df.to_csv(os.path.join(out_dir, "answer_proportions.csv"))
    stats_df.to_csv(os.path.join(out_dir, "answer_stats.csv"))

    _plot_grouped_bars(
        counts_df,
        f"{name} answer distribution (counts)",
        "count",
        os.path.join(out_dir, "answer_counts.png"),
    )
    _plot_grouped_bars(
        proportions_df,
        f"{name} answer distribution (proportions)",
        "proportion",
        os.path.join(out_dir, "answer_proportions.png"),
    )

    tvd_rows = []
    for left_name, right_name in [("train", "val"), ("train", "test"), ("val", "test")]:
        left = proportions_df[left_name].values
        right = proportions_df[right_name].values
        tvd_rows.append({
            "pair": f"{left_name} vs {right_name}",
            "total_variation_distance": float(_total_variation_distance(left, right)),
        })
    pd.DataFrame(tvd_rows).to_csv(os.path.join(out_dir, "answer_tvd.csv"), index=False)


def _load_asap_essay_sets(data_path):
    df = pd.read_csv(data_path, sep="\t", encoding="latin-1", usecols=["essay_set"])
    return sorted(df["essay_set"].dropna().unique().astype(int).tolist())


def main():
    parser = argparse.ArgumentParser(description="Analyze answer distribution across splits.")
    parser.add_argument("--dataset", choices=["asap", "ets", "all"], default="all")
    parser.add_argument("--essay-sets", nargs="*", type=int, default=None)
    parser.add_argument("--output-dir", default="analysis/dataset_distribution")
    args = parser.parse_args()

    if args.dataset in ("ets", "all"):
        train_data, val_data, test_data = prepare_ets_dataset()
        out_dir = os.path.join(args.output_dir, "ets")
        analyze_dataset("ETS", train_data, val_data, test_data, out_dir)

    if args.dataset in ("asap", "all"):
        data_path = "dataset/asap-aes/training_set_rel3.tsv"
        essay_sets = args.essay_sets
        if not essay_sets:
            essay_sets = _load_asap_essay_sets(data_path)
        for essay_set in essay_sets:
            train_data, val_data, test_data = prepare_asap_dataset(essay_set=essay_set)
            out_dir = os.path.join(args.output_dir, f"asap_set_{essay_set}")
            analyze_dataset(f"ASAP essay_set={essay_set}", train_data, val_data, test_data, out_dir)


if __name__ == "__main__":
    main()
