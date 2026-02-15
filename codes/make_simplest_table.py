#!/usr/bin/env python3
"""Generate the LaTeX table comparing simplest-seed refined rubrics vs. human expert rubrics (Table tab:simplest)."""
import argparse
from pathlib import Path
from typing import Optional


SIMPLEST_OURS_RUN = "zero_shot_base_simplest_True_train100_iteration5_top3_bs4-8-12_mc1"
HUMAN_EXPERT_RUN = "zero_shot_no_expert"

KNOWN_VENDORS = ("openai", "google", "qwen")

MODEL_LABELS = {
    "openai/gpt-5-mini": "GPT-5 mini",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen3-30B-A3B",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
}

MODEL_ORDER = [
    "openai/gpt-5-mini",
    "google/gemini-3-flash-preview",
    "qwen/qwen3-next-80b-a3b-instruct",
]

DATASET_LABELS = {
    "asap_1": "ASAP P1",
    "ets3": "TOEFL11",
    "ASAP2": "ASAP 2.0",
}

DATASET_ORDER = ["asap_1", "ASAP2", "ets3"]


def restore_model_name(model_dir_name: str) -> str:
    for vendor in KNOWN_VENDORS:
        prefix = vendor + "_"
        if model_dir_name.startswith(prefix):
            return model_dir_name.replace(prefix, vendor + "/", 1)
    return model_dir_name


def load_metric(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except ValueError:
        return None


def scoring_dataset_for_run(dataset: str, run_name: str) -> str:
    return dataset


def resolve_run_dir(root: Path, dataset: str, model_dir_name: str, run_name: str) -> Optional[Path]:
    primary = root / dataset / model_dir_name / run_name
    if primary.is_dir():
        return primary
    alt_dataset = scoring_dataset_for_run(dataset, run_name)
    if alt_dataset != dataset:
        alt = root / alt_dataset / model_dir_name / run_name
        if alt.is_dir():
            return alt
    return None


def build_table(rows):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        "\\caption{QWK comparison: rubrics refined from the simplest seed "
        "(``Based on the response\'s content, rate the response on a scale of 1 to 6.'') vs.\\ the rubric written by human experts. "
        "$\\Delta$ denotes QWK improvement over the Human Expert baseline.}"
    )
    lines.append("\\label{tab:simplest}")
    lines.append("\\begin{tabular}{llrr}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Dataset} & \\textbf{LLM} & \\textbf{Simplest} "
        "& \\textbf{$\\Delta$ vs.\\ Human} \\\\"
    )
    lines.append(" & & \\textbf{+ Ours} & \\textbf{Expert} \\\\")
    lines.append("\\midrule")

    last_dataset = None
    for dataset, model, simplest_qwk, human_qwk in rows:
        dataset_label = DATASET_LABELS.get(dataset, dataset)
        model_label = MODEL_LABELS.get(model, model)
        delta = simplest_qwk - human_qwk

        if last_dataset is not None and dataset != last_dataset:
            lines.append("\\midrule")

        ds_col = dataset_label if dataset != last_dataset else ""
        last_dataset = dataset

        if delta >= 0:
            delta_str = f"+{delta:.3f}"
        else:
            delta_str = f"$-${abs(delta):.3f}"

        lines.append(f"{ds_col} & {model_label} & {simplest_qwk:.3f} & {delta_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate simplest-seed vs human expert QWK comparison table.")
    parser.add_argument("--root", default="evaluation_results", help="Root directory of evaluation results.")
    parser.add_argument("--output", help="Write LaTeX table to a file instead of stdout.")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    rows = []
    for dataset in DATASET_ORDER:
        dataset_dir = root / dataset
        if not dataset_dir.is_dir():
            continue
        for model in MODEL_ORDER:
            if model not in MODEL_LABELS:
                continue
            model_dir_name = model.replace("/", "_")
            # Load simplest + ours QWK
            simplest_dir = resolve_run_dir(root, dataset, model_dir_name, SIMPLEST_OURS_RUN)
            if simplest_dir is None:
                print(f"  Skipping {dataset}/{model}: no simplest-ours run found")
                continue
            simplest_qwk = load_metric(simplest_dir / "qwk.txt")
            if simplest_qwk is None:
                print(f"  Skipping {dataset}/{model}: no simplest-ours qwk.txt")
                continue

            # Load human expert QWK
            human_dir = resolve_run_dir(root, dataset, model_dir_name, HUMAN_EXPERT_RUN)
            if human_dir is None:
                print(f"  Skipping {dataset}/{model}: no human expert run found")
                continue
            human_qwk = load_metric(human_dir / "qwk.txt")
            if human_qwk is None:
                print(f"  Skipping {dataset}/{model}: no human expert qwk.txt")
                continue

            print(f"  {dataset}/{model}: simplest={simplest_qwk:.3f}, human={human_qwk:.3f}, delta={simplest_qwk - human_qwk:+.3f}")
            rows.append((dataset, model, simplest_qwk, human_qwk))

    if not rows:
        raise SystemExit("No data found.")

    table = build_table(rows)
    if args.output:
        Path(args.output).write_text(table + "\n")
        print(f"Written to {args.output}")
    else:
        print()
        print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
