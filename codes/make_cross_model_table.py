#!/usr/bin/env python3
"""Generate LaTeX table for cross-model rubric transferability results."""
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple


KNOWN_VENDORS = ("openai", "google", "qwen")

MODEL_LABELS = {
    "openai/gpt-5-mini": "GPT-5 mini",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "qwen/qwen3-next-80b-a3b-instruct": "Qwen3-30B-A3B",
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

# Same-model run names
OURS_EXPERT_RUN = "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc1"
HUMAN_RUN = "zero_shot_no_expert"

# Cross-model run name pattern (without _from_ suffix)
CROSS_OURS_EXPERT_PREFIX = "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc1_from_"
CROSS_AC_EXPERT_PREFIX = "zero_shot_base_expert_False_train100_iteration1_top3_bs4-8-12_mc1_from_"


def model_to_dir(model: str) -> str:
    return model.replace("/", "_")


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


def load_cross_model_qwk(
    cross_root: Path, dataset: str, eval_model: str, source_model: str, method: str = "ours"
) -> Optional[float]:
    if method == "ours":
        prefix = CROSS_OURS_EXPERT_PREFIX
    else:
        prefix = CROSS_AC_EXPERT_PREFIX
    run_name = prefix + model_to_dir(source_model)
    run_dir = cross_root / dataset / model_to_dir(eval_model) / run_name
    if not run_dir.is_dir():
        return None
    return load_metric(run_dir / "qwk.txt")


def format_qwk(value: Optional[float], is_diagonal: bool = False, is_best_in_col: bool = False) -> str:
    if value is None:
        return "--"
    s = f"{value:.3f}"
    if is_diagonal:
        s = f"\\cellcolor{{gray!20}}{s}"
    if is_best_in_col:
        s = f"\\textbf{{{s}}}"
    return s


def build_transfer_table(
    same_root: Path,
    cross_root: Path,
) -> str:
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\caption{Cross-model rubric transferability (QWK). Each cell shows QWK when the rubric optimized by the \\textit{source} model (row) is used by the \\textit{evaluator} model (column). "
                 "Shaded diagonal cells denote same-model results. ``Human'' row shows the unrefined human rubric baseline. "
                 "Bold values indicate the best QWK per evaluator model within each dataset.}")
    lines.append("\\label{tab:cross_model}")

    n_models = len(MODEL_ORDER)
    col_spec = "l" + "r" * n_models
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header_cells = ["\\textbf{Source $\\backslash$ Evaluator}"]
    for m in MODEL_ORDER:
        header_cells.append(f"\\textbf{{{MODEL_LABELS[m]}}}")
    lines.append(" & ".join(header_cells) + " \\\\")

    for di, dataset in enumerate(DATASET_ORDER):
        if di > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{n_models + 1}}}{{l}}{{\\textit{{{DATASET_LABELS[dataset]}}}}} \\\\")

        # Collect all QWK values: rows = [Human, model1, model2, model3]
        # For each eval_model (column), find the best value
        all_rows = []

        # Human baseline row
        human_row = []
        for eval_model in MODEL_ORDER:
            qwk = load_same_model_qwk(same_root, dataset, eval_model, HUMAN_RUN)
            human_row.append(qwk)
        all_rows.append(("Human", human_row, [False] * n_models))

        # Source model rows (Ours method)
        for source_model in MODEL_ORDER:
            row_vals = []
            is_diag = []
            for eval_model in MODEL_ORDER:
                if source_model == eval_model:
                    # Same-model: use same_root
                    qwk = load_same_model_qwk(same_root, dataset, eval_model, OURS_EXPERT_RUN)
                    is_diag.append(True)
                else:
                    qwk = load_cross_model_qwk(cross_root, dataset, eval_model, source_model, "ours")
                    is_diag.append(False)
                row_vals.append(qwk)
            all_rows.append((MODEL_LABELS[source_model], row_vals, is_diag))

        # Find best per column (excluding Human row for fairness, or including all)
        best_per_col = []
        for col_idx in range(n_models):
            col_vals = [row[1][col_idx] for row in all_rows if row[1][col_idx] is not None]
            best_per_col.append(max(col_vals) if col_vals else None)

        # Format rows
        for row_label, row_vals, is_diag in all_rows:
            cells = [row_label]
            for col_idx in range(n_models):
                v = row_vals[col_idx]
                is_best = (v is not None and best_per_col[col_idx] is not None
                           and abs(v - best_per_col[col_idx]) < 1e-9)
                cells.append(format_qwk(v, is_diagonal=is_diag[col_idx], is_best_in_col=is_best))
            lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate cross-model transferability table.")
    parser.add_argument("--same-root", default="evaluation_results",
                        help="Root of same-model evaluation results.")
    parser.add_argument("--cross-root", default="evaluation_results_cross_model",
                        help="Root of cross-model evaluation results.")
    parser.add_argument("--output", help="Write LaTeX table to a file instead of stdout.")
    args = parser.parse_args()

    same_root = Path(args.same_root)
    cross_root = Path(args.cross_root)

    if not same_root.exists():
        raise SystemExit(f"Same-model root not found: {same_root}")
    if not cross_root.exists():
        raise SystemExit(f"Cross-model root not found: {cross_root}")

    table = build_transfer_table(same_root, cross_root)

    if args.output:
        Path(args.output).write_text(table + "\n")
        print(f"Written to {args.output}")
    else:
        print(table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
