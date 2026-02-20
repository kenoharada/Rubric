#!/usr/bin/env python3
"""
Rubric本文の pattern 一致箇所を背景色でハイライトした LaTeX Figure を生成する。

既定では codes/optimization_results から指定 run の best_rubric.txt を探索し、
一致した run ごとに 1 つの .tex を出力する。

Usage:
  python codes/rubric_analysis/make_rubric_with_pattern_highlight.py

  python codes/rubric_analysis/make_rubric_with_pattern_highlight.py \
    --config zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4 \
    --dataset asap_1 --model openai_gpt-5-mini
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from pattern_config import PATTERNS


DEFAULT_CONFIG = "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4"

# patternごとの背景色（淡色、TeX xcolor 形式）
PATTERN_BG_COLORS: dict[str, str] = {
    "if_rules": "yellow!35",
    "tie_breaker_boundary": "orange!30",
    "stepwise_process": "green!30",
    "quantitative_thresholds": "cyan!28",
    "score_cap_demotion": "red!24",
    "evidence_count_safeguard": "blue!22",
    "concrete_exemplification": "teal!28",
    # disabled 含む fallback
    "negative_prescriptive": "pink!25",
    "offtopic_or_irrelevance": "gray!25",
    "elaboration_taxonomy": "violet!22",
    "counterargument_nuance": "magenta!18",
    "organization_coherence": "lime!25",
    "grammar_mechanics": "brown!20",
}


def get_patterns(include_disabled: bool) -> list[dict]:
    if include_disabled:
        return list(PATTERNS)
    return [p for p in PATTERNS if p.get("enabled", True)]


def normalize_config_name(name: str) -> str:
    s = name.strip().strip("()").strip().strip("'").strip('"').strip()
    if not s:
        return s
    if s.startswith("zero_shot_"):
        s = s[len("zero_shot_") :]
    return os.path.basename(s)


def sanitize_id(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", raw)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def to_tex_name(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "", raw.title())
    return s or "X"


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def latex_escape(text: str, keep_newline: bool = True) -> str:
    normalized = (
        text.replace("\u00a0", " ")
        .replace("≤", "<=")
        .replace("≥", ">=")
        .replace("≈", "approx")
        .replace("≠", "!=")
        .replace("→", "->")
        .replace("←", "<-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("…", "...")
    )
    escaped = (
        normalized.replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("$", r"\$")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )
    if keep_newline:
        return r"\par ".join(escaped.replace("\r\n", "\n").replace("\r", "\n").split("\n"))
    return escaped.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def line_excerpt_with_context(text: str, regexes: list[re.Pattern[str]], context_lines: int) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    hit_indices: set[int] = set()
    for i, line in enumerate(lines):
        if any(r.search(line) for r in regexes):
            hit_indices.add(i)

    if not hit_indices:
        return "\n".join(lines[: min(100, len(lines))])

    keep: set[int] = set()
    for idx in hit_indices:
        for j in range(max(0, idx - context_lines), min(len(lines), idx + context_lines + 1)):
            keep.add(j)

    kept = sorted(keep)
    out: list[str] = []
    prev = -1
    for idx in kept:
        if prev != -1 and idx > prev + 1:
            omitted = idx - prev - 1
            out.append(f"... [{omitted} lines omitted] ...")
        out.append(lines[idx])
        prev = idx
    return "\n".join(out)


def find_non_overlapping_spans(text: str, patterns: list[dict]) -> list[tuple[int, int, dict]]:
    """pattern定義順で優先し、重複しない一致スパンだけを採用する。"""
    chosen: list[tuple[int, int, dict]] = []
    used = [False] * len(text)

    for p in patterns:
        cregex = re.compile(p["regex"], flags=re.IGNORECASE)
        for m in cregex.finditer(text):
            s, e = m.span()
            if s >= e:
                continue
            if any(used[s:e]):
                continue
            for i in range(s, e):
                used[i] = True
            chosen.append((s, e, p))
    chosen.sort(key=lambda x: x[0])
    return chosen


def to_highlighted_latex(
    text: str,
    spans: list[tuple[int, int, dict]],
    color_name_by_pid: dict[str, str],
) -> str:
    chunks: list[str] = []
    cursor = 0
    for s, e, p in spans:
        if cursor < s:
            chunks.append(latex_escape(text[cursor:s], keep_newline=True))
        pid = p["pattern_id"]
        color_name = color_name_by_pid.get(pid, "rpBgFallback")
        target = latex_escape(text[s:e], keep_newline=False)
        chunks.append(rf"\rpHl{{{color_name}}}{{{target}}}")
        cursor = e
    if cursor < len(text):
        chunks.append(latex_escape(text[cursor:], keep_newline=True))
    return "".join(chunks)


def resolve_results_dir(args_results_dir: str, project_root: Path) -> Path:
    if args_results_dir:
        return Path(args_results_dir)
    return project_root / "codes" / "optimization_results"


def find_run_infos(
    results_dir: Path,
    config_name: str,
    dataset_filter: str,
    model_filter: str,
) -> list[dict]:
    config_clean = normalize_config_name(config_name)
    candidate_names = [x for x in {config_name.strip(), config_clean} if x]
    infos: list[dict] = []

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        if dataset_filter and dataset_dir.name != dataset_filter:
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            if model_filter and model_dir.name != model_filter:
                continue
            for run_name in candidate_names:
                run_dir = model_dir / run_name
                best_path = run_dir / "best_rubric.txt"
                if not best_path.exists():
                    continue
                infos.append(
                    {
                        "dataset": dataset_dir.name,
                        "model": model_dir.name,
                        "run_name": run_name,
                        "best_path": str(best_path),
                    }
                )
    infos.sort(key=lambda x: (x["dataset"], x["model"], x["run_name"]))
    return infos


def build_color_defs(patterns: list[dict]) -> tuple[str, dict[str, str]]:
    color_lines: list[str] = []
    color_name_by_pid: dict[str, str] = {}
    color_lines.append(r"\colorlet{rpBgFallback}{gray!20}")
    for p in patterns:
        pid = p["pattern_id"]
        cname = "rpBg" + to_tex_name(pid)
        cspec = PATTERN_BG_COLORS.get(pid, "gray!20")
        color_name_by_pid[pid] = cname
        color_lines.append(rf"\colorlet{{{cname}}}{{{cspec}}}")
    color_defs = "\n".join(dict.fromkeys(color_lines))
    return color_defs, color_name_by_pid


def build_legend_latex(
    patterns: list[dict],
    color_name_by_pid: dict[str, str],
    match_counts: dict[str, int],
) -> str:
    items: list[str] = []
    for p in patterns:
        pid = p["pattern_id"]
        cnt = match_counts.get(pid, 0)
        if cnt <= 0:
            continue
        color_name = color_name_by_pid[pid]
        pname = p.get("name_en") or p.get("name_ja") or pid
        items.append(
            rf"\rpLegendItem{{{color_name}}}{{\textbf{{{latex_escape(pname, keep_newline=False)}}} (n={cnt})}}"
        )
    if not items:
        return r"\textit{No pattern matches were found.}"

    rows: list[str] = []
    for i in range(0, len(items), 3):
        chunk = items[i : i + 3]
        while len(chunk) < 3:
            chunk.append("")
        rows.append(" & ".join(chunk) + r" \\")

    return (
        r"\begin{tabular}{@{}p{0.32\linewidth}p{0.32\linewidth}p{0.32\linewidth}@{}}"
        + "\n"
        + "\n".join(rows)
        + "\n"
        + r"\end{tabular}"
    )


def make_figure_tex(
    rubric_text: str,
    patterns: list[dict],
    dataset: str,
    model: str,
    run_name: str,
    excerpt_mode: str,
    context_lines: int,
) -> str:
    regexes = [re.compile(p["regex"], flags=re.IGNORECASE) for p in patterns]
    if excerpt_mode == "excerpt":
        text = line_excerpt_with_context(rubric_text, regexes, context_lines)
    else:
        text = rubric_text

    spans = find_non_overlapping_spans(text, patterns)
    match_counts: dict[str, int] = {}
    for _, _, p in spans:
        pid = p["pattern_id"]
        match_counts[pid] = match_counts.get(pid, 0) + 1

    color_defs, color_name_by_pid = build_color_defs(patterns)
    highlighted = to_highlighted_latex(text, spans, color_name_by_pid)
    legend = build_legend_latex(patterns, color_name_by_pid, match_counts)
    label = "fig:rubric_pattern_bg_" + sanitize_id(f"{dataset}_{model}_{run_name}")

    if excerpt_mode == "excerpt":
        scope_note = (
            f"The rubric panel shows excerpts around matched lines "
            f"(±{context_lines} line context); omitted stretches are marked explicitly."
        )
    else:
        scope_note = "The rubric panel shows the full refined rubric."

    caption_intro = (
        "Background colors mark text spans in the refined rubric that match each pattern. "
        "The legend above shows the pattern-to-color mapping and match counts. "
        "When multiple patterns overlap on the same span, only one highlight is retained "
        "to keep the visualization readable."
    )
    caption = (
        f"{latex_escape(caption_intro, keep_newline=False)} "
        f"{latex_escape(scope_note, keep_newline=False)} "
        f"({latex_escape(dataset, keep_newline=False)}, "
        f"{latex_escape(model, keep_newline=False)}, "
        f"{latex_escape(run_name, keep_newline=False)})."
    )

    return rf"""
{color_defs}
\providecommand{{\rpHl}}[2]{{\begingroup\setlength{{\fboxsep}}{{0.3pt}}\colorbox{{#1}}{{\strut #2}}\endgroup}}
\providecommand{{\rpLegendItem}}[2]{{\begingroup\setlength{{\fboxsep}}{{1.2pt}}\colorbox{{#1}}{{\strut #2}}\endgroup}}
\begin{{figure*}}[t]
\centering
\begin{{tcolorbox}}[colback=white,colframe=black!25,title=Pattern Legend,fonttitle=\bfseries\small,fontupper=\scriptsize,boxsep=1pt,left=2pt,right=2pt,top=2pt,bottom=2pt]
{legend}
\end{{tcolorbox}}
\vspace{{1mm}}
\begin{{tcolorbox}}[colback=white,colframe=black!25,title=Refined Rubric (Pattern-Highlighted),fonttitle=\bfseries\small,fontupper=\scriptsize]
\ttfamily
{highlighted}
\end{{tcolorbox}}
\caption{{{caption}}}
\label{{{label}}}
\end{{figure*}}
""".strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rubric本文のpattern一致箇所を背景色ハイライトしたLaTeX Figureを生成",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="対象 run 名（zero_shot_ プレフィックス可）",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="optimization_results ルート（省略時: codes/optimization_results）",
    )
    parser.add_argument("--dataset", type=str, default="", help="dataset フィルタ（完全一致）")
    parser.add_argument("--model", type=str, default="", help="model フィルタ（完全一致）")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper_latex/figures/rubric_pattern_highlights_bg",
        help="出力ディレクトリ",
    )
    parser.add_argument(
        "--include_disabled",
        action="store_true",
        help="enabled=False の pattern も含める",
    )
    parser.add_argument(
        "--excerpt_mode",
        type=str,
        choices=["excerpt", "full"],
        default="excerpt",
        help="全文表示(full)か、マッチ周辺抜粋(excerpt)か",
    )
    parser.add_argument(
        "--context_lines",
        type=int,
        default=1,
        help="excerpt 時に保持する前後行数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    results_dir = resolve_results_dir(args.results_dir, project_root)
    if not results_dir.exists():
        raise FileNotFoundError(
            f"results_dir not found: {results_dir}. --results_dir を指定してください。"
        )

    patterns = get_patterns(include_disabled=args.include_disabled)
    if not patterns:
        raise RuntimeError("有効な pattern がありません。")

    run_infos = find_run_infos(
        results_dir=results_dir,
        config_name=args.config,
        dataset_filter=args.dataset,
        model_filter=args.model,
    )
    if not run_infos:
        raise RuntimeError("対象 run が見つかりません。config/dataset/model を確認してください。")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[str] = []
    for info in run_infos:
        rubric_text = safe_read_text(Path(info["best_path"]))
        fig_tex = make_figure_tex(
            rubric_text=rubric_text,
            patterns=patterns,
            dataset=info["dataset"],
            model=info["model"],
            run_name=info["run_name"],
            excerpt_mode=args.excerpt_mode,
            context_lines=max(0, args.context_lines),
        )
        stem = sanitize_id(
            f"fig_rubric_with_pattern_highlight_{info['dataset']}_{info['model']}_{info['run_name']}"
        )
        out_path = output_dir / f"{stem}.tex"
        out_path.write_text(fig_tex, encoding="utf-8")
        generated.append(out_path.name)

    # まとめて \input するための master file
    output_posix = output_dir.as_posix()
    if "/figures/" in output_posix:
        figures_rel_dir = output_posix.split("/figures/", 1)[1]
    elif output_posix.startswith("figures/"):
        figures_rel_dir = output_posix[len("figures/") :]
    else:
        figures_rel_dir = output_dir.name

    master_lines = [
        rf"\input{{figures/{figures_rel_dir}/{fname[:-4]}}}" for fname in generated
    ]
    (output_dir / "fig_rubric_with_pattern_highlight_all.tex").write_text(
        "\n\n".join(master_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Generated {len(generated)} file(s) in: {output_dir}")
    print("Master file: fig_rubric_with_pattern_highlight_all.tex")


if __name__ == "__main__":
    main()
