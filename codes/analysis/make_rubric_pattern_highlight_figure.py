"""
正規表現パターンにマッチした箇所を色付けし、論文用LaTeX Figureを生成するスクリプト。

出力:
- <output_dir>/fig_rubric_patterns_*.tex: 各RubricペアのFigure
- <output_dir>/fig_rubric_patterns_all.tex: 全Figureをまとめて\inputできるファイル

Usage:
    python analysis/make_rubric_pattern_highlight_figure.py \
      --run_names "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4,zero_shot_base_simplest_True_train100_iteration5_top3_bs4-8-12_mc4"
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from analyze_rubric_regex_changes import (
    DEFAULT_PATTERNS,
    RegexPatternDef,
    RubricPair,
    build_patterns,
    collect_rubric_pairs,
    filter_pairs,
    parse_csv_filter,
    parse_run_names_filter,
)


@dataclass
class PatternColor:
    pattern: RegexPatternDef
    type_id: str
    type_name_en: str
    type_desc_en: str
    color_name: str
    color_spec: str
    regex_compiled: re.Pattern[str]


@dataclass(frozen=True)
class TypeStyle:
    type_id: str
    type_name_en: str
    type_desc_en: str
    color_name: str
    color_spec: str


TYPE_STYLES: dict[str, TypeStyle] = {
    "rule_structure": TypeStyle(
        type_id="rule_structure",
        type_name_en="Rule Structure",
        type_desc_en="if/threshold/stepwise guidance",
        color_name="rpTypeRule",
        color_spec="red!80!black",
    ),
    "evidence_handling": TypeStyle(
        type_id="evidence_handling",
        type_name_en="Evidence Handling",
        type_desc_en="examples, repetition, and caps",
        color_name="rpTypeEvidence",
        color_spec="blue!80!black",
    ),
    "writing_quality": TypeStyle(
        type_id="writing_quality",
        type_name_en="Writing Quality",
        type_desc_en="organization and grammar/mechanics",
        color_name="rpTypeWriting",
        color_spec="teal!80!black",
    ),
    "other": TypeStyle(
        type_id="other",
        type_name_en="Other",
        type_desc_en="other matched rubric cues",
        color_name="rpTypeOther",
        color_spec="gray!80!black",
    ),
}


PATTERN_TO_TYPE: dict[str, str] = {
    "if_rules": "rule_structure",
    "tie_breaker_boundary": "rule_structure",
    "stepwise_process": "rule_structure",
    "anti_mechanical_counting": "rule_structure",
    "quantitative_thresholds": "rule_structure",
    "specific_examples_evidence": "evidence_handling",
    "offtopic_or_summary_cap": "evidence_handling",
    "repetition_noncount": "evidence_handling",
    "organization_coherence": "writing_quality",
    "grammar_mechanics": "writing_quality",
}


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def sanitize_id(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", raw)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "x"


def latex_escape(text: str) -> str:
    # pdfLaTeXで扱いづらいUnicodeを先にASCIIへ寄せる
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
    escaped = escaped.replace("\r\n", "\n").replace("\r", "\n")
    lines = escaped.split("\n")
    # \\ は「There's no line here to end」を起こしやすいため使わず、段落区切りで改行を表現
    return r"\par ".join(lines)


def build_pattern_colors(patterns: Iterable[RegexPatternDef]) -> list[PatternColor]:
    out: list[PatternColor] = []
    for p in patterns:
        type_id = PATTERN_TO_TYPE.get(p.pattern_id, "other")
        style = TYPE_STYLES.get(type_id, TYPE_STYLES["other"])
        out.append(
            PatternColor(
                pattern=p,
                type_id=style.type_id,
                type_name_en=style.type_name_en,
                type_desc_en=style.type_desc_en,
                color_name=style.color_name,
                color_spec=style.color_spec,
                regex_compiled=re.compile(p.regex, flags=re.IGNORECASE),
            )
        )
    return out


def line_excerpt_with_context(text: str, regexes: list[re.Pattern[str]], context: int) -> str:
    lines = text.splitlines()
    if not lines:
        return text

    hit_indices: set[int] = set()
    for i, line in enumerate(lines):
        if any(r.search(line) for r in regexes):
            hit_indices.add(i)

    if not hit_indices:
        return "\n".join(lines[: min(80, len(lines))])

    keep: set[int] = set()
    for idx in hit_indices:
        for j in range(max(0, idx - context), min(len(lines), idx + context + 1)):
            keep.add(j)

    kept_sorted = sorted(keep)
    out_lines: list[str] = []
    prev = -1
    for idx in kept_sorted:
        if prev != -1 and idx > prev + 1:
            omitted = idx - prev - 1
            out_lines.append(f"... [{omitted} lines omitted] ...")
        out_lines.append(lines[idx])
        prev = idx
    return "\n".join(out_lines)


def find_non_overlapping_spans(text: str, pattern_colors: list[PatternColor]) -> list[tuple[int, int, PatternColor]]:
    selected: list[tuple[int, int, PatternColor]] = []

    def overlaps(a: int, b: int, spans: list[tuple[int, int, PatternColor]]) -> bool:
        for s, e, _ in spans:
            if not (b <= s or a >= e):
                return True
        return False

    for pc in pattern_colors:
        for m in pc.regex_compiled.finditer(text):
            s, e = m.span()
            if s == e:
                continue
            if overlaps(s, e, selected):
                continue
            selected.append((s, e, pc))
    selected.sort(key=lambda x: x[0])
    return selected


def highlight_text_to_latex(
    text: str, pattern_colors: list[PatternColor]
) -> tuple[str, list[PatternColor]]:
    spans = find_non_overlapping_spans(text, pattern_colors)
    used_color_names: set[str] = set()
    chunks: list[str] = []
    cursor = 0
    for s, e, pc in spans:
        if cursor < s:
            chunks.append(latex_escape(text[cursor:s]))
        chunks.append(rf"\textcolor{{{pc.color_name}}}{{{latex_escape(text[s:e])}}}")
        used_color_names.add(pc.color_name)
        cursor = e
    if cursor < len(text):
        chunks.append(latex_escape(text[cursor:]))
    used = [pc for pc in pattern_colors if pc.color_name in used_color_names]
    return "".join(chunks), used


def unique_type_styles_from_patterns(used_patterns: list[PatternColor]) -> list[TypeStyle]:
    seen: set[str] = set()
    out: list[TypeStyle] = []
    for pc in used_patterns:
        if pc.type_id in seen:
            continue
        seen.add(pc.type_id)
        out.append(TYPE_STYLES.get(pc.type_id, TYPE_STYLES["other"]))
    # 表示順を固定
    order = ["rule_structure", "evidence_handling", "writing_quality", "other"]
    out.sort(key=lambda t: order.index(t.type_id) if t.type_id in order else len(order))
    return out


def build_legend_latex(used_patterns: list[PatternColor]) -> str:
    if not used_patterns:
        return r"\textit{No pattern matches found.}"
    used_types = unique_type_styles_from_patterns(used_patterns)
    items = []
    for style in used_types:
        items.append(
            rf"\textcolor{{{style.color_name}}}{{\textbf{{{latex_escape(style.type_name_en)}}}}}"
            + rf" ({latex_escape(style.type_desc_en)})"
        )
    return r" \quad ".join(items)


def build_caption_color_note(used_patterns: list[PatternColor]) -> str:
    used_types = unique_type_styles_from_patterns(used_patterns)
    if not used_types:
        return r"Color types: no matched patterns."
    parts = []
    for style in used_types:
        parts.append(
            rf"\textcolor{{{style.color_name}}}{{\textbf{{{latex_escape(style.type_name_en)}}}}}"
            + rf" ({latex_escape(style.type_desc_en)})"
        )
    return r"Color types: " + "; ".join(parts) + "."


def make_figure_tex(
    pair: RubricPair,
    pattern_colors: list[PatternColor],
    excerpt_mode: str,
    context_lines: int,
) -> str:
    before_raw = safe_read_text(Path(pair.initial_path))
    after_raw = safe_read_text(Path(pair.best_path))

    if excerpt_mode == "excerpt":
        regexes = [pc.regex_compiled for pc in pattern_colors]
        before = line_excerpt_with_context(before_raw, regexes, context_lines)
        after = line_excerpt_with_context(after_raw, regexes, context_lines)
    else:
        before = before_raw
        after = after_raw

    before_latex, used_before = highlight_text_to_latex(before, pattern_colors)
    after_latex, used_after = highlight_text_to_latex(after, pattern_colors)
    used_map = {pc.color_name: pc for pc in (used_before + used_after)}
    used_all = [pc for pc in pattern_colors if pc.color_name in used_map]

    label = "fig:rubric_pattern_" + sanitize_id(
        f"{pair.dataset}_{pair.model_name}_{pair.run_name}"
    )
    caption = (
        f"Pattern-highlighted rubric comparison ({pair.dataset}, {pair.model_name}, "
        f"{pair.run_name}). Matched spans are color-coded by regex pattern."
    )
    caption_note = build_caption_color_note(used_all)

    # type単位で重複なく色定義
    type_defs_seen: set[str] = set()
    color_def_lines: list[str] = []
    for pc in pattern_colors:
        if pc.color_name in type_defs_seen:
            continue
        type_defs_seen.add(pc.color_name)
        color_def_lines.append(rf"\colorlet{{{pc.color_name}}}{{{pc.color_spec}}}")
    color_defs = "\n".join(color_def_lines)
    legend = build_legend_latex(used_all)

    return rf"""
{color_defs}
\begin{{figure*}}[t]
\centering
\begin{{tcolorbox}}[colback=white,colframe=black!25,title=Pattern Legend,fonttitle=\bfseries\small,fontupper=\scriptsize,boxsep=1pt,left=2pt,right=2pt,top=2pt,bottom=2pt]
{legend}
\end{{tcolorbox}}
\vspace{{2mm}}
\begin{{minipage}}[t]{{0.485\textwidth}}
\begin{{tcolorbox}}[colback=white,colframe=black!25,title=Initial Rubric,fonttitle=\bfseries\small,fontupper=\scriptsize,breakable]
\ttfamily
{before_latex}
\end{{tcolorbox}}
\end{{minipage}}
\hfill
\begin{{minipage}}[t]{{0.485\textwidth}}
\begin{{tcolorbox}}[colback=white,colframe=black!25,title=Optimized Rubric,fonttitle=\bfseries\small,fontupper=\scriptsize,breakable]
\ttfamily
{after_latex}
\end{{tcolorbox}}
\end{{minipage}}
\caption{{{latex_escape(caption)} {caption_note}}}
\label{{{label}}}
\end{{figure*}}
""".strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rubricのpattern一致箇所を色付けしたLaTeX Figureを生成"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="codes/optimization_results",
        help="initial_rubric.txt / best_rubric.txt の探索ルート",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="paper_latex/figures/rubric_pattern_highlights",
        help="Figure texの出力先",
    )
    parser.add_argument(
        "--run_names",
        type=str,
        default="",
        help="対象run名（カンマ区切り、zero_shot_プレフィックス可）",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="datasetフィルタ（カンマ区切り）",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="modelフィルタ（カンマ区切り）",
    )
    parser.add_argument(
        "--seed_prompts",
        type=str,
        default="",
        help="seed_promptフィルタ（カンマ区切り）",
    )
    parser.add_argument(
        "--with_rationale",
        type=str,
        default="",
        help="with_rationaleフィルタ（カンマ区切り）",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="生成対象ペア数の上限。0なら全件",
    )
    parser.add_argument(
        "--patterns_json",
        type=str,
        default="",
        help="正規表現パターンJSON（analyze_rubric_regex_changes.pyと同形式）",
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
        help="excerpt時に保持する前後行数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    pairs = collect_rubric_pairs(results_root)
    pairs = filter_pairs(
        pairs=pairs,
        datasets=parse_csv_filter(args.datasets),
        models=parse_csv_filter(args.models),
        seed_prompts=parse_csv_filter(args.seed_prompts),
        with_rationale=parse_csv_filter(args.with_rationale),
        run_names=parse_run_names_filter(args.run_names),
        max_pairs=args.max_pairs,
    )
    if not pairs:
        raise RuntimeError("条件に一致するRubricペアがありません。")

    pattern_defs = build_patterns(args.patterns_json) if args.patterns_json else DEFAULT_PATTERNS
    pattern_colors = build_pattern_colors(pattern_defs)

    generated_files: list[str] = []
    for pair in pairs:
        fig_tex = make_figure_tex(
            pair=pair,
            pattern_colors=pattern_colors,
            excerpt_mode=args.excerpt_mode,
            context_lines=args.context_lines,
        )
        file_stem = sanitize_id(
            f"fig_rubric_patterns_{pair.dataset}_{pair.model_name}_{pair.run_name}"
        )
        out_path = output_dir / f"{file_stem}.tex"
        out_path.write_text(fig_tex, encoding="utf-8")
        generated_files.append(out_path.name)

    # \input先は「figures/配下からの相対」に揃える
    output_posix = output_dir.as_posix()
    if "/figures/" in output_posix:
        figures_rel_dir = output_posix.split("/figures/", 1)[1]
    elif output_posix.startswith("figures/"):
        figures_rel_dir = output_posix[len("figures/") :]
    else:
        figures_rel_dir = output_dir.name

    all_input_lines = []
    for fname in generated_files:
        all_input_lines.append(rf"\input{{figures/{figures_rel_dir}/{fname[:-4]}}}")
    (output_dir / "fig_rubric_patterns_all.tex").write_text(
        "\n\n".join(all_input_lines) + "\n", encoding="utf-8"
    )

    print(f"Generated {len(generated_files)} figure tex files in: {output_dir}")
    print("Master file: fig_rubric_patterns_all.tex")


if __name__ == "__main__":
    main()
