"""
パターン説明専用のLaTeX Figureを1つ生成するスクリプト。

各パターンについて:
- 何を検出する基準か（説明）
- 典型キュー（どんな語で検出するか）
- 最適化後 rubric (best_rubric.txt) からランダム抽出した実例 2 件

Usage:
    python codes/analysis/make_pattern_explanation_figure.py
    python codes/analysis/make_pattern_explanation_figure.py \
      --run_names "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4,zero_shot_base_simplest_True_train100_iteration5_top3_bs4-8-12_mc4"
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class SnippetExample:
    snippet_latex: str
    snippet_key: str


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def latex_escape_inline(text: str) -> str:
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
    return (
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


def build_snippet(
    text: str,
    start: int,
    end: int,
    context_chars: int,
    match_color_name: str,
) -> tuple[str, str]:
    left = max(0, start - context_chars)
    right = min(len(text), end + context_chars)

    before = collapse_ws(text[left:start])
    target = collapse_ws(text[start:end])
    after = collapse_ws(text[end:right])
    if not target:
        return "", ""

    plain_parts: list[str] = []
    latex_parts: list[str] = []

    if left > 0:
        plain_parts.append("...")
        latex_parts.append("...")
    if before:
        plain_parts.append(before)
        latex_parts.append(latex_escape_inline(before))

    plain_parts.append(target)
    latex_parts.append(
        rf"\textcolor{{{match_color_name}}}{{\textbf{{{latex_escape_inline(target)}}}}}"
    )

    if after:
        plain_parts.append(after)
        latex_parts.append(latex_escape_inline(after))
    if right < len(text):
        plain_parts.append("...")
        latex_parts.append("...")

    plain_key = " ".join(plain_parts).lower().strip()
    latex_text = " ".join(latex_parts).strip()
    return latex_text, plain_key


def collect_pattern_examples(
    pairs: list[RubricPair],
    patterns: list[RegexPatternDef],
    context_chars: int,
    max_examples_per_doc: int,
    match_color_name: str,
) -> dict[str, list[SnippetExample]]:
    examples: dict[str, list[SnippetExample]] = {p.pattern_id: [] for p in patterns}
    seen: dict[str, set[str]] = {p.pattern_id: set() for p in patterns}
    compiled = {
        p.pattern_id: re.compile(p.regex, flags=re.IGNORECASE) for p in patterns
    }

    for pair in pairs:
        text = safe_read_text(Path(pair.best_path))
        for p in patterns:
            pid = p.pattern_id
            taken = 0
            for m in compiled[pid].finditer(text):
                if taken >= max_examples_per_doc:
                    break
                s, e = m.span()
                if s == e:
                    continue
                snippet_latex, snippet_key = build_snippet(
                    text=text,
                    start=s,
                    end=e,
                    context_chars=context_chars,
                    match_color_name=match_color_name,
                )
                if not snippet_key:
                    continue
                if snippet_key in seen[pid]:
                    continue
                seen[pid].add(snippet_key)
                examples[pid].append(
                    SnippetExample(
                        snippet_latex=snippet_latex,
                        snippet_key=snippet_key,
                    )
                )
                taken += 1
    return examples


def choose_examples(
    patterns: list[RegexPatternDef],
    examples: dict[str, list[SnippetExample]],
    k: int,
    seed: int,
) -> dict[str, list[SnippetExample]]:
    out: dict[str, list[SnippetExample]] = {}
    for idx, p in enumerate(patterns):
        candidates = examples.get(p.pattern_id, [])
        if len(candidates) <= k:
            out[p.pattern_id] = candidates
            continue
        rng = random.Random(seed + (idx + 1) * 7919)
        out[p.pattern_id] = rng.sample(candidates, k)
    return out


def build_pattern_block_latex(
    index: int,
    pattern: RegexPatternDef,
    selected_examples: list[SnippetExample],
) -> str:
    name_en = pattern.name_en or pattern.name_ja or pattern.pattern_id.replace("_", " ").title()
    desc_en = pattern.description_en or pattern.description_ja or pattern.regex
    cues_en = pattern.cues_en or pattern.regex

    lines: list[str] = []
    lines.append(rf"\textbf{{{index}. {latex_escape_inline(name_en)}}}")
    lines.append(
        rf"\textit{{What this pattern captures:}} {latex_escape_inline(desc_en)}"
    )
    lines.append(rf"\textit{{Typical cues:}} {latex_escape_inline(cues_en)}")
    if selected_examples:
        for j, ex in enumerate(selected_examples, start=1):
            lines.append(
                rf"\textit{{Example {j}:}} "
                + "``"
                + ex.snippet_latex
                + "''"
            )
    else:
        lines.append(r"\textit{No matched examples found in selected optimized rubrics.}")
    return r"\par ".join(lines)


def make_figure_tex(
    patterns: list[RegexPatternDef],
    selected: dict[str, list[SnippetExample]],
    examples_per_pattern: int,
    seed: int,
    match_color_name: str,
    match_color_spec: str,
) -> str:
    blocks: list[str] = []
    for i, p in enumerate(patterns, start=1):
        blocks.append(build_pattern_block_latex(i, p, selected.get(p.pattern_id, [])))

    body = r"\par\vspace{1.2mm}\par ".join(blocks)
    caption = (
        "Pattern criteria overview for rubric-change analysis. "
        "For each pattern, we show its intent and two randomly sampled snippets "
        "from optimized rubrics where the regex matched."
        f" (examples-per-pattern={examples_per_pattern}, seed={seed})"
    )

    return rf"""
\colorlet{{{match_color_name}}}{{{match_color_spec}}}
\begin{{figure*}}[t]
\centering
\begin{{tcolorbox}}[colback=white,colframe=black!25,title=Pattern Criteria with Real Optimized-Rubric Examples,fonttitle=\bfseries\small,fontupper=\scriptsize,boxsep=1pt,left=2pt,right=2pt,top=2pt,bottom=2pt]
{body}
\end{{tcolorbox}}
\caption{{{latex_escape_inline(caption)}}}
\label{{fig:pattern_criteria_examples}}
\end{{figure*}}
""".strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="パターン説明専用Figure（各パターンの説明 + ランダム2例）を生成"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="codes/optimization_results",
        help="best_rubric.txt を探索するルート",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="paper_latex/figures/fig_pattern_criteria_examples.tex",
        help="出力するFigure .tex のパス",
    )
    parser.add_argument(
        "--patterns_json",
        type=str,
        default="codes/analysis/rubric_regex_changes/patterns_used.json",
        help="正規表現パターンJSON",
    )
    parser.add_argument(
        "--examples_per_pattern",
        type=int,
        default=2,
        help="各パターンで抽出する例の数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ランダムサンプル用シード",
    )
    parser.add_argument(
        "--context_chars",
        type=int,
        default=55,
        help="マッチ前後に含める文字数",
    )
    parser.add_argument(
        "--max_examples_per_doc",
        type=int,
        default=8,
        help="1 rubric 文書あたり1パターンで採用する最大候補数",
    )
    parser.add_argument("--run_names", type=str, default="", help="run名フィルタ（カンマ区切り）")
    parser.add_argument("--datasets", type=str, default="", help="datasetフィルタ（カンマ区切り）")
    parser.add_argument("--models", type=str, default="", help="modelフィルタ（カンマ区切り）")
    parser.add_argument("--seed_prompts", type=str, default="", help="seed_promptフィルタ（カンマ区切り）")
    parser.add_argument("--with_rationale", type=str, default="", help="with_rationaleフィルタ（カンマ区切り）")
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="対象 rubric ペア数の上限（0は無制限）",
    )
    parser.add_argument(
        "--match_color_name",
        type=str,
        default="pcMatch",
        help="例中のマッチ強調に使う色名",
    )
    parser.add_argument(
        "--match_color_spec",
        type=str,
        default="red!75!black",
        help="例中のマッチ強調色（xcolor形式）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("条件に一致する rubric ペアがありません。")

    patterns = build_patterns(args.patterns_json) if args.patterns_json else DEFAULT_PATTERNS
    examples = collect_pattern_examples(
        pairs=pairs,
        patterns=patterns,
        context_chars=args.context_chars,
        max_examples_per_doc=args.max_examples_per_doc,
        match_color_name=args.match_color_name,
    )
    selected = choose_examples(
        patterns=patterns,
        examples=examples,
        k=max(1, args.examples_per_pattern),
        seed=args.seed,
    )

    figure_tex = make_figure_tex(
        patterns=patterns,
        selected=selected,
        examples_per_pattern=max(1, args.examples_per_pattern),
        seed=args.seed,
        match_color_name=args.match_color_name,
        match_color_spec=args.match_color_spec,
    )
    output_path.write_text(figure_tex, encoding="utf-8")

    print(f"Generated: {output_path}")
    print(f"Pairs used: {len(pairs)}")
    for p in patterns:
        pid = p.pattern_id
        print(
            f"- {pid}: candidates={len(examples.get(pid, []))}, "
            f"selected={len(selected.get(pid, []))}"
        )


if __name__ == "__main__":
    main()
