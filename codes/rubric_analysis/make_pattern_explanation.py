#!/usr/bin/env python3
"""
pattern_config.py の各パターン説明 + ランダム具体例(各2件)を含む
LaTeX Figure を生成する。

既定では run:
  zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4
を対象に、最適化後 rubric (best_rubric.txt) からスニペットを抽出する。

Usage:
  python codes/rubric_analysis/make_pattern_explanation.py
  python codes/rubric_analysis/make_pattern_explanation.py \
    --config zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4 \
    --output paper_latex/figures/fig_pattern_explanation.tex
  # 再現可能にしたい場合のみ seed を指定
  python codes/rubric_analysis/make_pattern_explanation.py --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

from pattern_config import PATTERNS, TYPE_INFO


@dataclass(frozen=True)
class SnippetExample:
    snippet_latex: str
    snippet_key: str
    source: str


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


def resolve_results_dirs(args_results_dir: str, project_root: Path) -> list[Path]:
    if args_results_dir:
        path = Path(args_results_dir)
        return [path] if path.exists() else []

    default_path = project_root / "codes" / "optimization_results"
    return [default_path] if default_path.exists() else []


def find_run_dirs(results_dirs: list[Path], config_name: str) -> list[dict]:
    config_clean = normalize_config_name(config_name)
    candidate_names = [x for x in {config_name.strip(), config_clean} if x]

    matches: list[dict] = []
    seen: set[str] = set()

    for results_dir in results_dirs:
        for dataset_dir in sorted(results_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for model_dir in sorted(dataset_dir.iterdir()):
                if not model_dir.is_dir():
                    continue
                for run_name in candidate_names:
                    run_dir = model_dir / run_name
                    best_path = run_dir / "best_rubric.txt"
                    if not best_path.exists():
                        continue
                    key = f"{dataset_dir.name}/{model_dir.name}/{run_name}"
                    if key in seen:
                        continue
                    seen.add(key)
                    matches.append(
                        {
                            "results_root": str(results_dir),
                            "dataset": dataset_dir.name,
                            "model": model_dir.name,
                            "config": run_name,
                            "run_dir": str(run_dir),
                            "best_path": str(best_path),
                        }
                    )
    matches.sort(key=lambda x: (x["dataset"], x["model"], x["config"]))
    return matches


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
    run_infos: list[dict],
    patterns: list[dict],
    context_chars: int,
    max_examples_per_doc: int,
    match_color_name: str,
) -> dict[str, list[SnippetExample]]:
    examples: dict[str, list[SnippetExample]] = {p["pattern_id"]: [] for p in patterns}
    seen: dict[str, set[str]] = {p["pattern_id"]: set() for p in patterns}
    compiled = {
        p["pattern_id"]: re.compile(p["regex"], flags=re.IGNORECASE) for p in patterns
    }

    for run in run_infos:
        best_text = safe_read_text(Path(run["best_path"]))
        source = f"{run['dataset']}/{run['model']}"
        for p in patterns:
            pid = p["pattern_id"]
            taken = 0
            for m in compiled[pid].finditer(best_text):
                if taken >= max_examples_per_doc:
                    break
                s, e = m.span()
                if s == e:
                    continue
                snippet_latex, snippet_key = build_snippet(
                    text=best_text,
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
                        source=source,
                    )
                )
                taken += 1
    return examples


def choose_examples(
    patterns: list[dict],
    examples: dict[str, list[SnippetExample]],
    k: int,
    seed: int | None,
) -> dict[str, list[SnippetExample]]:
    out: dict[str, list[SnippetExample]] = {}
    sys_rng = random.SystemRandom()
    for idx, p in enumerate(patterns):
        candidates = examples.get(p["pattern_id"], [])
        if len(candidates) <= k:
            out[p["pattern_id"]] = candidates
            continue
        if seed is None:
            out[p["pattern_id"]] = sys_rng.sample(candidates, k)
        else:
            rng = random.Random(seed + (idx + 1) * 10007)
            out[p["pattern_id"]] = rng.sample(candidates, k)
    return out


def build_pattern_block_latex(
    index: int,
    pattern: dict,
    selected_examples: list[SnippetExample],
) -> str:
    pid = pattern["pattern_id"]
    type_id = pattern.get("type_id", "other")
    type_label = TYPE_INFO.get(type_id, {}).get("label_en", type_id)
    name_en = pattern.get("name_en") or pattern.get("name_ja") or pid.replace("_", " ").title()
    desc_en = pattern.get("description_en") or pattern.get("description_ja") or pattern.get("regex", "")
    cues_en = pattern.get("cues_en") or pattern.get("regex", "")

    lines: list[str] = []
    lines.append(rf"\textbf{{{index}. {latex_escape_inline(name_en)}}}")
    lines.append(rf"\textit{{Category:}} {latex_escape_inline(type_label)}")
    lines.append(rf"\textit{{What this pattern captures:}} {latex_escape_inline(desc_en)}")
    lines.append(rf"\textit{{Typical cues:}} {latex_escape_inline(cues_en)}")

    if selected_examples:
        for j, ex in enumerate(selected_examples, start=1):
            lines.append(
                rf"\textit{{Example {j}:}} ``{ex.snippet_latex}'' "
            )
    else:
        lines.append(r"\textit{No matched examples found in the selected run.}")

    return r"\par ".join(lines)


def make_figure_tex(
    patterns: list[dict],
    selected: dict[str, list[SnippetExample]],
    run_count: int,
    examples_per_pattern: int,
    match_color_name: str,
    match_color_spec: str,
    label: str,
) -> str:
    blocks: list[str] = []
    for i, p in enumerate(patterns, start=1):
        blocks.append(build_pattern_block_latex(i, p, selected.get(p["pattern_id"], [])))

    body = r"\par\vspace{1.2mm}\par ".join(blocks)
    caption = (
        "Overview of rubric-refinement patterns and representative rubric snippets. "
        "For each pattern, we provide a short interpretation and randomly sampled "
        "matched spans from refined rubrics; highlighted words indicate the cue "
        "expressions that triggered each pattern. "
    )
    label_clean = label.strip()

    return rf"""
\colorlet{{{match_color_name}}}{{{match_color_spec}}}
\begin{{figure*}}[t]
\centering
\begin{{tcolorbox}}[
colback=white,
colframe=black!25,
title=Pattern Explanations with Random Matched Snippets,
fonttitle=\bfseries\small,
fontupper=\scriptsize,
boxsep=1pt,left=2pt,right=2pt,top=2pt,bottom=2pt]
{body}
\end{{tcolorbox}}
\caption{{{latex_escape_inline(caption)}}}
\label{{{label_clean}}}
\end{{figure*}}
""".strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pattern_config.py の各パターン説明 + ランダム例付きTeX Figureを生成する"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4",
        help="対象 run 名（zero_shot_ 付きでも可）",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="optimization_results のルートを明示指定する場合のパス",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help="dataset でフィルタ（完全一致）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="model でフィルタ（完全一致）",
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
        default=None,
        help="ランダムサンプル用シード（未指定なら毎回ランダム）",
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
        help="1文書あたり1パターンで採用する候補の上限",
    )
    parser.add_argument(
        "--include_disabled",
        action="store_true",
        help="enabled=False のパターンも含める",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="paper_latex/figures/fig_pattern_explanation.tex",
        help="出力する .tex ファイル",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="fig:pattern_explanation",
        help="Figure の \\label",
    )
    parser.add_argument(
        "--match_color_name",
        type=str,
        default="pcMatch",
        help="マッチ箇所の強調色名",
    )
    parser.add_argument(
        "--match_color_spec",
        type=str,
        default="red!75!black",
        help="マッチ強調色（xcolor形式）",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログを表示")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    results_dirs = resolve_results_dirs(args.results_dir, project_root)
    if not results_dirs:
        raise FileNotFoundError(
            "optimization_results が見つかりません。--results_dir を指定してください。"
        )

    run_infos = find_run_dirs(results_dirs=results_dirs, config_name=args.config)
    if args.dataset:
        run_infos = [r for r in run_infos if r["dataset"] == args.dataset]
    if args.model:
        run_infos = [r for r in run_infos if r["model"] == args.model]
    if not run_infos:
        raise RuntimeError(
            f"config '{args.config}' に一致する best_rubric.txt が見つかりません。"
        )

    patterns = get_patterns(include_disabled=args.include_disabled)
    if not patterns:
        raise RuntimeError("有効なパターンがありません。")

    examples = collect_pattern_examples(
        run_infos=run_infos,
        patterns=patterns,
        context_chars=max(10, args.context_chars),
        max_examples_per_doc=max(1, args.max_examples_per_doc),
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
        run_count=len(run_infos),
        examples_per_pattern=max(1, args.examples_per_pattern),
        match_color_name=args.match_color_name,
        match_color_spec=args.match_color_spec,
        label=args.label,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(figure_tex, encoding="utf-8")

    print(f"Generated: {output_path}")
    print(f"Run directories used: {len(run_infos)}")
    if args.verbose:
        for r in run_infos:
            print(f"- {r['dataset']}/{r['model']}/{r['config']}")
    for p in patterns:
        pid = p["pattern_id"]
        print(
            f"- {pid}: candidates={len(examples.get(pid, []))}, "
            f"selected={len(selected.get(pid, []))}"
        )


if __name__ == "__main__":
    main()
