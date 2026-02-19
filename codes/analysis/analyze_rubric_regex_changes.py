"""
初期Rubricと最適化後Rubricの差分を、正規表現ベースで粗判定するスクリプト。

主な出力:
- pair_pattern_results.csv: ペア×パターン単位の判定結果
- pattern_summary.csv: パターンごとの全体集計
- pattern_summary_by_group.csv: dataset / seed_prompt / with_rationale 別の集計
- summary.md: 人間向けサマリ

Usage:
    python analysis/analyze_rubric_regex_changes.py
    python analysis/analyze_rubric_regex_changes.py --datasets asap_1,ets3 --seed_prompts expert
    python analysis/analyze_rubric_regex_changes.py --run_names "zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4,zero_shot_base_simplest_True_train100_iteration5_top3_bs4-8-12_mc4"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Literal


RUN_NAME_PATTERN = re.compile(
    r"^(?P<optimize_method>.+)_(?P<seed_prompt>simplest|simple|expert|self)"
    r"_(?P<with_rationale>True|False)_train(?P<train_size>\d+)"
    r"_iteration(?P<iteration>\d+)_top(?P<top_k>\d+)_bs(?P<batch_sizes>[\d-]+)_mc(?P<mc_runs>\d+)$"
)

Status = Literal[
    "added_or_strengthened",
    "removed_or_weakened",
    "present_both",
    "absent_both",
]


@dataclass
class RubricPair:
    pair_id: str
    dataset: str
    model_name: str
    run_name: str
    initial_path: str
    best_path: str
    seed_prompt: str | None
    with_rationale: str | None


@dataclass
class RegexPatternDef:
    pattern_id: str
    name_ja: str
    description_ja: str
    regex: str


DEFAULT_PATTERNS: list[RegexPatternDef] = [
    RegexPatternDef(
        pattern_id="if_rules",
        name_ja="条件付きルール",
        description_ja="if/when等で条件分岐を明示する記述",
        regex=r"\bif\b|\bwhen\b",
    ),
    RegexPatternDef(
        pattern_id="tie_breaker_boundary",
        name_ja="境界・タイブレーク",
        description_ja="境界ケース（4 vs 5等）やtie-breakerの記述",
        regex=r"tie-?break|borderline|boundary|threshold|4\s*vs\.?\s*5|5\s*vs\.?\s*4",
    ),
    RegexPatternDef(
        pattern_id="stepwise_process",
        name_ja="ステップ式手順",
        description_ja="Step/チェックリスト/手順化された採点フロー",
        regex=r"\bstep\b|checklist|workflow|procedure|first\b|second\b|third\b",
    ),
    RegexPatternDef(
        pattern_id="anti_mechanical_counting",
        name_ja="機械的カウントの抑制",
        description_ja="単純な数え上げを禁止・抑制する記述",
        regex=r"do not count|not by mechanical counting|not mechanically|do not equate",
    ),
    RegexPatternDef(
        pattern_id="specific_examples_evidence",
        name_ja="具体例・根拠の要求",
        description_ja="具体例・証拠・説明リンクの要求",
        regex=r"for example|e\.g\.|specific example|illustration|anecdote|evidence",
    ),
    RegexPatternDef(
        pattern_id="offtopic_or_summary_cap",
        name_ja="脱線・要約偏重への制御",
        description_ja="off-topicやsummary-onlyを低スコアに制限",
        regex=r"off-?topic|irrelevant|digression|summary-only|\bcap\b",
    ),
    RegexPatternDef(
        pattern_id="organization_coherence",
        name_ja="構成・一貫性",
        description_ja="organization/coherence/transition等の構成要素",
        regex=r"organization|coherence|logical flow|transition",
    ),
    RegexPatternDef(
        pattern_id="grammar_mechanics",
        name_ja="文法・メカニクス",
        description_ja="grammar/mechanics/spelling/punctuation等",
        regex=r"grammar|mechanics|spelling|punctuation",
    ),
    RegexPatternDef(
        pattern_id="repetition_noncount",
        name_ja="重複主張の非カウント",
        description_ja="repetition/restatementを別理由として数えない",
        regex=r"repetition|restatement|double-?count|do not double-?count",
    ),
    RegexPatternDef(
        pattern_id="quantitative_thresholds",
        name_ja="数量しきい値",
        description_ja="at least / <= / % など数的しきい値",
        regex=r"at least|at most|<=|>=|%|\b\d+\s*(?:reasons?|examples?|sentences?|words?)\b",
    ),
]


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_run_name(run_name: str) -> dict[str, str | int | None]:
    m = RUN_NAME_PATTERN.match(run_name)
    if not m:
        return {
            "seed_prompt": None,
            "with_rationale": None,
        }
    gd = m.groupdict()
    return {
        "seed_prompt": gd["seed_prompt"],
        "with_rationale": gd["with_rationale"],
    }


def collect_rubric_pairs(results_root: Path) -> list[RubricPair]:
    pairs: list[RubricPair] = []
    for initial_path in results_root.rglob("initial_rubric.txt"):
        best_path = initial_path.with_name("best_rubric.txt")
        if not best_path.exists():
            continue

        rel = initial_path.relative_to(results_root)
        if len(rel.parts) < 4:
            continue
        dataset, model_name, run_name = rel.parts[0], rel.parts[1], rel.parts[2]
        parsed = parse_run_name(run_name)
        pair_id = f"{dataset}/{model_name}/{run_name}"
        pairs.append(
            RubricPair(
                pair_id=pair_id,
                dataset=dataset,
                model_name=model_name,
                run_name=run_name,
                initial_path=str(initial_path),
                best_path=str(best_path),
                seed_prompt=parsed["seed_prompt"],  # type: ignore[arg-type]
                with_rationale=parsed["with_rationale"],  # type: ignore[arg-type]
            )
        )
    pairs.sort(key=lambda x: x.pair_id)
    return pairs


def parse_csv_filter(arg: str) -> set[str] | None:
    vals = [x.strip() for x in arg.split(",") if x.strip()]
    return set(vals) if vals else None


def normalize_requested_run_name(name: str) -> str:
    # 入力の余分な括弧・引用符を除去
    s = name.strip().strip("()").strip().strip("'").strip('"').strip()
    if not s:
        return s
    # evaluation_results側のrun名（zero_shot_プレフィックス）も受け付ける
    if s.startswith("zero_shot_"):
        s = s[len("zero_shot_") :]
    # 万一パスが渡された場合は末尾要素を採用
    s = os.path.basename(s)
    return s


def parse_run_names_filter(arg: str) -> set[str] | None:
    if not arg.strip():
        return None
    normalized = set()
    for token in arg.split(","):
        run_name = normalize_requested_run_name(token)
        if run_name:
            normalized.add(run_name)
    return normalized if normalized else None


def filter_pairs(
    pairs: list[RubricPair],
    datasets: set[str] | None,
    models: set[str] | None,
    seed_prompts: set[str] | None,
    with_rationale: set[str] | None,
    run_names: set[str] | None,
    max_pairs: int,
) -> list[RubricPair]:
    out: list[RubricPair] = []
    for p in pairs:
        if datasets and p.dataset not in datasets:
            continue
        if models and p.model_name not in models:
            continue
        if seed_prompts and (p.seed_prompt or "") not in seed_prompts:
            continue
        if with_rationale and (p.with_rationale or "") not in with_rationale:
            continue
        if run_names and p.run_name not in run_names:
            continue
        out.append(p)
        if max_pairs > 0 and len(out) >= max_pairs:
            break
    return out


def build_patterns(patterns_json: str) -> list[RegexPatternDef]:
    if not patterns_json:
        return DEFAULT_PATTERNS

    raw = json.loads(Path(patterns_json).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "patterns" in raw:
        raw = raw["patterns"]
    patterns: list[RegexPatternDef] = []
    for row in raw:
        patterns.append(
            RegexPatternDef(
                pattern_id=row["pattern_id"],
                name_ja=row["name_ja"],
                description_ja=row["description_ja"],
                regex=row["regex"],
            )
        )
    return patterns


def classify_status(
    count_before: int, count_after: int, presence_only: bool
) -> Status:
    if presence_only:
        has_before = count_before > 0
        has_after = count_after > 0
        if (not has_before) and has_after:
            return "added_or_strengthened"
        if has_before and (not has_after):
            return "removed_or_weakened"
        if has_before and has_after:
            return "present_both"
        return "absent_both"

    if count_before == 0 and count_after > 0:
        return "added_or_strengthened"
    if count_before > 0 and count_after == 0:
        return "removed_or_weakened"
    if count_before > 0 and count_after > 0:
        if count_after > count_before:
            return "added_or_strengthened"
        if count_after < count_before:
            return "removed_or_weakened"
        return "present_both"
    return "absent_both"


def run_analysis(
    pairs: list[RubricPair],
    patterns: list[RegexPatternDef],
    presence_only: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    compiled = []
    for p in patterns:
        compiled.append((p, re.compile(p.regex, flags=re.IGNORECASE)))

    pair_rows: list[dict[str, object]] = []
    for pair in pairs:
        before = safe_read_text(Path(pair.initial_path))
        after = safe_read_text(Path(pair.best_path))
        for pdef, cregex in compiled:
            count_before = len(cregex.findall(before))
            count_after = len(cregex.findall(after))
            status = classify_status(count_before, count_after, presence_only)
            pair_rows.append(
                {
                    "pair_id": pair.pair_id,
                    "dataset": pair.dataset,
                    "model_name": pair.model_name,
                    "run_name": pair.run_name,
                    "seed_prompt": pair.seed_prompt or "",
                    "with_rationale": pair.with_rationale or "",
                    "pattern_id": pdef.pattern_id,
                    "pattern_name_ja": pdef.name_ja,
                    "pattern_desc_ja": pdef.description_ja,
                    "regex": pdef.regex,
                    "count_before": count_before,
                    "count_after": count_after,
                    "status": status,
                }
            )

    summary_rows = summarize_rows(pair_rows, group_name="overall", group_value="all")

    group_rows: list[dict[str, object]] = []
    for group_key in ["dataset", "seed_prompt", "with_rationale"]:
        values = sorted({str(r[group_key]) for r in pair_rows if str(r[group_key])})
        for val in values:
            subset = [r for r in pair_rows if str(r[group_key]) == val]
            group_rows.extend(summarize_rows(subset, group_name=group_key, group_value=val))

    return pair_rows, summary_rows, group_rows


def summarize_rows(
    rows: list[dict[str, object]], group_name: str, group_value: str
) -> list[dict[str, object]]:
    by_pattern: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_pattern[str(r["pattern_id"])].append(r)

    out: list[dict[str, object]] = []
    for pattern_id in sorted(by_pattern.keys()):
        subset = by_pattern[pattern_id]
        n = len(subset)
        counter = Counter(str(r["status"]) for r in subset)
        avg_before = mean(float(r["count_before"]) for r in subset) if subset else 0.0
        avg_after = mean(float(r["count_after"]) for r in subset) if subset else 0.0
        med_before = median(float(r["count_before"]) for r in subset) if subset else 0.0
        med_after = median(float(r["count_after"]) for r in subset) if subset else 0.0
        sample = subset[0]

        out.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "pattern_id": pattern_id,
                "pattern_name_ja": sample["pattern_name_ja"],
                "pattern_desc_ja": sample["pattern_desc_ja"],
                "pair_pattern_count": n,
                "added_or_strengthened": counter["added_or_strengthened"],
                "removed_or_weakened": counter["removed_or_weakened"],
                "present_both": counter["present_both"],
                "absent_both": counter["absent_both"],
                "added_or_strengthened_rate": counter["added_or_strengthened"] / n if n else 0.0,
                "removed_or_weakened_rate": counter["removed_or_weakened"] / n if n else 0.0,
                "present_both_rate": counter["present_both"] / n if n else 0.0,
                "absent_both_rate": counter["absent_both"] / n if n else 0.0,
                "avg_count_before": avg_before,
                "avg_count_after": avg_after,
                "median_count_before": med_before,
                "median_count_after": med_after,
            }
        )
    out.sort(key=lambda x: float(x["added_or_strengthened_rate"]), reverse=True)
    return out


def build_summary_md(
    pair_count: int,
    summary_rows: list[dict[str, object]],
    group_rows: list[dict[str, object]],
    presence_only: bool,
) -> str:
    lines: list[str] = []
    lines.append("# Rubric差分の正規表現ベース粗判定\n")
    lines.append(f"- 対象ペア数: **{pair_count}**")
    lines.append(
        f"- 判定モード: **{'presence_only' if presence_only else 'count_based'}**"
    )

    lines.append("\n## 全体サマリ\n")
    lines.append(
        "| pattern_id | 項目 | added/strengthened | removed/weakened | present_both | absent_both |"
    )
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            f"| `{r['pattern_id']}` | {r['pattern_name_ja']} | "
            f"{r['added_or_strengthened']} ({float(r['added_or_strengthened_rate']):.3f}) | "
            f"{r['removed_or_weakened']} ({float(r['removed_or_weakened_rate']):.3f}) | "
            f"{r['present_both']} ({float(r['present_both_rate']):.3f}) | "
            f"{r['absent_both']} ({float(r['absent_both_rate']):.3f}) |"
        )

    lines.append("\n## グループ別（dataset / seed_prompt / with_rationale）\n")
    for gname in ["dataset", "seed_prompt", "with_rationale"]:
        subset = [r for r in group_rows if r["group_name"] == gname]
        if not subset:
            continue
        lines.append(f"### {gname}")
        by_group_value: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in subset:
            by_group_value[str(row["group_value"])].append(row)
        for gval in sorted(by_group_value.keys()):
            lines.append(f"- {gval}")
            top = sorted(
                by_group_value[gval],
                key=lambda x: float(x["added_or_strengthened_rate"]),
                reverse=True,
            )[:5]
            for row in top:
                lines.append(
                    f"  - `{row['pattern_id']}`: added/strengthened="
                    f"{row['added_or_strengthened']} ({float(row['added_or_strengthened_rate']):.3f})"
                )
    lines.append("\n## 出力ファイル\n")
    lines.append("- `pair_pattern_results.csv`")
    lines.append("- `pattern_summary.csv`")
    lines.append("- `pattern_summary_by_group.csv`")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rubric差分の正規表現ベース粗判定"
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
        default="codes/analysis/rubric_regex_changes",
        help="出力ディレクトリ",
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
        "--run_names",
        type=str,
        default="",
        help=(
            "run_nameフィルタ（カンマ区切り）。"
            " optimization_results形式(base_expert_...) / evaluation_results形式(zero_shot_base_expert_...) のどちらでも可"
        ),
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="対象ペア数の上限。0なら全件",
    )
    parser.add_argument(
        "--presence_only",
        action="store_true",
        help="出現回数ではなく有無だけで判定する",
    )
    parser.add_argument(
        "--patterns_json",
        type=str,
        default="",
        help="正規表現パターンをJSONで上書きする場合のパス",
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
    if not pairs:
        raise RuntimeError("Rubricペアが見つかりませんでした。")

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
        raise RuntimeError("フィルタ後に対象ペアが0件です。")

    patterns = build_patterns(args.patterns_json)
    pair_rows, summary_rows, group_rows = run_analysis(
        pairs=pairs,
        patterns=patterns,
        presence_only=args.presence_only,
    )

    write_csv(output_dir / "pair_pattern_results.csv", pair_rows)
    write_csv(output_dir / "pattern_summary.csv", summary_rows)
    write_csv(output_dir / "pattern_summary_by_group.csv", group_rows)

    (output_dir / "patterns_used.json").write_text(
        json.dumps([p.__dict__ for p in patterns], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_md = build_summary_md(
        pair_count=len(pairs),
        summary_rows=summary_rows,
        group_rows=group_rows,
        presence_only=args.presence_only,
    )
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
