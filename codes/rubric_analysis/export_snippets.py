#!/usr/bin/env python3
"""
全 rubric からパターンマッチのスニペットを抽出し、
パターンごとに all_patterns.txt へ書き出す。

各スニペットは空行2つで区切られ、目視で確認・選択できる。

使用例:
  python export_snippets.py --config base_expert_True_train100_iteration5_top3_bs4-8-12_mc4
  python export_snippets.py --config base_expert_True_train100_iteration5_top3_bs4-8-12_mc4 --output-dir ./snippets
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from pattern_config import PATTERNS
from search_pattern import find_run_dirs, get_enabled_patterns


CONTEXT_CHARS = 40
MAX_LEN = 120


def extract_all_snippets(
    text: str, regex: str, context_chars: int = CONTEXT_CHARS, max_len: int = MAX_LEN
) -> list[str]:
    """テキスト中の全マッチについて周辺コンテキスト付きスニペットを返す。"""
    snippets: list[str] = []
    seen: set[str] = set()

    for m in re.finditer(regex, text, re.IGNORECASE):
        start = max(0, m.start() - context_chars)
        end = min(len(text), m.end() + context_chars)
        snippet = text[start:end].replace('\n', ' ').strip()

        # max_len を超える場合はマッチ中心でトリム
        if len(snippet) > max_len:
            offset = m.start() - start
            mid = offset + len(m.group()) // 2
            half = max_len // 2
            s = max(0, mid - half)
            snippet = snippet[s:s + max_len].strip()

        prefix = "\u2026" if start > 0 else ""
        suffix = "\u2026" if end < len(text) else ""
        snippet = prefix + snippet + suffix

        # 重複除去
        if snippet not in seen:
            seen.add(snippet)
            snippets.append(snippet)

    return snippets


def main():
    parser = argparse.ArgumentParser(
        description="パターンマッチのスニペットをタイプ別テキストファイルに書き出す",
    )
    parser.add_argument(
        "--config", required=True,
        help="Run config 名 (例: base_expert_True_train100_iteration5_top3_bs4-8-12_mc4)",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="optimization_results ディレクトリへのパス (省略時は自動検出)",
    )
    parser.add_argument("--dataset", default=None, help="データセットでフィルタ")
    parser.add_argument("--model", default=None, help="モデルでフィルタ")
    parser.add_argument(
        "--output-dir", "-o", default=None,
        help="出力ディレクトリ (省略時はスクリプトと同階層の snippets/)",
    )
    args = parser.parse_args()

    # ── パス解決 ──
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.results_dir:
        results_dirs = [Path(args.results_dir)]
    else:
        candidates = [
            script_dir.parent / "optimization_results",
            project_root / "optimization_results",
        ]
        results_dirs = [c for c in candidates if c.exists()]
        if not results_dirs:
            print("Error: optimization_results ディレクトリが見つかりません。--results-dir を指定してください。")
            return

    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "snippets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── パターン読込 ──
    patterns = get_enabled_patterns()
    print(f"Loaded {len(patterns)} enabled patterns")

    # ── run 探索 ──
    all_runs: list[dict] = []
    seen_keys: set[str] = set()
    for rd in results_dirs:
        for run in find_run_dirs(rd, args.config):
            key = f"{run['dataset']}/{run['model']}"
            if key not in seen_keys:
                seen_keys.add(key)
                all_runs.append(run)

    if args.dataset:
        all_runs = [r for r in all_runs if r["dataset"] == args.dataset]
    if args.model:
        all_runs = [r for r in all_runs if r["model"] == args.model]

    if not all_runs:
        print(f"Error: config '{args.config}' にマッチする run が見つかりません。")
        return

    print(f"Found {len(all_runs)} matching run(s)")

    # ── スニペット収集 ──
    # pattern_id -> list[(source, snippet)]
    by_pattern: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for run in all_runs:
        best_path = run["path"] / "best_rubric.txt"
        if not best_path.exists():
            continue
        best_text = best_path.read_text(encoding="utf-8")
        source = f"{run['dataset']}/{run['model']}"

        for p in patterns:
            snippets = extract_all_snippets(best_text, p["regex"])
            for s in snippets:
                by_pattern[p["pattern_id"]].append((source, s))

    # ── 書き出し ──
    pat_lookup = {p["pattern_id"]: p for p in patterns}
    out_path = output_dir / "all_patterns.txt"
    lines: list[str] = []

    for pid in sorted(by_pattern.keys()):
        p = pat_lookup[pid]
        entries = by_pattern[pid]
        lines.append(f"{'=' * 70}")
        lines.append(f"  {p['name_en']}  ({pid})")
        lines.append(f"  regex: {p['regex']}")
        lines.append(f"  {len(entries)} snippet(s)")
        lines.append(f"{'=' * 70}")
        lines.append("")

        for source, snippet in entries:
            lines.append(f"[{source}]")
            lines.append(snippet)
            lines.append("")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    n_total = sum(len(v) for v in by_pattern.values())
    print(f"  {out_path}  ({n_total} snippets)")

    print("Done.")


if __name__ == "__main__":
    main()
