#!/usr/bin/env python3
"""
Rubric パターン分析スクリプト
Before (initial_rubric.txt) と After (best_rubric.txt) の正規表現マッチ数を比較し
LaTeX 表を出力する。

使用例:
  python search_pattern.py --config base_expert_True_train100_iteration5_top3_bs4-8-12_mc4
  python search_pattern.py --config base_expert_True_train100_iteration5_top3_bs4-8-12_mc4 --dataset asap_1
  python search_pattern.py --config base_expert_True_train100_iteration5_top3_bs4-8-12_mc4 --density
  python search_pattern.py --config base_expert_True_train100_iteration5_top3_bs4-8-12_mc4 --output table.tex
"""

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path

from pattern_config import PATTERNS


# ── helpers ─────────────────────────────────────────────────


def get_enabled_patterns():
    return [p for p in PATTERNS if p.get("enabled", True)]


def count_matches(text: str, regex: str) -> int:
    """テキスト中の正規表現マッチ総数を返す (case-insensitive)。"""
    return len(re.findall(regex, text, re.IGNORECASE))


def escape_latex(text: str) -> str:
    """LaTeX 特殊文字をエスケープする。"""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'),
        ('%', r'\%'),
        ('$', r'\$'),
        ('#', r'\#'),
        ('_', r'\_'),
        ('{', r'\{'),
        ('}', r'\}'),
        ('~', r'\textasciitilde{}'),
        ('^', r'\textasciicircum{}'),
    ]
    for char, repl in replacements:
        text = text.replace(char, repl)
    return text


def find_sample_snippet(
    text: str, regex: str, context_chars: int = 25, max_len: int = 60
) -> str:
    """テキストからランダムに1つのマッチを選び、周辺コンテキスト付きスニペットを返す。

    毎回の実行で異なるスニペットが選ばれる。
    """
    matches = list(re.finditer(regex, text, re.IGNORECASE))
    if not matches:
        return ""

    m = random.choice(matches)

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
    return prefix + snippet + suffix


def find_run_dirs(results_dir: Path, config_name: str) -> list[dict]:
    """config_name にマッチする全 run ディレクトリを探索する。"""
    # evaluation_results の "zero_shot_" プレフィックスを strip
    config_clean = config_name
    if config_clean.startswith("zero_shot_"):
        config_clean = config_clean[len("zero_shot_"):]

    matches = []
    if not results_dir.exists():
        return matches

    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for name in {config_clean, config_name}:
                config_dir = model_dir / name
                if config_dir.is_dir() and config_dir not in [m["path"] for m in matches]:
                    matches.append({
                        "path": config_dir,
                        "dataset": dataset_dir.name,
                        "model": model_dir.name,
                        "config": name,
                    })
    return matches


def analyze_rubric_pair(initial_path: Path, best_path: Path, patterns: list[dict]):
    """initial / best のペアについてパターンマッチ数とサンプルスニペットを返す。"""
    initial_text = initial_path.read_text(encoding="utf-8") if initial_path.exists() else ""
    best_text = best_path.read_text(encoding="utf-8") if best_path.exists() else ""

    init_len = len(initial_text)
    best_len = len(best_text)

    results = []
    samples: dict[str, str] = {}
    for p in patterns:
        bc = count_matches(initial_text, p["regex"])
        ac = count_matches(best_text, p["regex"])
        results.append({
            "pattern_id": p["pattern_id"],
            "name_en": p["name_en"],
            "before_count": bc,
            "after_count": ac,
            "delta": ac - bc,
            # density per 1k chars
            "before_density": (bc / init_len * 1000) if init_len else 0.0,
            "after_density": (ac / best_len * 1000) if best_len else 0.0,
        })
        # best rubric からランダムなサンプルスニペットを抽出
        snippet = find_sample_snippet(best_text, p["regex"])
        if snippet:
            samples[p["pattern_id"]] = snippet
    return results, init_len, best_len, samples


# ── LaTeX 出力 ──────────────────────────────────────────────

def _build_caption(
    all_results: list[list[dict]],
    patterns: list[dict],
    n_runs: int,
    use_density: bool,
    sorted_pids: list[str],
) -> str:
    """表のキャプションを動的に構築する。"""
    metric = "density (per 1\\,k chars)" if use_density else "match counts"

    # パターン説明を構築 (表と同じ順序)
    pid_to_pat = {p["pattern_id"]: p for p in patterns}
    pat_strs: list[str] = []
    for pid in sorted_pids:
        p = pid_to_pat.get(pid)
        if p:
            cues = p.get("cues_en", "")
            pat_strs.append(
                rf"\textbf{{{p['name_en']}}} (\textit{{{cues}}})"
            )
    pattern_note = "; ".join(pat_strs) + "."

    caption = (
        rf"Regex-based {metric} in human-authored rubrics "
        rf"(\textbf{{Before}}) vs.\ iteratively refined rubrics "
        rf"(\textbf{{After}}), averaged over {n_runs} "
        rf"run{'s' if n_runs > 1 else ''}. "
        rf"Each pattern is detected by case-insensitive keyword matching. "
        rf"{pattern_note} "
        rf"$\Delta$ = After $-$ Before."
    )
    return caption


def generate_latex_table(
    all_results: list[list[dict]],
    config_name: str,
    patterns: list[dict],
    use_density: bool = False,
    all_samples: dict[str, str] | None = None,
) -> str:
    """集約された結果から LaTeX booktabs 表を生成する。"""
    n_runs = len(all_results)
    if all_samples is None:
        all_samples = {}

    # パターンごとに集約
    agg: dict[str, dict[str, list]] = defaultdict(
        lambda: {"before": [], "after": [], "delta": [],
                 "before_d": [], "after_d": []}
    )
    for run_res in all_results:
        for r in run_res:
            pid = r["pattern_id"]
            agg[pid]["before"].append(r["before_count"])
            agg[pid]["after"].append(r["after_count"])
            agg[pid]["delta"].append(r["delta"])
            agg[pid]["before_d"].append(r["before_density"])
            agg[pid]["after_d"].append(r["after_density"])

    # パターンメタ情報 (先頭 run から)
    meta = {r["pattern_id"]: r for r in all_results[0]}

    # After の平均値が大きい順にソート
    def _after_avg(pid):
        a = agg[pid]
        if use_density:
            return sum(a["after_d"]) / n_runs
        return sum(a["after"]) / n_runs

    sorted_pids = sorted(meta.keys(), key=_after_avg, reverse=True)

    caption = _build_caption(all_results, patterns, n_runs, use_density, sorted_pids)

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    # lines.append(r"\small")
    lines.append(r"\begin{tabular}{lp{5.5cm}rrr}")
    lines.append(r"\toprule")
    lines.append(r"Pattern & Example Snippet & Before & After & $\Delta$ \\")
    lines.append(r"\midrule")

    for pid in sorted_pids:
        pm = meta[pid]
        a = agg[pid]

        if use_density:
            bv = sum(a["before_d"]) / n_runs
            av = sum(a["after_d"]) / n_runs
        else:
            bv = sum(a["before"]) / n_runs
            av = sum(a["after"]) / n_runs
        dv = av - bv

        bs = f"{bv:.1f}"
        as_ = f"{av:.1f}"
        ds = f"{dv:+.1f}"

        snippet_raw = all_samples.get(pid, "")
        snippet_tex = rf"\textit{{{escape_latex(snippet_raw)}}}" if snippet_raw else ""
        lines.append(
            f"{pm['name_en']} & {snippet_tex} & {bs} & {as_} & {ds} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\label{tab:rubric_pattern_changes}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_per_run_table(
    all_results: list[list[dict]],
    run_infos: list[dict],
) -> str:
    """run ごとの詳細テキストテーブルを返す (stdout 表示用)。"""
    out: list[str] = []
    for run_res, info in zip(all_results, run_infos):
        header = f"{info['dataset']}/{info['model']}"
        out.append(f"\n{'─' * 70}")
        out.append(f"  {header}")
        out.append(f"{'─' * 70}")
        out.append(f"  {'Pattern':<30s}  {'Before':>6s}  {'After':>6s}  {'Delta':>6s}")
        out.append(f"  {'─' * 54}")
        for r in run_res:
            out.append(
                f"  {r['name_en']:<30s}  {r['before_count']:>6d}  "
                f"{r['after_count']:>6d}  {r['delta']:>+6d}"
            )
    return "\n".join(out)


# ── main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rubric パターン変化分析: Before/After の正規表現マッチ数を比較して LaTeX 表を出力",
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
        "--density", action="store_true",
        help="マッチ数の代わりに密度 (per 1k chars) を使用",
    )
    parser.add_argument("--output", "-o", default=None, help="LaTeX 出力ファイルパス")
    parser.add_argument("--verbose", "-v", action="store_true", help="run 毎の詳細表示")
    args = parser.parse_args()

    # ── パス解決 ──
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.results_dir:
        results_dirs = [Path(args.results_dir)]
    else:
        # codes/ の中にスクリプトがあるので、同階層の optimization_results を優先
        candidates = [
            script_dir.parent / "optimization_results",   # codes/optimization_results
            project_root / "optimization_results",         # プロジェクト直下
        ]
        results_dirs = [c for c in candidates if c.exists()]
        if not results_dirs:
            print("Error: optimization_results ディレクトリが見つかりません。--results-dir を指定してください。")
            return

    # ── パターン読込 ──
    patterns = get_enabled_patterns()
    print(f"Loaded {len(patterns)} enabled patterns")

    # ── run 探索 (dataset+model で重複排除) ──
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

    print(f"Found {len(all_runs)} matching run(s):")
    for r in all_runs:
        print(f"  {r['dataset']}/{r['model']}/{r['config']}")

    # ── 分析 ──
    all_results: list[list[dict]] = []
    # pattern_id -> list[snippet] : 全 run から候補を集め最後にランダム選択
    sample_candidates: dict[str, list[str]] = defaultdict(list)
    valid_runs: list[dict] = []
    for run in all_runs:
        initial = run["path"] / "initial_rubric.txt"
        best = run["path"] / "best_rubric.txt"
        if not initial.exists() or not best.exists():
            print(f"  ⚠ rubric ファイル不足: {run['path']}")
            continue
        results, il, bl, samples = analyze_rubric_pair(initial, best, patterns)
        all_results.append(results)
        valid_runs.append(run)
        for pid, snippet in samples.items():
            sample_candidates[pid].append(snippet)
        if args.verbose:
            print(f"  {run['dataset']}/{run['model']}  (init={il} chars, best={bl} chars)")

    if not all_results:
        print("Error: 有効な rubric ペアが見つかりませんでした。")
        return

    # ── verbose: run 毎テーブル ──
    if args.verbose:
        print(generate_per_run_table(all_results, valid_runs))

    # ── 各パターンのサンプルスニペットをランダムに1つ選択 ──
    all_samples: dict[str, str] = {
        pid: random.choice(candidates)
        for pid, candidates in sample_candidates.items()
    }

    # ── LaTeX 出力 ──
    latex = generate_latex_table(
        all_results, args.config, patterns,
        use_density=args.density, all_samples=all_samples,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex, encoding="utf-8")
        print(f"\nLaTeX table → {out_path}")
    else:
        print(f"\n{'=' * 60}")
        print("LaTeX Output:")
        print(f"{'=' * 60}")
        print(latex)


if __name__ == "__main__":
    main()
