"""
実験コスト分析スクリプト

OpenRouterのアクティビティCSVから、runごとのトークン数・コストを集計し、
設定間の比較（iteration数、MC数、モデル、データセット等）を出力する。

Usage:
    python analyze_cost.py [--csv PATH] [--output_dir PATH]
"""

import argparse
import csv
import re
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# =============================================================================
# モデルごとの価格設定 (USD per 1M tokens)
# TODO: 実際の価格に置き換えてください
# =============================================================================
MODEL_PRICING = {
    "openai/gpt-5-mini": {
        "input": 0.250,       # USD per 1M input tokens
        "output": 2.000,      # USD per 1M output tokens
        "reasoning": 2.000,   # USD per 1M reasoning tokens (= output扱い)
        "cached_input": 0.025,  # USD per 1M cached input tokens
    },
    "qwen/qwen3-next-80b-a3b-instruct": {
        "input": 0.15,
        "output": 1.20,
        "reasoning": 1.20,
        "cached_input": 0.15,
    },
    "google/gemini-3-flash-preview": {
        "input": 0.50,
        "output": 3.00,
        "reasoning": 3.00,
        "cached_input": 0.50,
    },
}

# model_permaslug → 正規化モデル名 のマッピング
_MODEL_SLUG_MAP: dict[str, str] = {}


def _normalize_model(permaslug: str) -> str:
    """model_permaslug (例: openai/gpt-5-mini-2025-08-07) を正規化名に変換"""
    if permaslug in _MODEL_SLUG_MAP:
        return _MODEL_SLUG_MAP[permaslug]
    for canonical in MODEL_PRICING:
        if permaslug.startswith(canonical):
            _MODEL_SLUG_MAP[permaslug] = canonical
            return canonical
    # フォールバック: "/" 以降のバージョン番号を除去
    _MODEL_SLUG_MAP[permaslug] = permaslug
    return permaslug


# =============================================================================
# user フィールドのパース
# =============================================================================

# OpenRouter の user フィールド長制限により truncate + hash が付く場合がある
OPENROUTER_USER_MAX_LEN = 128
SESSION_GAP_MINUTES = 90
SESSION_GAP = timedelta(minutes=SESSION_GAP_MINUTES)


def _reverse_truncated_user(user: str) -> str:
    """
    llm_router.py の _normalize_openrouter_user と同じロジックで
    truncate されたかどうかを推定するが、元文字列が不明なため
    truncated なものはそのまま返す（後でグルーピング時に hash suffix で識別）
    """
    return user


def _parse_created_at(created_at: str) -> datetime | None:
    """CSV の created_at を datetime に変換（失敗時は None）"""
    if not created_at:
        return None
    value = created_at.strip()
    if not value:
        return None

    # 例: "2026-02-18 06:28:35.429", "2026-02-18T06:28:35.429Z"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_user_field(user: str) -> dict:
    """
    user フィールドから実験設定を抽出する。

    パターン:
    - Optimization: {dataset}_{model}_{method}_{seed}_{rationale}_train{n}_iteration{n}_top{k}_bs{bs}_mc{mc}
    - Evaluation:   eval_{dataset}_{model}_{method}_{opt_method}_{seed}_{rationale}_train{n}_iteration{n}_top{k}_bs{bs}_mc{mc}
    - Cross-model:  eval_cross_{dataset}_{eval_model}_from_{opt_model}_{method}_{opt_method}_{seed}_{rationale}_train{n}_...
    """
    info = {"raw_user": user, "phase": "unknown"}

    # ---- Phase detection ----
    if user.startswith("eval_cross_"):
        info["phase"] = "eval_cross"
    elif user.startswith("eval_"):
        info["phase"] = "eval"
    else:
        info["phase"] = "optimization"

    # ---- 共通パラメータ抽出 (正規表現) ----
    m = re.search(r"_train(\d+)", user)
    if m:
        info["train_size"] = int(m.group(1))

    m = re.search(r"_iteration(\d+)", user)
    if m:
        info["iteration"] = int(m.group(1))

    m = re.search(r"_top(\d+)", user)
    if m:
        info["top_k"] = int(m.group(1))

    m = re.search(r"_bs([\d\-]+)", user)
    if m:
        info["batch_sizes"] = m.group(1)

    m = re.search(r"_mc(\d+)", user)
    if m:
        info["monte_carlo"] = int(m.group(1))

    # with_rationale
    if "_True_" in user:
        info["with_rationale"] = True
    elif "_False_" in user:
        info["with_rationale"] = False

    # seed_prompt
    for sp in ["expert", "simplest", "simple", "self"]:
        if f"_{sp}_" in user:
            info["seed_prompt"] = sp
            break

    # dataset
    for ds in ["ASAP2", "asap_1", "ets3", "ets"]:
        if ds in user:
            info["dataset"] = ds
            break

    # model (推定: user フィールド内のモデル名部分)
    for model_key in ["openai_gpt-5-mini", "qwen_qwen3-next-80b-a3b-instruct", "google_gemini-3-flash-preview"]:
        if model_key in user:
            info["model"] = model_key.replace("_", "/", 1)
            break

    # eval_cross の場合: from_{opt_model} を抽出
    if info["phase"] == "eval_cross":
        m = re.search(r"_from_(openai_gpt-5-mini|qwen_qwen3-next-80b-a3b-instruct|google_gemini-3-flash-preview)", user)
        if m:
            info["opt_model"] = m.group(1).replace("_", "/", 1)

    # eval method
    for method in ["few_shot", "zero_shot"]:
        if method in user:
            info["eval_method"] = method
            break

    return info


# =============================================================================
# コスト計算
# =============================================================================

def calc_cost(tokens_prompt: int, tokens_completion: int, tokens_reasoning: int,
              tokens_cached: int, model: str) -> float:
    """1件のAPI呼び出しのコストを計算 (USD)"""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0

    # cached tokens は input tokens に含まれている場合がある
    # 実際の non-cached input = tokens_prompt - tokens_cached
    non_cached_input = max(0, tokens_prompt - tokens_cached)
    cost = (
        non_cached_input * pricing["input"] / 1_000_000
        + tokens_cached * pricing["cached_input"] / 1_000_000
        + tokens_completion * pricing["output"] / 1_000_000
        + tokens_reasoning * pricing["reasoning"] / 1_000_000
    )
    return cost


# =============================================================================
# メイン集計
# =============================================================================

def load_and_aggregate(csv_path: str) -> dict:
    """
    CSVを読み込み、run (user フィールド) ごとに集計する。
    ただし同一 user でも created_at の差が 90 分以上あれば別 run として扱う。

    Returns:
        dict: run_key -> {
            "info": parsed user info,
            "tokens_prompt": int,
            "tokens_completion": int,
            "tokens_reasoning": int,
            "tokens_cached": int,
            "cost_usd": float,
            "byok_usage": float,
            "api_calls": int,
        }
    """
    rows_by_user: dict[str, list[dict]] = defaultdict(list)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            user = row.get("user", "")
            if not user:
                continue

            tp = int(row.get("tokens_prompt", 0) or 0)
            tc = int(row.get("tokens_completion", 0) or 0)
            tr = int(row.get("tokens_reasoning", 0) or 0)
            tca = int(row.get("tokens_cached", 0) or 0)
            byok = float(row.get("byok_usage_inference", 0) or 0)
            model_slug = row.get("model_permaslug", "")
            model = _normalize_model(model_slug)

            cost = calc_cost(tp, tc, tr, tca, model)
            created_at = _parse_created_at(row.get("created_at", ""))

            rows_by_user[user].append({
                "row_idx": row_idx,
                "created_at": created_at,
                "tokens_prompt": tp,
                "tokens_completion": tc,
                "tokens_reasoning": tr,
                "tokens_cached": tca,
                "cost_usd": cost,
                "byok_usage": byok,
            })

    runs: dict[str, dict] = {}
    for user, user_rows in rows_by_user.items():
        ordered_rows = sorted(
            user_rows,
            key=lambda r: (r["created_at"] is None, r["created_at"] or datetime.min, r["row_idx"]),
        )

        sessions: list[list[dict]] = []
        current_session: list[dict] = []
        prev_created_at: datetime | None = None

        for r in ordered_rows:
            current_created_at = r["created_at"]
            if (
                current_session
                and current_created_at is not None
                and prev_created_at is not None
                and current_created_at - prev_created_at >= SESSION_GAP
            ):
                sessions.append(current_session)
                current_session = []

            current_session.append(r)
            if current_created_at is not None:
                prev_created_at = current_created_at

        if current_session:
            sessions.append(current_session)

        base_info = parse_user_field(user)
        user_is_split = len(sessions) > 1

        for session_idx, session_rows in enumerate(sessions, start=1):
            if user_is_split:
                session_start = next(
                    (x["created_at"] for x in session_rows if x["created_at"] is not None),
                    None,
                )
                if session_start is not None:
                    suffix = session_start.strftime("%Y%m%d%H%M")
                else:
                    suffix = f"row{session_rows[0]['row_idx']}"
                run_key = f"{user}__session{session_idx}_{suffix}"
            else:
                run_key = user

            entry = {
                "info": dict(base_info),
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "tokens_reasoning": 0,
                "tokens_cached": 0,
                "cost_usd": 0.0,
                "byok_usage": 0.0,
                "api_calls": 0,
            }
            for r in session_rows:
                entry["tokens_prompt"] += r["tokens_prompt"]
                entry["tokens_completion"] += r["tokens_completion"]
                entry["tokens_reasoning"] += r["tokens_reasoning"]
                entry["tokens_cached"] += r["tokens_cached"]
                entry["cost_usd"] += r["cost_usd"]
                entry["byok_usage"] += r["byok_usage"]
                entry["api_calls"] += 1
            runs[run_key] = entry

    return runs


# =============================================================================
# レポート生成
# =============================================================================

def format_number(n: int) -> str:
    """数値を読みやすい形式に"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def print_run_summary(runs: dict, output_dir: Path | None = None):
    """run ごとのサマリーテーブル"""
    lines = []
    lines.append("=" * 120)
    lines.append("RUN-LEVEL COST SUMMARY")
    lines.append("=" * 120)

    # Phase ごとにグループ
    for phase in ["optimization", "eval", "eval_cross"]:
        phase_runs = {k: v for k, v in runs.items() if v["info"].get("phase") == phase}
        if not phase_runs:
            continue

        lines.append(f"\n--- {phase.upper()} RUNS ({len(phase_runs)} runs) ---")
        lines.append(
            f"{'Run':<85} {'Calls':>6} {'Input':>10} {'Output':>10} {'Reason':>10} {'Cost($)':>10} {'BYOK($)':>10}"
        )
        lines.append("-" * 145)

        total_cost = 0.0
        total_byok = 0.0
        total_calls = 0
        total_input = 0
        total_output = 0
        total_reasoning = 0

        for user_key in sorted(phase_runs.keys()):
            r = phase_runs[user_key]
            display_key = user_key[:83] + ".." if len(user_key) > 85 else user_key
            lines.append(
                f"{display_key:<85} {r['api_calls']:>6} "
                f"{format_number(r['tokens_prompt']):>10} "
                f"{format_number(r['tokens_completion']):>10} "
                f"{format_number(r['tokens_reasoning']):>10} "
                f"{r['cost_usd']:>10.4f} "
                f"{r['byok_usage']:>10.4f}"
            )
            total_cost += r["cost_usd"]
            total_byok += r["byok_usage"]
            total_calls += r["api_calls"]
            total_input += r["tokens_prompt"]
            total_output += r["tokens_completion"]
            total_reasoning += r["tokens_reasoning"]

        lines.append("-" * 145)
        lines.append(
            f"{'SUBTOTAL':<85} {total_calls:>6} "
            f"{format_number(total_input):>10} "
            f"{format_number(total_output):>10} "
            f"{format_number(total_reasoning):>10} "
            f"{total_cost:>10.4f} "
            f"{total_byok:>10.4f}"
        )

    # 全体合計
    grand_cost = sum(r["cost_usd"] for r in runs.values())
    grand_byok = sum(r["byok_usage"] for r in runs.values())
    grand_calls = sum(r["api_calls"] for r in runs.values())
    grand_input = sum(r["tokens_prompt"] for r in runs.values())
    grand_output = sum(r["tokens_completion"] for r in runs.values())
    grand_reasoning = sum(r["tokens_reasoning"] for r in runs.values())
    lines.append("\n" + "=" * 120)
    lines.append(
        f"{'GRAND TOTAL':<85} {grand_calls:>6} "
        f"{format_number(grand_input):>10} "
        f"{format_number(grand_output):>10} "
        f"{format_number(grand_reasoning):>10} "
        f"{grand_cost:>10.4f} "
        f"{grand_byok:>10.4f}"
    )
    lines.append("=" * 120)

    report = "\n".join(lines)
    print(report)
    if output_dir:
        (output_dir / "run_summary.txt").write_text(report, encoding="utf-8")


def print_comparison_by_setting(runs: dict, output_dir: Path | None = None):
    """設定ごとの比較分析"""
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("COST COMPARISON BY SETTINGS")
    lines.append("=" * 120)

    # optimization runs のみ対象（evalは別途）
    opt_runs = {k: v for k, v in runs.items() if v["info"].get("phase") == "optimization"}

    # --- 1. モデル別 ---
    lines.append("\n[1] BY MODEL")
    _compare_by_key(opt_runs, "model", lines)

    # --- 2. データセット別 ---
    lines.append("\n[2] BY DATASET")
    _compare_by_key(opt_runs, "dataset", lines)

    # --- 3. Iteration数別 ---
    lines.append("\n[3] BY ITERATION COUNT")
    _compare_by_key(opt_runs, "iteration", lines)

    # --- 4. Monte Carlo runs別 ---
    lines.append("\n[4] BY MONTE CARLO RUNS")
    _compare_by_key(opt_runs, "monte_carlo", lines)

    # --- 5. Seed prompt別 ---
    lines.append("\n[5] BY SEED PROMPT")
    _compare_by_key(opt_runs, "seed_prompt", lines)

    # --- 6. With rationale別 ---
    lines.append("\n[6] BY WITH_RATIONALE")
    _compare_by_key(opt_runs, "with_rationale", lines)

    # --- 7. Eval runs: method 別 ---
    eval_runs = {k: v for k, v in runs.items() if v["info"].get("phase") == "eval"}
    if eval_runs:
        lines.append("\n[7] EVAL RUNS BY METHOD")
        _compare_by_key(eval_runs, "eval_method", lines)

        lines.append("\n[8] EVAL RUNS BY MODEL")
        _compare_by_key(eval_runs, "model", lines)

    report = "\n".join(lines)
    print(report)
    if output_dir:
        (output_dir / "comparison_by_setting.txt").write_text(report, encoding="utf-8")


def _compare_by_key(runs: dict, key: str, lines: list):
    """特定のキーでグルーピングして比較"""
    groups: dict[str, list] = defaultdict(list)
    for user_key, r in runs.items():
        val = r["info"].get(key, "N/A")
        groups[str(val)].append(r)

    lines.append(
        f"  {'Value':<45} {'Runs':>5} {'Calls':>8} {'Input':>12} {'Output':>12} "
        f"{'Reason':>12} {'Cost($)':>10} {'BYOK($)':>10} {'Avg Cost/Run':>12}"
    )
    lines.append("  " + "-" * 130)

    for val in sorted(groups.keys()):
        group = groups[val]
        n_runs = len(group)
        total_calls = sum(r["api_calls"] for r in group)
        total_input = sum(r["tokens_prompt"] for r in group)
        total_output = sum(r["tokens_completion"] for r in group)
        total_reasoning = sum(r["tokens_reasoning"] for r in group)
        total_cost = sum(r["cost_usd"] for r in group)
        total_byok = sum(r["byok_usage"] for r in group)
        avg_cost = total_cost / n_runs if n_runs > 0 else 0

        lines.append(
            f"  {val:<45} {n_runs:>5} {total_calls:>8} "
            f"{format_number(total_input):>12} "
            f"{format_number(total_output):>12} "
            f"{format_number(total_reasoning):>12} "
            f"{total_cost:>10.4f} "
            f"{total_byok:>10.4f} "
            f"{avg_cost:>12.4f}"
        )


def print_iteration_cost_breakdown(runs: dict, output_dir: Path | None = None):
    """
    iteration数によるコスト増加の詳細分析。
    同一条件で iteration だけが異なるペアを見つけて比較する。
    """
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("ITERATION COST SCALING ANALYSIS")
    lines.append("=" * 120)
    lines.append("同一条件でiteration数のみ異なるrunの比較:\n")

    opt_runs = {k: v for k, v in runs.items() if v["info"].get("phase") == "optimization"}

    # グルーピングキー: (dataset, model, seed_prompt, with_rationale, batch_sizes, mc)
    config_groups: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for user_key, r in opt_runs.items():
        info = r["info"]
        group_key = (
            info.get("dataset"),
            info.get("model"),
            info.get("seed_prompt"),
            info.get("with_rationale"),
            info.get("batch_sizes"),
            info.get("monte_carlo"),
        )
        iteration = info.get("iteration")
        if iteration is not None:
            config_groups[group_key][iteration] = r

    found_pairs = False
    for group_key, iter_map in sorted(config_groups.items()):
        if len(iter_map) < 2:
            continue
        found_pairs = True
        ds, model, seed, rationale, bs, mc = group_key
        lines.append(f"  Config: dataset={ds}, model={model}, seed={seed}, rationale={rationale}, bs={bs}, mc={mc}")

        sorted_iters = sorted(iter_map.keys())
        lines.append(
            f"    {'Iteration':>10} {'Calls':>8} {'Input':>12} {'Output':>12} "
            f"{'Reason':>12} {'Cost($)':>10} {'BYOK($)':>10} {'vs prev':>10}"
        )
        prev_cost = None
        for it in sorted_iters:
            r = iter_map[it]
            ratio = ""
            if prev_cost and prev_cost > 0:
                ratio = f"x{r['cost_usd'] / prev_cost:.2f}"
            lines.append(
                f"    {it:>10} {r['api_calls']:>8} "
                f"{format_number(r['tokens_prompt']):>12} "
                f"{format_number(r['tokens_completion']):>12} "
                f"{format_number(r['tokens_reasoning']):>12} "
                f"{r['cost_usd']:>10.4f} "
                f"{r['byok_usage']:>10.4f} "
                f"{ratio:>10}"
            )
            prev_cost = r["cost_usd"]
        lines.append("")

    if not found_pairs:
        lines.append("  (iteration数が異なるrun のペアが見つかりませんでした)")

    report = "\n".join(lines)
    print(report)
    if output_dir:
        (output_dir / "iteration_scaling.txt").write_text(report, encoding="utf-8")


def print_phase_summary(runs: dict, output_dir: Path | None = None):
    """Optimization vs Evaluation のコスト比較"""
    lines = []
    lines.append("\n" + "=" * 120)
    lines.append("COST BY PHASE (Optimization vs Evaluation)")
    lines.append("=" * 120)

    for phase in ["optimization", "eval", "eval_cross"]:
        phase_runs = [r for r in runs.values() if r["info"].get("phase") == phase]
        if not phase_runs:
            continue
        total_cost = sum(r["cost_usd"] for r in phase_runs)
        total_byok = sum(r["byok_usage"] for r in phase_runs)
        total_calls = sum(r["api_calls"] for r in phase_runs)
        total_input = sum(r["tokens_prompt"] for r in phase_runs)
        total_output = sum(r["tokens_completion"] for r in phase_runs)
        total_reasoning = sum(r["tokens_reasoning"] for r in phase_runs)
        lines.append(
            f"  {phase:<20} runs={len(phase_runs):>3}, calls={total_calls:>6}, "
            f"input={format_number(total_input):>10}, output={format_number(total_output):>10}, "
            f"reasoning={format_number(total_reasoning):>10}, "
            f"cost=${total_cost:.4f}, byok=${total_byok:.4f}"
        )

    report = "\n".join(lines)
    print(report)
    if output_dir:
        (output_dir / "phase_summary.txt").write_text(report, encoding="utf-8")


def export_csv_summary(runs: dict, output_dir: Path):
    """run ごとのサマリーを CSV で出力"""
    out_path = output_dir / "run_cost_summary.csv"
    fieldnames = [
        "user", "phase", "dataset", "model", "seed_prompt", "with_rationale",
        "iteration", "monte_carlo", "batch_sizes", "top_k", "train_size",
        "eval_method", "opt_model",
        "api_calls", "tokens_prompt", "tokens_completion", "tokens_reasoning",
        "tokens_cached", "cost_usd", "byok_usage",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for user_key in sorted(runs.keys()):
            r = runs[user_key]
            info = r["info"]
            row = {
                "user": user_key,
                "phase": info.get("phase", ""),
                "dataset": info.get("dataset", ""),
                "model": info.get("model", ""),
                "seed_prompt": info.get("seed_prompt", ""),
                "with_rationale": info.get("with_rationale", ""),
                "iteration": info.get("iteration", ""),
                "monte_carlo": info.get("monte_carlo", ""),
                "batch_sizes": info.get("batch_sizes", ""),
                "top_k": info.get("top_k", ""),
                "train_size": info.get("train_size", ""),
                "eval_method": info.get("eval_method", ""),
                "opt_model": info.get("opt_model", ""),
                "api_calls": r["api_calls"],
                "tokens_prompt": r["tokens_prompt"],
                "tokens_completion": r["tokens_completion"],
                "tokens_reasoning": r["tokens_reasoning"],
                "tokens_cached": r["tokens_cached"],
                "cost_usd": round(r["cost_usd"], 6),
                "byok_usage": round(r["byok_usage"], 6),
            }
            writer.writerow(row)
    print(f"\nCSV exported to: {out_path}")


# =============================================================================
# エントリポイント
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment cost analysis from OpenRouter activity CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).parent / "openrouter_activity_2026-02-19.csv"),
        help="Path to OpenRouter activity CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "reports"),
        help="Directory for output reports",
    )
    args = parser.parse_args()

    csv_path = args.csv
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV: {csv_path}")
    runs = load_and_aggregate(csv_path)
    print(f"Found {len(runs)} unique runs from CSV\n")

    print_run_summary(runs, output_dir)
    print_phase_summary(runs, output_dir)
    print_comparison_by_setting(runs, output_dir)
    print_iteration_cost_breakdown(runs, output_dir)
    export_csv_summary(runs, output_dir)

    print(f"\nAll reports saved to: {output_dir}/")


if __name__ == "__main__":
    main()
