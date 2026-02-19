#!/usr/bin/env python3
"""
Geminiの構造化出力を使って、初期Rubricと最適化後Rubricの差分を自動分析するスクリプト。

処理フロー:
1. optimization_results配下から initial_rubric.txt / best_rubric.txt のペアを収集
2. 差分の粗統計（文字数・単語数・行数など）を算出
3. サンプルペアから「よく見られる差分項目」をGeminiで自動抽出（構造化出力）
4. 各ペアについて、各項目の変化方向をGeminiで判定（構造化出力）
5. 論文用に使いやすいCSV / JSONL / Markdownを出力
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Literal

import dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


RUN_NAME_PATTERN = re.compile(
    r"^(?P<optimize_method>.+)_(?P<seed_prompt>simplest|simple|expert|self)"
    r"_(?P<with_rationale>True|False)_train(?P<train_size>\d+)"
    r"_iteration(?P<iteration>\d+)_top(?P<top_k>\d+)_bs(?P<batch_sizes>[\d-]+)_mc(?P<mc_runs>\d+)$"
)


@dataclass
class RubricPair:
    pair_id: str
    dataset: str
    model_name: str
    run_name: str
    run_dir: str
    initial_path: str
    best_path: str
    optimize_method: str | None = None
    seed_prompt: str | None = None
    with_rationale: str | None = None
    train_size: int | None = None
    iteration: int | None = None
    top_k: int | None = None
    batch_sizes: str | None = None
    mc_runs: int | None = None


@dataclass
class PairMetrics:
    pair_id: str
    changed: bool
    initial_chars: int
    best_chars: int
    char_delta: int
    char_ratio: float
    initial_words: int
    best_words: int
    word_delta: int
    initial_lines: int
    best_lines: int
    line_delta: int
    similarity_ratio: float


class DifferenceItem(BaseModel):
    item_id: str = Field(description="snake_case ID。英数字とアンダースコアのみ")
    item_name_ja: str = Field(description="項目名（日本語）")
    definition_ja: str = Field(description="この項目が意味する差分の定義（日本語）")
    judge_hint_ja: str = Field(description="判定時に何を見るかの短い補助説明（日本語）")


class DiscoveryOutput(BaseModel):
    summary_ja: str
    items: list[DifferenceItem]


JudgeStatus = Literal[
    "added_or_strengthened",
    "removed_or_weakened",
    "present_both",
    "absent_both",
    "uncertain",
]


class ItemJudgement(BaseModel):
    item_id: str
    status: JudgeStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_before: str = Field(description="初期Rubric側の根拠抜粋。なければ空文字")
    evidence_after: str = Field(description="最適化後Rubric側の根拠抜粋。なければ空文字")
    note_ja: str


class PairJudgement(BaseModel):
    pair_summary_ja: str
    item_judgements: list[ItemJudgement]


DEFAULT_ITEMS = [
    DifferenceItem(
        item_id="explicit_decision_rules",
        item_name_ja="明示的な判定ルール",
        definition_ja="if/thenや条件文など、採点判断を明文化した規則が追加・強化されている",
        judge_hint_ja="条件分岐や必須条件、禁止条件の記述を見る",
    ),
    DifferenceItem(
        item_id="score_boundary_guidance",
        item_name_ja="スコア境界の具体化",
        definition_ja="隣接スコア（例: 4 vs 5）の見分け方が具体化されている",
        judge_hint_ja="境界ケース、しきい値、tie-breakerの説明を見る",
    ),
    DifferenceItem(
        item_id="stepwise_scoring_process",
        item_name_ja="ステップ式採点手順",
        definition_ja="採点を順序立てた手順・チェックリストとして提示している",
        judge_hint_ja="Step 1/2やチェックリスト形式の有無を見る",
    ),
    DifferenceItem(
        item_id="evidence_specificity_requirement",
        item_name_ja="具体例・根拠の厳格化",
        definition_ja="理由や証拠に対して、具体性や説明リンクを要求する規則が追加・強化されている",
        judge_hint_ja="具体例、根拠、説明の必須条件を見る",
    ),
    DifferenceItem(
        item_id="repetition_noncount_rule",
        item_name_ja="反復の非カウント規則",
        definition_ja="言い換えや重複主張を別理由として数えない規則がある",
        judge_hint_ja="repetitionやdouble-count禁止ルールを見る",
    ),
    DifferenceItem(
        item_id="mechanics_penalty_clarification",
        item_name_ja="文法・表記エラー処理の明確化",
        definition_ja="文法/表記エラーをどの程度減点するか、内容との優先順位が明確化されている",
        judge_hint_ja="grammar/mechanicsの減点条件や例外規則を見る",
    ),
    DifferenceItem(
        item_id="offtopic_or_summary_cap",
        item_name_ja="脱線・要約偏重への上限制御",
        definition_ja="off-topicやsummary-only回答を低スコアに制限するルールがある",
        judge_hint_ja="cap, 上限, summary-only, irrelevantの記述を見る",
    ),
    DifferenceItem(
        item_id="quantitative_thresholds",
        item_name_ja="数量しきい値の導入",
        definition_ja="理由数・例数・誤り率など数的しきい値が導入されている",
        judge_hint_ja="at least N, <= N, %などの閾値表現を見る",
    ),
]


def load_env() -> None:
    # 実行ディレクトリ側と、codes/.env の両方を読む
    dotenv.load_dotenv()
    script_path = Path(__file__).resolve()
    codes_env = script_path.parents[1] / ".env"
    if codes_env.exists():
        dotenv.load_dotenv(codes_env)


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[TRUNCATED]..."


def parse_run_name(run_name: str) -> dict[str, str | int | None]:
    match = RUN_NAME_PATTERN.match(run_name)
    if not match:
        return {
            "optimize_method": None,
            "seed_prompt": None,
            "with_rationale": None,
            "train_size": None,
            "iteration": None,
            "top_k": None,
            "batch_sizes": None,
            "mc_runs": None,
        }
    gd = match.groupdict()
    return {
        "optimize_method": gd["optimize_method"],
        "seed_prompt": gd["seed_prompt"],
        "with_rationale": gd["with_rationale"],
        "train_size": int(gd["train_size"]),
        "iteration": int(gd["iteration"]),
        "top_k": int(gd["top_k"]),
        "batch_sizes": gd["batch_sizes"],
        "mc_runs": int(gd["mc_runs"]),
    }


def normalize_item_id(item_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", item_id.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "item"


def normalize_items(items: list[DifferenceItem], item_count: int) -> list[DifferenceItem]:
    deduped: list[DifferenceItem] = []
    used_ids: set[str] = set()

    for item in items:
        normalized_id = normalize_item_id(item.item_id)
        if normalized_id in used_ids:
            suffix = 2
            while f"{normalized_id}_{suffix}" in used_ids:
                suffix += 1
            normalized_id = f"{normalized_id}_{suffix}"
        used_ids.add(normalized_id)
        deduped.append(
            DifferenceItem(
                item_id=normalized_id,
                item_name_ja=item.item_name_ja,
                definition_ja=item.definition_ja,
                judge_hint_ja=item.judge_hint_ja,
            )
        )
        if len(deduped) >= item_count:
            break

    if len(deduped) < item_count:
        for fallback in DEFAULT_ITEMS:
            if len(deduped) >= item_count:
                break
            fb_id = normalize_item_id(fallback.item_id)
            if fb_id in used_ids:
                continue
            used_ids.add(fb_id)
            deduped.append(
                DifferenceItem(
                    item_id=fb_id,
                    item_name_ja=fallback.item_name_ja,
                    definition_ja=fallback.definition_ja,
                    judge_hint_ja=fallback.judge_hint_ja,
                )
            )
    return deduped


def collect_rubric_pairs(results_root: Path) -> list[RubricPair]:
    pairs: list[RubricPair] = []
    for initial_path in results_root.rglob("initial_rubric.txt"):
        best_path = initial_path.with_name("best_rubric.txt")
        if not best_path.exists():
            continue

        try:
            rel = initial_path.relative_to(results_root)
        except ValueError:
            continue

        if len(rel.parts) < 4:
            continue

        dataset, model_name, run_name = rel.parts[0], rel.parts[1], rel.parts[2]
        run_dir = initial_path.parent
        parsed = parse_run_name(run_name)
        pair_id = f"{dataset}/{model_name}/{run_name}"

        pairs.append(
            RubricPair(
                pair_id=pair_id,
                dataset=dataset,
                model_name=model_name,
                run_name=run_name,
                run_dir=str(run_dir),
                initial_path=str(initial_path),
                best_path=str(best_path),
                optimize_method=parsed["optimize_method"],  # type: ignore[arg-type]
                seed_prompt=parsed["seed_prompt"],  # type: ignore[arg-type]
                with_rationale=parsed["with_rationale"],  # type: ignore[arg-type]
                train_size=parsed["train_size"],  # type: ignore[arg-type]
                iteration=parsed["iteration"],  # type: ignore[arg-type]
                top_k=parsed["top_k"],  # type: ignore[arg-type]
                batch_sizes=parsed["batch_sizes"],  # type: ignore[arg-type]
                mc_runs=parsed["mc_runs"],  # type: ignore[arg-type]
            )
        )
    pairs.sort(key=lambda x: x.pair_id)
    return pairs


def filter_pairs(
    pairs: list[RubricPair],
    datasets: set[str] | None,
    models: set[str] | None,
    seed_prompts: set[str] | None,
    with_rationale: set[str] | None,
) -> list[RubricPair]:
    filtered: list[RubricPair] = []
    for p in pairs:
        if datasets and p.dataset not in datasets:
            continue
        if models and p.model_name not in models:
            continue
        if seed_prompts and (p.seed_prompt or "") not in seed_prompts:
            continue
        if with_rationale and (p.with_rationale or "") not in with_rationale:
            continue
        filtered.append(p)
    return filtered


def compute_pair_metrics(pair: RubricPair) -> PairMetrics:
    before = safe_read_text(Path(pair.initial_path))
    after = safe_read_text(Path(pair.best_path))
    changed = before != after
    return PairMetrics(
        pair_id=pair.pair_id,
        changed=changed,
        initial_chars=len(before),
        best_chars=len(after),
        char_delta=len(after) - len(before),
        char_ratio=(len(after) + 1) / (len(before) + 1),
        initial_words=count_words(before),
        best_words=count_words(after),
        word_delta=count_words(after) - count_words(before),
        initial_lines=len(before.splitlines()),
        best_lines=len(after.splitlines()),
        line_delta=len(after.splitlines()) - len(before.splitlines()),
        similarity_ratio=SequenceMatcher(None, before, after).ratio(),
    )


def summarize_metrics(metrics: list[PairMetrics]) -> dict[str, object]:
    if not metrics:
        return {}

    def stats(vals: list[float]) -> dict[str, float]:
        return {
            "mean": float(mean(vals)),
            "median": float(median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
        }

    changed_count = sum(1 for m in metrics if m.changed)
    char_deltas = [m.char_delta for m in metrics]
    word_deltas = [m.word_delta for m in metrics]
    line_deltas = [m.line_delta for m in metrics]
    char_ratios = [m.char_ratio for m in metrics]
    similarity = [m.similarity_ratio for m in metrics]

    return {
        "pair_count": len(metrics),
        "changed_count": changed_count,
        "changed_rate": changed_count / len(metrics),
        "char_delta": stats([float(v) for v in char_deltas]),
        "word_delta": stats([float(v) for v in word_deltas]),
        "line_delta": stats([float(v) for v in line_deltas]),
        "char_ratio": stats(char_ratios),
        "similarity_ratio": stats(similarity),
    }


def metrics_by_group(
    pairs: list[RubricPair], metrics_map: dict[str, PairMetrics]
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[PairMetrics]] = defaultdict(list)
    for p in pairs:
        grouped[f"dataset={p.dataset}"].append(metrics_map[p.pair_id])
        if p.seed_prompt:
            grouped[f"seed_prompt={p.seed_prompt}"].append(metrics_map[p.pair_id])
        if p.with_rationale:
            grouped[f"with_rationale={p.with_rationale}"].append(metrics_map[p.pair_id])
    return {k: summarize_metrics(v) for k, v in sorted(grouped.items())}


def pick_discovery_samples(
    pairs: list[RubricPair],
    metrics_map: dict[str, PairMetrics],
    sample_size: int,
) -> list[RubricPair]:
    non_identical = [p for p in pairs if metrics_map[p.pair_id].changed]
    if not non_identical:
        return pairs[:sample_size]

    # 変化が大きいものを優先しつつ、datasetの偏りを減らす
    non_identical.sort(key=lambda p: metrics_map[p.pair_id].char_delta, reverse=True)
    by_dataset: dict[str, list[RubricPair]] = defaultdict(list)
    for p in non_identical:
        by_dataset[p.dataset].append(p)

    selected: list[RubricPair] = []
    dataset_keys = sorted(by_dataset.keys())
    idx = 0
    while len(selected) < sample_size:
        made_progress = False
        for ds in dataset_keys:
            candidates = by_dataset[ds]
            if idx < len(candidates) and len(selected) < sample_size:
                selected.append(candidates[idx])
                made_progress = True
        if not made_progress:
            break
        idx += 1
    return selected[:sample_size]


def structured_call(
    client: genai.Client,
    model_name: str,
    prompt: str,
    schema_model: type[BaseModel],
    max_output_tokens: int,
    temperature: float,
    retries: int,
    retry_wait_sec: float,
) -> BaseModel:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_model,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
            parsed = response.parsed
            if isinstance(parsed, schema_model):
                return parsed
            if isinstance(parsed, dict):
                return schema_model.model_validate(parsed)
            # parsedが空のときはtextから再パース
            text = response.text or ""
            return schema_model.model_validate_json(text)
        except Exception as err:  # noqa: BLE001
            last_err = err
            if attempt < retries:
                sleep_sec = retry_wait_sec * attempt
                print(
                    f"[WARN] Gemini structured call failed (attempt {attempt}/{retries}): {err}"
                )
                time.sleep(sleep_sec)
            else:
                break
    assert last_err is not None
    raise RuntimeError(
        "Gemini構造化出力の呼び出しに失敗しました。"
        f" model={model_name}, retries={retries}, error={type(last_err).__name__}: {last_err}"
    ) from last_err


def discover_items_with_gemini(
    client: genai.Client,
    model_name: str,
    sample_pairs: list[RubricPair],
    item_count: int,
    max_chars_per_rubric: int,
    retries: int,
    retry_wait_sec: float,
) -> DiscoveryOutput:
    payload = []
    for p in sample_pairs:
        before = truncate_text(safe_read_text(Path(p.initial_path)), max_chars_per_rubric)
        after = truncate_text(safe_read_text(Path(p.best_path)), max_chars_per_rubric)
        payload.append(
            {
                "pair_id": p.pair_id,
                "dataset": p.dataset,
                "model": p.model_name,
                "seed_prompt": p.seed_prompt,
                "with_rationale": p.with_rationale,
                "before": before,
                "after": after,
            }
        )

    prompt = f"""
あなたは研究補助者です。以下は「初期Rubric（before）」と「最適化後Rubric（after）」の複数ペアです。
よく見られる差分を、後続の自動判定で使えるチェック項目として整理してください。

要件:
- 項目数は厳密に{item_count}件
- 各項目は「1ペアを読んだときに判定可能」な粒度にする
- 内容が重複しないようにする
- item_id は snake_case（英数字とアンダースコアのみ）
- 日本語で簡潔に記述する

入力ペア(JSON):
{json.dumps(payload, ensure_ascii=False)}
"""

    result = structured_call(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema_model=DiscoveryOutput,
        max_output_tokens=4096,
        temperature=0.0,
        retries=retries,
        retry_wait_sec=retry_wait_sec,
    )
    assert isinstance(result, DiscoveryOutput)
    result.items = normalize_items(result.items, item_count)
    return result


def judge_single_pair_with_gemini(
    client: genai.Client,
    model_name: str,
    pair: RubricPair,
    items: list[DifferenceItem],
    max_chars_per_rubric: int,
    retries: int,
    retry_wait_sec: float,
) -> PairJudgement:
    before = truncate_text(safe_read_text(Path(pair.initial_path)), max_chars_per_rubric)
    after = truncate_text(safe_read_text(Path(pair.best_path)), max_chars_per_rubric)
    item_payload = [item.model_dump() for item in items]

    prompt = f"""
あなたはRubric差分の判定者です。
次のbefore/afterペアについて、指定された各項目ごとに変化方向を判定してください。

判定ラベル:
- added_or_strengthened: afterで新規追加または明確に強化
- removed_or_weakened: afterで削除または弱化
- present_both: before/afterの両方に同程度で存在
- absent_both: before/afterの両方で見当たらない
- uncertain: 情報不足や曖昧で断定困難

注意:
- item_idは入力項目にあるものをそのまま使う
- confidenceは0.0〜1.0
- evidence_before/evidence_afterは短い抜粋（なければ空文字）
- note_jaは簡潔に

対象項目:
{json.dumps(item_payload, ensure_ascii=False)}

Rubric before:
{before}

Rubric after:
{after}
"""

    result = structured_call(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema_model=PairJudgement,
        max_output_tokens=8192,
        temperature=0.0,
        retries=retries,
        retry_wait_sec=retry_wait_sec,
    )
    assert isinstance(result, PairJudgement)
    return result


def normalize_pair_judgement(
    raw: PairJudgement, items: list[DifferenceItem]
) -> PairJudgement:
    expected_ids = [i.item_id for i in items]
    by_id: dict[str, ItemJudgement] = {}
    for j in raw.item_judgements:
        if j.item_id in expected_ids and j.item_id not in by_id:
            by_id[j.item_id] = j

    normalized = []
    for item in items:
        if item.item_id in by_id:
            normalized.append(by_id[item.item_id])
            continue
        normalized.append(
            ItemJudgement(
                item_id=item.item_id,
                status="uncertain",
                confidence=0.0,
                evidence_before="",
                evidence_after="",
                note_ja="出力欠落のため補完",
            )
        )
    raw.item_judgements = normalized
    return raw


def aggregate_item_results(
    pair_results: list[dict[str, object]],
    items: list[DifferenceItem],
) -> list[dict[str, object]]:
    status_keys = [
        "added_or_strengthened",
        "removed_or_weakened",
        "present_both",
        "absent_both",
        "uncertain",
    ]
    counter: dict[str, Counter[str]] = {item.item_id: Counter() for item in items}
    total_pairs = len(pair_results)

    for rec in pair_results:
        item_judgements = rec.get("item_judgements", [])
        assert isinstance(item_judgements, list)
        for j in item_judgements:
            status = j.get("status")
            item_id = j.get("item_id")
            if isinstance(item_id, str) and isinstance(status, str):
                counter[item_id][status] += 1

    item_map = {i.item_id: i for i in items}
    rows: list[dict[str, object]] = []
    for item_id, cnt in counter.items():
        row: dict[str, object] = {
            "item_id": item_id,
            "item_name_ja": item_map[item_id].item_name_ja if item_id in item_map else "",
            "definition_ja": item_map[item_id].definition_ja if item_id in item_map else "",
            "pair_count": total_pairs,
        }
        for key in status_keys:
            row[key] = cnt[key]
            row[f"{key}_rate"] = (cnt[key] / total_pairs) if total_pairs else 0.0
        rows.append(row)
    rows.sort(key=lambda r: r["added_or_strengthened"], reverse=True)  # type: ignore[arg-type]
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_num(v: float) -> str:
    return f"{v:.3f}"


def build_summary_markdown(
    metric_summary_all: dict[str, object],
    metric_summary_groups: dict[str, dict[str, object]],
    item_summary_rows: list[dict[str, object]],
    discovery: DiscoveryOutput,
) -> str:
    lines: list[str] = []
    lines.append("# Rubric差分分析サマリ\n")
    lines.append("## 1. 粗い定量比較\n")

    pair_count = metric_summary_all.get("pair_count", 0)
    changed_count = metric_summary_all.get("changed_count", 0)
    changed_rate = metric_summary_all.get("changed_rate", 0.0)
    lines.append(
        f"- 対象ペア数: **{pair_count}**（変化あり: **{changed_count}** / {format_num(float(changed_rate))}）"
    )

    def append_metric_block(name: str, key: str) -> None:
        block = metric_summary_all.get(key, {})
        if not isinstance(block, dict):
            return
        lines.append(
            f"- {name}: mean={format_num(float(block.get('mean', 0.0)))}, "
            f"median={format_num(float(block.get('median', 0.0)))}, "
            f"min={format_num(float(block.get('min', 0.0)))}, "
            f"max={format_num(float(block.get('max', 0.0)))}"
        )

    append_metric_block("文字数差分(after-before)", "char_delta")
    append_metric_block("単語数差分(after-before)", "word_delta")
    append_metric_block("行数差分(after-before)", "line_delta")
    append_metric_block("文字数比(after/before)", "char_ratio")
    append_metric_block("文字列類似度(SequenceMatcher)", "similarity_ratio")

    lines.append("\n## 2. LLMが抽出した差分項目\n")
    lines.append(f"- 要約: {discovery.summary_ja}")
    for item in discovery.items:
        lines.append(f"- `{item.item_id}`: **{item.item_name_ja}** - {item.definition_ja}")

    lines.append("\n## 3. 項目別の出現傾向（LLM-as-a-Judge）\n")
    if item_summary_rows:
        lines.append(
            "| item_id | 項目名 | added/strengthened | present_both | removed/weakened | absent_both | uncertain |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in item_summary_rows:
            lines.append(
                f"| `{row['item_id']}` | {row['item_name_ja']} | "
                f"{row['added_or_strengthened']} ({format_num(float(row['added_or_strengthened_rate']))}) | "
                f"{row['present_both']} ({format_num(float(row['present_both_rate']))}) | "
                f"{row['removed_or_weakened']} ({format_num(float(row['removed_or_weakened_rate']))}) | "
                f"{row['absent_both']} ({format_num(float(row['absent_both_rate']))}) | "
                f"{row['uncertain']} ({format_num(float(row['uncertain_rate']))}) |"
            )

    if metric_summary_groups:
        lines.append("\n## 4. グループ別の粗統計\n")
        for group_name, summary in metric_summary_groups.items():
            if not summary:
                continue
            lines.append(f"### {group_name}")
            lines.append(
                f"- 対象: {summary.get('pair_count', 0)} ペア, 変化あり {summary.get('changed_count', 0)} "
                f"({format_num(float(summary.get('changed_rate', 0.0)))})"
            )

    lines.append("\n## 5. 出力ファイル\n")
    lines.append("- `pair_metrics.csv`: ペアごとの粗統計")
    lines.append("- `discovered_items.json`: 差分項目（構造化出力）")
    lines.append("- `pair_judgements.jsonl`: ペアごとの判定結果")
    lines.append("- `item_level_results.csv`: ペア×項目のフラット表")
    lines.append("- `summary_by_item.csv`: 項目別集計")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemini構造化出力でRubric差分を分析する"
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="codes/optimization_results",
        help="initial_rubric.txt / best_rubric.txt があるルート",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="codes/analysis/rubric_diff_judge",
        help="分析結果の出力先ディレクトリ",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Geminiモデル名",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=0,
        help="判定対象ペア数の上限。0なら全件",
    )
    parser.add_argument(
        "--discovery_samples",
        type=int,
        default=10,
        help="差分項目抽出に使うサンプルペア数",
    )
    parser.add_argument(
        "--item_count",
        type=int,
        default=10,
        help="抽出する差分項目数",
    )
    parser.add_argument(
        "--max_chars_discovery",
        type=int,
        default=6000,
        help="差分項目抽出時に各rubricから読む最大文字数",
    )
    parser.add_argument(
        "--max_chars_judge",
        type=int,
        default=16000,
        help="各ペア判定時に各rubricから読む最大文字数",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="対象datasetをカンマ区切り指定（例: asap_1,ets3）",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="対象model名をカンマ区切り指定",
    )
    parser.add_argument(
        "--seed_prompts",
        type=str,
        default="",
        help="対象seed_promptをカンマ区切り指定（expert,simplest等）",
    )
    parser.add_argument(
        "--with_rationale",
        type=str,
        default="",
        help="対象with_rationaleをカンマ区切り指定（True,False）",
    )
    parser.add_argument(
        "--items_json",
        type=str,
        default="",
        help="差分項目JSONを指定すると自動抽出をスキップ",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Gemini呼び出しを行わず、粗統計とテンプレ項目のみ出力",
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数seed")
    parser.add_argument("--retries", type=int, default=3, help="Gemini呼び出しリトライ回数")
    parser.add_argument(
        "--retry_wait_sec",
        type=float,
        default=2.0,
        help="Gemini呼び出し失敗時の待機秒数",
    )
    return parser.parse_args()


def parse_csv_filter(arg: str) -> set[str] | None:
    values = [x.strip() for x in arg.split(",") if x.strip()]
    return set(values) if values else None


def main() -> None:
    load_env()
    args = parse_args()
    random.seed(args.seed)

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_root.exists():
        raise FileNotFoundError(f"results_root not found: {results_root}")

    all_pairs = collect_rubric_pairs(results_root)
    if not all_pairs:
        raise RuntimeError("initial_rubric.txt / best_rubric.txt のペアが見つかりませんでした。")

    pairs = filter_pairs(
        all_pairs,
        datasets=parse_csv_filter(args.datasets),
        models=parse_csv_filter(args.models),
        seed_prompts=parse_csv_filter(args.seed_prompts),
        with_rationale=parse_csv_filter(args.with_rationale),
    )
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("フィルタ後に対象ペアが0件です。")

    metrics = [compute_pair_metrics(p) for p in pairs]
    metrics_map = {m.pair_id: m for m in metrics}
    metric_rows = []
    for p in pairs:
        m = metrics_map[p.pair_id]
        row = {**asdict(p), **asdict(m)}
        metric_rows.append(row)

    write_csv(output_dir / "pair_metrics.csv", metric_rows)
    metric_summary_all = summarize_metrics(metrics)
    metric_summary_groups = metrics_by_group(pairs, metrics_map)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(
            {"overall": metric_summary_all, "by_group": metric_summary_groups},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 差分項目の準備
    if args.items_json:
        items_data = json.loads(Path(args.items_json).read_text(encoding="utf-8"))
        if isinstance(items_data, dict) and "items" in items_data:
            items_data = items_data["items"]
        items = [DifferenceItem.model_validate(x) for x in items_data]
        items = normalize_items(items, args.item_count)
        discovery = DiscoveryOutput(
            summary_ja="items_jsonから読み込み",
            items=items,
        )
    elif args.dry_run:
        items = normalize_items(DEFAULT_ITEMS, args.item_count)
        discovery = DiscoveryOutput(
            summary_ja="dry_runのためテンプレート項目を使用",
            items=items,
        )
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY が未設定です。Gemini構造化出力を使うには環境変数を設定してください。"
            )
        client = genai.Client(api_key=api_key)
        sample_pairs = pick_discovery_samples(
            pairs=pairs,
            metrics_map=metrics_map,
            sample_size=args.discovery_samples,
        )
        discovery = discover_items_with_gemini(
            client=client,
            model_name=args.model,
            sample_pairs=sample_pairs,
            item_count=args.item_count,
            max_chars_per_rubric=args.max_chars_discovery,
            retries=args.retries,
            retry_wait_sec=args.retry_wait_sec,
        )
        items = discovery.items

    (output_dir / "discovered_items.json").write_text(
        json.dumps([i.model_dump() for i in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "discovery_summary.json").write_text(
        discovery.model_dump_json(indent=2),
        encoding="utf-8",
    )

    pair_results: list[dict[str, object]] = []
    item_level_rows: list[dict[str, object]] = []

    if args.dry_run:
        for p in pairs:
            rec = {
                "pair_id": p.pair_id,
                "dataset": p.dataset,
                "model_name": p.model_name,
                "run_name": p.run_name,
                "pair_summary_ja": "dry_runのため判定未実施",
                "item_judgements": [
                    {
                        "item_id": item.item_id,
                        "status": "uncertain",
                        "confidence": 0.0,
                        "evidence_before": "",
                        "evidence_after": "",
                        "note_ja": "dry_run",
                    }
                    for item in items
                ],
            }
            pair_results.append(rec)
    else:
        api_key = os.getenv("GOOGLE_API_KEY")
        assert api_key is not None
        client = genai.Client(api_key=api_key)

        total = len(pairs)
        for idx, p in enumerate(pairs, start=1):
            print(f"[{idx}/{total}] judging: {p.pair_id}")
            raw_judgement = judge_single_pair_with_gemini(
                client=client,
                model_name=args.model,
                pair=p,
                items=items,
                max_chars_per_rubric=args.max_chars_judge,
                retries=args.retries,
                retry_wait_sec=args.retry_wait_sec,
            )
            norm = normalize_pair_judgement(raw_judgement, items)
            rec = {
                "pair_id": p.pair_id,
                "dataset": p.dataset,
                "model_name": p.model_name,
                "run_name": p.run_name,
                "pair_summary_ja": norm.pair_summary_ja,
                "item_judgements": [j.model_dump() for j in norm.item_judgements],
            }
            pair_results.append(rec)

    # item-level table
    for rec in pair_results:
        for j in rec["item_judgements"]:  # type: ignore[index]
            assert isinstance(j, dict)
            row = {
                "pair_id": rec["pair_id"],
                "dataset": rec["dataset"],
                "model_name": rec["model_name"],
                "run_name": rec["run_name"],
                "item_id": j.get("item_id"),
                "status": j.get("status"),
                "confidence": j.get("confidence"),
                "evidence_before": j.get("evidence_before"),
                "evidence_after": j.get("evidence_after"),
                "note_ja": j.get("note_ja"),
            }
            item_level_rows.append(row)

    with (output_dir / "pair_judgements.jsonl").open("w", encoding="utf-8") as f:
        for rec in pair_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    write_csv(output_dir / "item_level_results.csv", item_level_rows)

    item_summary_rows = aggregate_item_results(pair_results=pair_results, items=items)
    write_csv(output_dir / "summary_by_item.csv", item_summary_rows)

    summary_md = build_summary_markdown(
        metric_summary_all=metric_summary_all,
        metric_summary_groups=metric_summary_groups,
        item_summary_rows=item_summary_rows,
        discovery=discovery,
    )
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
