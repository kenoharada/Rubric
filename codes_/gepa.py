# minimal_gepa_optimize.py
# -*- coding: utf-8 -*-

# このファイルは、GEPAの「optimize」機能を最小限で単一ファイル化した実装です。
# - LLM呼び出しは、ユーザ提供の get_llm_response をそのまま利用します（OpenRouter, Chat Completions）。
# - 元実装（src/gepa/api.py, src/gepa/adapters/default_adapter/default_adapter.py 他）の重要処理のみを抽出・簡略化しています。
# - どの処理をどう簡略化したかは、各所に「簡略化ポイント」としてコメントしています。

from dataclasses import dataclass
from typing import Any, TypedDict
import random
import requests
import json
import os
import dotenv

dotenv.load_dotenv()

def get_llm_response(messages, model_name, config):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY"),
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model_name,
            "messages": messages,
            **(config or {}),
        }),
        timeout=60,
    )
    response.raise_for_status()
    from types import SimpleNamespace
    data = json.loads(response.text, object_hook=lambda d: SimpleNamespace(**d))
    return data.choices[0].message.content


# 非同期版
try:
    import aiohttp
    import asyncio
except ImportError:
    aiohttp = None  # 利用側で ImportError に備える

async def get_llm_respnose_async(messages, model_name, config, session: "aiohttp.ClientSession | None" = None, timeout: int = 60):
    """
    非同期で OpenRouter Chat Completions を呼び出す。
    引数:
      messages: [{"role": "...", "content": "..."}]
      model_name: モデルID
      config: 追加ペイロード(dict)
      session: 使い回したい aiohttp.ClientSession (任意)
      timeout: 秒
    戻り値: アシスタントの text コンテンツ (str)
    """
    if aiohttp is None:
        raise RuntimeError("aiohttp がインストールされていません。pip install aiohttp を行ってください。")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer " + os.getenv("OPENROUTER_API_KEY", ""),
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        **(config or {}),
    }

    owns_session = session is None
    if owns_session:
        session = aiohttp.ClientSession()
    try:
        async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"OpenRouter error {resp.status}: {text}")
            data = await resp.json()
        # choices[0].message.content を安全に取り出す
        return (
            data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
        )
    finally:
        if owns_session:
            await session.close()
# ========== 型/コンテナ（元: src/gepa/core/adapter.py の最小移植） ==========
@dataclass
class EvaluationBatch:
    outputs: list[dict]
    scores: list[float]
    trajectories: list[dict] | None = None

@dataclass
class GEPAResult:
    best_candidate: dict[str, str]
    best_val_score: float
    history: list[dict]  # 各イテレーションのログ（簡易）
# ================================================================


# ========== データ型（元: DefaultAdapterの入出力型を最小移植） ==========
class DefaultDataInst(TypedDict):
    input: str                   # ユーザ入力
    additional_context: dict[str, str]  # 任意の補助情報
    answer: str                  # 正解（文字列一致判定に使用）

class DefaultTrajectory(TypedDict):
    data: DefaultDataInst
    full_assistant_response: str

class DefaultRolloutOutput(TypedDict):
    full_assistant_response: str
# ================================================================


# ========== 最小アダプタ（元: src/gepa/adapters/default_adapter/default_adapter.py を簡略化） ==========
class MinimalDefaultAdapter:
    """
    簡略化ポイント:
    - litellmや並列/バッチ呼び出しを全て削除し、get_llm_responseを1件ずつ呼ぶだけに簡略化。
    - スコアは「正解文字列が応答に含まれるか」の単純な一致判定のみ（元実装の要点）。
    - capture_traces=True時だけTrajを詰める最小限の挙動を維持。
    """

    def __init__(self, task_model_name: str, task_model_config: dict | None = None, failure_score: float = 0.0):
        self.task_model_name = task_model_name
        self.task_model_config = task_model_config or {}
        self.failure_score = failure_score

    def evaluate(
        self,
        batch: list[DefaultDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        outputs: list[DefaultRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[DefaultTrajectory] | None = [] if capture_traces else None

        # 1コンポーネント前提: 最初の要素をsystem promptとみなす（元実装と同様の前提）
        # 簡略化ポイント: 複数コンポーネント対応/自由なキー名対応は省略
        system_content = next(iter(candidate.values()))

        for data in batch:
            user_content = data["input"]
            # OpenRouterのChat形式で呼ぶ（ユーザ提供の実装を活用）
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            try:
                assistant_response = get_llm_response(messages, self.task_model_name, self.task_model_config).strip()
            except Exception as e:
                assistant_response = f"Error calling task model: {e!s}"

            output = {"full_assistant_response": assistant_response}
            # TODO: より高度なスコアリングロジックを入れる余地あり
            score = 1.0 if data["answer"] in assistant_response else self.failure_score

            outputs.append(output)
            scores.append(score)

            if capture_traces:
                trajectories.append(
                    {
                        "data": data,
                        "full_assistant_response": assistant_response,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        元のDefaultAdapterのロジックを簡略移植:
        - Inputs / Generated Outputs / Feedback を作る。
        - 正解時と不正解時でFeedbackの文言を切替。
        - 1コンポーネント（先頭キー）前提。
        """
        ret_d: dict[str, list[dict[str, Any]]] = {}

        # 簡略化: 1コンポーネントのみ対応（複数の場合は先頭のみ）
        comp = components_to_update[0]

        items: list[dict[str, Any]] = []
        trace_instances = list(zip(eval_batch.trajectories or [], eval_batch.scores, eval_batch.outputs, strict=False))

        for traj, score, _ in trace_instances:
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]

            if score > 0.0:
                feedback = f"生成は正解でした。最終的な回答には必ず '{data['answer']}' が含まれていました。"
            else:
                additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
                feedback = (
                    f"生成は不正解でした。正しい答えは '{data['answer']}' です。"
                    "回答文中にこの値を確実に含めてください。"
                )
                if additional_context_str:
                    feedback += f"\n追加文脈:\n{additional_context_str}"

            items.append(
                {
                    "Inputs": data["input"],
                    "Generated Outputs": generated_outputs,
                    "Feedback": feedback,
                }
            )

        ret_d[comp] = items
        return ret_d
# ================================================================


# ========== リフレクション: 新しいプロンプトの提案（簡略版） ==========
def propose_new_text_via_reflection(
    reflection_model_name: str,
    reflection_model_config: dict | None,
    base_instruction: str,
    dataset_with_feedback: list[dict[str, Any]],
) -> str:
    """
    簡略化ポイント:
    - 元実装のReflectiveMutationProposerやDSPy Signatureは使わず、素朴な1プロンプトで
      「現在の指示」と「フィードバックの要約」を渡して改良案を1テキストで返す。
    - 返却テキストのパースも単純化（``` で囲まれていれば中身、なければ全体）。
    """

    # 反省用データをシンプルなテキストにまとめる
    # ここでは小さめの要約で十分（LLMトークン節約）
    lines = []
    max_items = min(6, len(dataset_with_feedback))  # 過剰に長くならないよう上限
    for i in range(max_items):
        item = dataset_with_feedback[i]
        inp = str(item.get("Inputs", ""))[:400]
        gen = str(item.get("Generated Outputs", ""))[:400]
        fb = str(item.get("Feedback", ""))[:600]
        lines.append(f"- Case {i+1}\n  Inputs: {inp}\n  ModelOutput: {gen}\n  Feedback: {fb}")
    feedback_block = "\n".join(lines)

    system_msg = (
        "あなたはプロンプトエンジニアです。与えられたタスク用のシステムプロンプトを、"
        "与えられたフィードバックを踏まえて1つの改良版テキストに書き換えてください。"
        "出力は改良されたプロンプト本文のみ（追加説明なし）。"
    )
    user_msg = f"""現在のプロンプト:
<<<
{base_instruction}
>>>

フィードバック要約:
{feedback_block}

要件:
- 出力は改良版プロンプト本文のみ（余計な枠や説明は不要）。
- 箇条書き等は任意。ただし最終回答（モデル応答）に正解の数値/文字列を必ず含ませるような指針を入れること。
"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    raw = get_llm_response(messages, reflection_model_name, reflection_model_config or {})
    text = raw.strip()

    # ``` で囲まれていれば中身抽出（雑にだが頑健）
    if text.count("```") >= 2:
        start = text.find("```")
        end = text.rfind("```")
        if 0 <= start < end:
            inner = text[start + 3 : end].strip()
            # 言語指定がある場合 ```text などを剥がす
            if "\n" in inner:
                first_nl = inner.find("\n")
                maybe_lang = inner[:first_nl].strip().lower()
                if len(maybe_lang) < 12 and all(c.isalpha() or c in "-_" for c in maybe_lang):
                    inner = inner[first_nl + 1 :].strip()
            return inner

    return text
# ================================================================


# ========== 最小エンジン optimize（元: src/gepa/api.py を簡略化） ==========
def optimize(
    seed_candidate: dict[str, str],
    trainset: list[DefaultDataInst],
    valset: list[DefaultDataInst] | None = None,
    task_model_name: str | None = None,
    reflection_model_name: str | None = None,

    # 主要ハイパラ（必要最小限のみ）
    reflection_minibatch_size: int = 3,
    max_metric_calls: int = 100,
    perfect_score: float = 1.0,
    skip_perfect_score: bool = True,

    # 乱数/再現
    seed: int = 0,

    # OpenRouterの補助設定
    task_model_config: dict | None = None,
    reflection_model_config: dict | None = None,

    # ログ最小表示
    verbose: bool = True,
) -> GEPAResult:
    """
    簡略化ポイント:
    - 複数戦略（Pareto, Merge, ComponentSelector, ExperimentTracker, 進捗バー, W&B/MLflow等）は全削除。
    - 反復: 「現候補をミニバッチ評価 → 反省データ生成 → 1回だけ改良案提案 → 同じミニバッチで比較 → 改善なら採用」
    - ベスト更新時のみ全valset評価（平均スコア）を取り直し、結果のbest_candidate/best_val_scoreを更新。
    - componentは1個前提（最初のキー）。複数対応は省略。
    """

    assert task_model_name is not None, "task_model_name を指定してください（最小実装では必須）。"
    assert reflection_model_name is not None, "reflection_model_name を指定してください（最小実装では必須）。"

    if valset is None:
        valset = trainset

    rng = random.Random(seed)
    adapter = MinimalDefaultAdapter(task_model_name, task_model_config)

    # 1コンポーネント前提: 最初のキー名を取得
    comp_name = next(iter(seed_candidate.keys()))

    # 初期化
    curr_candidate = dict(seed_candidate)
    best_candidate = dict(seed_candidate)
    metric_calls = 0  # 予算カウント（評価と反省両方のLLM呼び出しを含める簡略版）
    history: list[dict] = []

    # 初期のvalスコア評価
    init_eval = adapter.evaluate(valset, curr_candidate, capture_traces=False)
    metric_calls += len(valset)  # valset件数分呼び出し
    best_val = sum(init_eval.scores) / max(1, len(init_eval.scores))

    if verbose:
        print(f"[Init] val_avg={best_val:.3f}, budget_used={metric_calls}/{max_metric_calls}")

    # 予算内で反復
    it = 0
    while metric_calls < max_metric_calls:
        it += 1

        # 1) 反省用ミニバッチのサンプリング
        minibatch = rng.sample(trainset, k=min(reflection_minibatch_size, len(trainset)))

        # 2) 現候補をミニバッチ評価（トレース付き）
        eval_curr = adapter.evaluate(minibatch, curr_candidate, capture_traces=True)
        metric_calls += len(minibatch)
        sum_curr = sum(eval_curr.scores)
        perfect_sum = perfect_score * len(minibatch)

        if verbose:
            print(f"[Iter {it}] curr_sum={sum_curr:.3f}/{perfect_sum:.3f}, budget={metric_calls}/{max_metric_calls}")

        # 3) 全問正解なら、（簡略版では）改良提案をスキップして次へ
        if skip_perfect_score and sum_curr >= perfect_sum:
            history.append({"iter": it, "accepted": False, "reason": "perfect_on_minibatch"})
            if verbose:
                print(f"[Iter {it}] Skip reflection (perfect minibatch).")
            continue

        # 4) 反省データ作成（1コンポーネントのみ）
        reflective_dataset = adapter.make_reflective_dataset(
            curr_candidate,
            eval_curr,
            components_to_update=[comp_name],
        )
        dataset_for_comp = reflective_dataset[comp_name]

        # 5) 改良案のプロンプト生成（Reflection LM呼び出し）
        new_text = propose_new_text_via_reflection(
            reflection_model_name, reflection_model_config, curr_candidate[comp_name], dataset_for_comp
        )
        metric_calls += 1  # 反省LM 1コール

        # 6) 新候補の評価（同じミニバッチ）
        new_candidate = dict(curr_candidate)
        new_candidate[comp_name] = new_text
        eval_new = adapter.evaluate(minibatch, new_candidate, capture_traces=False)
        metric_calls += len(minibatch)
        sum_new = sum(eval_new.scores)

        improved = sum_new >= sum_curr  # 簡略: 改善(同等以上)なら採用
        if verbose:
            print(f"[Iter {it}] new_sum={sum_new:.3f}/{perfect_sum:.3f} -> {'ACCEPT' if improved else 'REJECT'}")

        if improved:
            curr_candidate = new_candidate

            # ベスト更新確認のため valset で再評価
            eval_val = adapter.evaluate(valset, curr_candidate, capture_traces=False)
            metric_calls += len(valset)
            val_avg = sum(eval_val.scores) / max(1, len(eval_val.scores))
            if val_avg > best_val:
                best_val = val_avg
                best_candidate = dict(curr_candidate)
                if verbose:
                    print(f"[Iter {it}] Best updated! val_avg={best_val:.3f} (budget={metric_calls}/{max_metric_calls})")

            history.append({"iter": it, "accepted": True, "minibatch_curr_sum": float(sum_curr), "minibatch_new_sum": float(sum_new), "val_avg": float(val_avg)})
        else:
            history.append({"iter": it, "accepted": False, "minibatch_curr_sum": float(sum_curr), "minibatch_new_sum": float(sum_new)})

        if metric_calls >= max_metric_calls:
            if verbose:
                print(f"[Stop] Budget exhausted: {metric_calls}/{max_metric_calls}")
            break

    return GEPAResult(best_candidate=best_candidate, best_val_score=best_val, history=history)


# ========== 簡単な使い方（実行例）。API課金/環境依存のためコメントアウト ==========
if __name__ == "__main__":
    # 例: 単純な加算問題1件のみ（実運用では複数サンプルを用意）
    train = [
        {"input": "What is 2 + 2?", "additional_context": {}, "answer": "よん"},
        {"input": "Compute 10-3.", "additional_context": {}, "answer": "なな"},
        {"input": "3*3=?", "additional_context": {}, "answer": "きゅう"},
    ]
    val = train

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. Answer briefly and include the exact final number."
    }

    # 環境変数 OPENROUTER_API_KEY を設定してから実行してください。
    res = optimize(
        seed_candidate=seed_prompt,
        trainset=train,
        valset=val,
        task_model_name="openai/gpt-4o-mini",
        reflection_model_name="openai/gpt-4.1-mini",
        reflection_minibatch_size=2,
        max_metric_calls=3,
        seed=0,
        verbose=True,
    )

    print("Best Prompt:", res.best_candidate["system_prompt"])
    print("Best Val Score:", res.best_val_score)