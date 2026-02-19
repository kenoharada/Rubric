from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Protocol, Dict, Any
import random
import re
import math


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class LabeledExample:
    """
    One human-labeled example in the golden set D*.

    For summarization / hallucination:
        source = article
        target = summary

    For data-to-text:
        source = structured data expression
        target = generated sentence
    """
    source: str
    target: str
    human_score: float


# -----------------------------
# LLM interface (plug your provider here)
# -----------------------------
class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Return n generations (strings).
        """
        ...


class DummyLLM:
    """
    A dummy LLM for testing code paths without real API calls.
    - Drafting: returns a random-ish criteria text.
    - Evaluation: returns a random score in range.
    - Refinement: returns 'refined' criteria.
    """
    def __init__(self, score_min: int, score_max: int, seed: int = 0) -> None:
        self.score_min = score_min
        self.score_max = score_max
        random.seed(seed)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        stop: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        outs: List[str] = []
        for _ in range(n):
            if "## Induced Criteria" in prompt:
                outs.append(
                    "Criteria for [Aspect]:\n"
                    "- Focus on correctness and completeness.\n"
                    "- Penalize hallucinated or unsupported content.\n"
                    "- Reward clarity and conciseness.\n"
                )
            elif "Please refine and improve a scoring criteria" in prompt:
                outs.append(
                    "Criteria for [Aspect]:\n"
                    "- Improved: emphasize alignment with human judgments.\n"
                    "- Add detail for borderline cases.\n"
                )
            else:
                # scoring prompt
                score = random.randint(self.score_min, self.score_max)
                outs.append(f"Score: {score}\nReason: (dummy)")
        return outs


# -----------------------------
# Correlation (meta-correlation metric f)
# -----------------------------
def _rankdata_average_ties(values: Sequence[float]) -> List[float]:
    """
    Rank data with average ranks for ties (1..n).
    """
    n = len(values)
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def pearson_corr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return num / (denx * deny)


def spearman_corr(x: Sequence[float], y: Sequence[float]) -> float:
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    return pearson_corr(rx, ry)


# -----------------------------
# Prompt templates (Figure 5 / Figure 11)
# -----------------------------
DRAFTING_TEMPLATE_SUMMARIZATION = """## Instructions
Please infer the scoring criteria for the following task:
[Score the following summary of a news article on its [Aspect]. Please return your score on how the summary
is consistent with the article in the scale of 1 to 5, with 1 being the lowest.]
- The following is some examples on evaluation scores of [Aspect] of summary (in the range of 1 to 5, where 1
being the lowest).
- Please carefully read all the article, summary and their assigned score, and induce the most possible scoring
rule and criteria used.
- It is optimal that, by using the induced criteria, you are very likely to assign a same score on [Aspect] to the
provided reference scores.
## Criteria for [Aspect]
- The scoring criteria been used. Now it is not explicitly provided, and you should induce it from the following
samples.
- The induced criteria should be able to explain the scores of all the samples provided, being generic and
concise.
## Examples
[In-Context Few-Shot Samples]
## Induced Criteria
Criteria for [Aspect]:
"""

REFINEMENT_TEMPLATE = """Please refine and improve a scoring criteria used by a large language model in evaluating the [Aspect] of [Task].
Large language models (LLMs) are powerful neural models that can evaluate the quality of [Task]. However,
LLMs may not always agree with human judgments. Please refine the scoring criteria used by LLMs to
improve its correlation with human expert scores.
To refine the scoring criteria used by the LLM in evaluating the [Aspect] of [Task], please follow the following
instructions step-by-step:
1. Carefully read each example, understand each [Source acronym (e.g. article)] and its corresponding
[Target acronym (e.g. summary)], and get your initial assessment of its quality on [Aspect].
2. Compare the test score obtained by the LLM according to the criteria and the ground-truth score from
human experts. Please think why the correlation is limited by using the current criteria, and how can you
improve the criteria to increase the correlation between LLM’s score and human expert score. If there is a
small gap or no gap, this means the criteria work well in this case.
3. Read all of the test cases and rethink how you could refine the current criteria based on your observations
and analysis. Then, refine the criteria to make it concise, accurate, and consistent with human judgments.
When refining the criteria, you can do the following: 1) modification: adjust some parts of the criteria to
increase its correlation with the scoring criteria that you think might used by human experts; 2) paraphrase: if
the criteria is good enough, you can consider paraphrasing it to make more concise and easy to understand;
3) adding aspects or details: if you fine some new underlying scoring rules not covered by the current criteria,
consider adding them as a new line of injecting to current criteria, but make sure not to make the criteria
too long and redundant; 4) calibrate: you can take other methods you think being helpful to improve the
correlation with human experts.
Please return only your refined criteria without any additional sentences.
Old criteria: [Previous Criteria Drafts]
Examples: [In-Context Few-Shot Samples]
"""


def _format_fewshot_examples(
    examples: Sequence[LabeledExample],
    *,
    source_name: str,
    target_name: str,
    score_name: str = "Expert Score",
) -> str:
    blocks: List[str] = []
    for i, ex in enumerate(examples, 1):
        blocks.append(
            f"Example {i}:\n"
            f"{source_name}:\n{ex.source}\n\n"
            f"{target_name}:\n{ex.target}\n\n"
            f"{score_name}: {ex.human_score}\n"
        )
    return "\n".join(blocks).strip()


def build_drafting_prompt_summarization(
    *,
    aspect: str,
    fewshot_examples: Sequence[LabeledExample],
    source_name: str = "Article",
    target_name: str = "Summary",
) -> str:
    prompt = DRAFTING_TEMPLATE_SUMMARIZATION.replace("[Aspect]", aspect)
    fewshot = _format_fewshot_examples(
        fewshot_examples,
        source_name=source_name,
        target_name=target_name,
        score_name="Expert Score",
    )
    return prompt.replace("[In-Context Few-Shot Samples]", fewshot)


def build_refinement_prompt(
    *,
    aspect: str,
    task: str,
    old_criteria: str,
    misaligned_examples: Sequence[Tuple[LabeledExample, float]],
    source_name: str,
    target_name: str,
) -> str:
    """
    misaligned_examples: list of (example, llm_pred_score)
    """
    prompt = REFINEMENT_TEMPLATE.replace("[Aspect]", aspect).replace("[Task]", task)

    # Include both human score and LLM test score in the example block to enable "Compare..." step.
    ex_blocks: List[str] = []
    for i, (ex, pred) in enumerate(misaligned_examples, 1):
        ex_blocks.append(
            f"Case {i}:\n"
            f"{source_name}:\n{ex.source}\n\n"
            f"{target_name}:\n{ex.target}\n\n"
            f"Human expert score: {ex.human_score}\n"
            f"LLM test score (by current criteria): {pred}\n"
        )
    examples_text = "\n".join(ex_blocks).strip()

    prompt = prompt.replace("[Previous Criteria Drafts]", old_criteria)
    prompt = prompt.replace("[In-Context Few-Shot Samples]", examples_text)
    return prompt


# -----------------------------
# Scoring prompt (needed to compute f(c, D*))
# (Paper shows evaluation templates in Figure 8/9/10, but Algorithm 1 needs some scoring call.)
# -----------------------------
def build_scoring_prompt_summarization(
    *,
    aspect: str,
    source: str,
    target: str,
    criteria: str,
    score_min: int,
    score_max: int,
    source_name: str = "Article",
    target_name: str = "Summary",
) -> str:
    return (
        f"## Instructions\n"
        f"Score the following summary of a news article on its {aspect}.\n"
        f"Return your score on a scale of {score_min} to {score_max} (integer).\n\n"
        f"## Example\n"
        f"{source_name}:\n{source}\n\n"
        f"{target_name}:\n{target}\n\n"
        f"## Criteria for {aspect}\n"
        f"{criteria}\n\n"
        f"## Evaluation\n"
        f"First output only the score as: Score: <number>\n"
        f"Then optionally provide reasoning.\n"
    )


_SCORE_RE = re.compile(r"(?:^|\n)\s*(?:Score\s*[:：]\s*)?(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_score(text: str, *, score_min: int, score_max: int) -> Optional[float]:
    m = _SCORE_RE.search(text.strip())
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    # clamp
    if val < score_min:
        val = float(score_min)
    if val > score_max:
        val = float(score_max)
    return val


# -----------------------------
# AutoCalibrate (Algorithm 1)
# -----------------------------
class AutoCalibrate:
    def __init__(
        self,
        *,
        llm: LLMClient,
        aspect: str,
        task_name: str,
        score_min: int,
        score_max: int,
        correlation_fn: Callable[[Sequence[float], Sequence[float]], float] = spearman_corr,
        rng: Optional[random.Random] = None,
        source_name: str = "Article",
        target_name: str = "Summary",
    ) -> None:
        self.llm = llm
        self.aspect = aspect
        self.task_name = task_name
        self.score_min = score_min
        self.score_max = score_max
        self.correlation_fn = correlation_fn
        self.rng = rng or random.Random(0)
        self.source_name = source_name
        self.target_name = target_name

    # ---------
    # Drafting stage (Algorithm 1: lines 1-7)
    # ---------
    def drafting(
        self,
        D_star: Sequence[LabeledExample],
        *,
        fewshot_sizes: Sequence[int],
        monte_carlo_trials: int,
        temperature: float,
        temperature_sampling_count: int,
        max_tokens: int = 768,
    ) -> List[str]:
        """
        Draft candidate criteria via LLM with Monte-Carlo sampling over few-shot exemplars
        and temperature sampling (diverse criteria). (Algorithm 1: 1-7)
        """
        candidates: List[str] = []

        for li in fewshot_sizes:
            for _ in range(monte_carlo_trials):
                fewshot = self.rng.sample(list(D_star), k=min(li, len(D_star)))
                prompt = build_drafting_prompt_summarization(
                    aspect=self.aspect,
                    fewshot_examples=fewshot,
                    source_name=self.source_name,
                    target_name=self.target_name,
                )
                outs = self.llm.generate(
                    prompt,
                    temperature=temperature,
                    n=temperature_sampling_count,
                    max_tokens=max_tokens,
                )
                for o in outs:
                    crit = o.strip()
                    if crit:
                        candidates.append(crit)

        return self._dedup_keep_order(candidates)

    # ---------
    # Revisiting stage (Algorithm 1: lines 8-15 + return)
    # ---------
    def revisiting(
        self,
        D_star: Sequence[LabeledExample],
        draft_candidates: Sequence[str],
        *,
        top_k: int,
        misaligned_pool_size: int,
        refine_fewshot_sizes: Sequence[int],
        refine_monte_carlo_trials: int,
        refine_temperature: float,
        refine_temperature_sampling_count: int,
        draft_eval_max_tokens: int = 64,
        refine_max_tokens: int = 768,
    ) -> Tuple[str, Dict[str, float]]:
        """
        - Evaluate each criteria on D* to compute f(c, D*)
        - Keep Top-K
        - For each Top-K criteria, collect misaligned examples
        - Refine criteria with LLM using Figure 11 prompt
        - Re-evaluate and return best criteria
        """
        # 8: Revisit C and retain top-K candidates
        scored = self._score_criteria_set(D_star, list(draft_candidates), max_tokens=draft_eval_max_tokens)
        top = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
        top_criteria = [c for c, _ in top]

        # 9: Collect mis-aligned examples DR_i for each ci in C
        misaligned_map: Dict[str, List[Tuple[LabeledExample, float, float]]] = {}
        for c in top_criteria:
            preds = self._predict_scores(D_star, c, max_tokens=draft_eval_max_tokens)
            triples: List[Tuple[LabeledExample, float, float]] = []
            for ex, pred in zip(D_star, preds):
                if pred is None:
                    continue
                err = abs(float(pred) - float(ex.human_score))
                triples.append((ex, float(pred), err))
            triples.sort(key=lambda t: t[2], reverse=True)
            misaligned_map[c] = triples[:misaligned_pool_size]

        # 10-15: refine each top criteria, add refined criteria into candidate set
        refined_candidates: List[str] = list(draft_candidates)

        for c in top_criteria:
            pool = misaligned_map.get(c, [])
            if not pool:
                continue

            for li in refine_fewshot_sizes:
                for _ in range(refine_monte_carlo_trials):
                    sampled = self.rng.sample(pool, k=min(li, len(pool)))
                    # keep only (example, pred_score) for prompt
                    prompt = build_refinement_prompt(
                        aspect=self.aspect,
                        task=self.task_name,
                        old_criteria=c,
                        misaligned_examples=[(ex, pred) for (ex, pred, _err) in sampled],
                        source_name=self.source_name,
                        target_name=self.target_name,
                    )
                    outs = self.llm.generate(
                        prompt,
                        temperature=refine_temperature,
                        n=refine_temperature_sampling_count,
                        max_tokens=refine_max_tokens,
                    )
                    for o in outs:
                        crit = o.strip()
                        if crit:
                            refined_candidates.append(crit)

        refined_candidates = self._dedup_keep_order(refined_candidates)

        # Return: best criteria over combined pool
        final_scored = self._score_criteria_set(D_star, refined_candidates, max_tokens=draft_eval_max_tokens)
        best = max(final_scored.items(), key=lambda kv: kv[1])[0]
        return best, final_scored

    # ---------
    # Convenience: full run
    # ---------
    def run(
        self,
        D_star: Sequence[LabeledExample],
        *,
        fewshot_sizes: Sequence[int],
        monte_carlo_trials: int,
        drafting_temperature: float,
        drafting_temperature_sampling_count: int,
        top_k: int,
        misaligned_pool_size: int,
        refine_fewshot_sizes: Sequence[int],
        refine_monte_carlo_trials: int,
        refine_temperature: float,
        refine_temperature_sampling_count: int,
    ) -> Tuple[str, Dict[str, float]]:
        drafts = self.drafting(
            D_star,
            fewshot_sizes=fewshot_sizes,
            monte_carlo_trials=monte_carlo_trials,
            temperature=drafting_temperature,
            temperature_sampling_count=drafting_temperature_sampling_count,
        )
        return self.revisiting(
            D_star,
            drafts,
            top_k=top_k,
            misaligned_pool_size=misaligned_pool_size,
            refine_fewshot_sizes=refine_fewshot_sizes,
            refine_monte_carlo_trials=refine_monte_carlo_trials,
            refine_temperature=refine_temperature,
            refine_temperature_sampling_count=refine_temperature_sampling_count,
        )

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _predict_scores(self, D_star: Sequence[LabeledExample], criteria: str, *, max_tokens: int) -> List[Optional[float]]:
        preds: List[Optional[float]] = []
        for ex in D_star:
            prompt = build_scoring_prompt_summarization(
                aspect=self.aspect,
                source=ex.source,
                target=ex.target,
                criteria=criteria,
                score_min=self.score_min,
                score_max=self.score_max,
                source_name=self.source_name,
                target_name=self.target_name,
            )
            out = self.llm.generate(prompt, temperature=0.0, n=1, max_tokens=max_tokens)[0]
            preds.append(parse_score(out, score_min=self.score_min, score_max=self.score_max))
        return preds

    def _score_criteria_set(
        self,
        D_star: Sequence[LabeledExample],
        criteria_set: Sequence[str],
        *,
        max_tokens: int,
    ) -> Dict[str, float]:
        human = [float(ex.human_score) for ex in D_star]
        results: Dict[str, float] = {}
        for c in criteria_set:
            preds = self._predict_scores(D_star, c, max_tokens=max_tokens)
            # drop missing
            paired = [(p, h) for p, h in zip(preds, human) if p is not None]
            if len(paired) < 2:
                results[c] = -1.0
                continue
            pvals, hvals = zip(*paired)
            results[c] = float(self.correlation_fn(list(pvals), list(hvals)))
        return results

    @staticmethod
    def _dedup_keep_order(items: Sequence[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for it in items:
            key = it.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out


# -----------------------------
# Example usage (with DummyLLM)
# -----------------------------
if __name__ == "__main__":
    # D* (golden set) is user-provided (human expert labels).
    D_star = [
        LabeledExample(source="Article text A ...", target="Summary A ...", human_score=5),
        LabeledExample(source="Article text B ...", target="Summary B ...", human_score=2),
        LabeledExample(source="Article text C ...", target="Summary C ...", human_score=4),
        LabeledExample(source="Article text D ...", target="Summary D ...", human_score=1),
    ]

    llm = DummyLLM(score_min=1, score_max=5, seed=42)

    ac = AutoCalibrate(
        llm=llm,
        aspect="Consistency",
        task_name="summarization",
        score_min=1,
        score_max=5,
        correlation_fn=spearman_corr,
        rng=random.Random(123),
        source_name="Article",
        target_name="Summary",
    )

    best_criteria, all_scores = ac.run(
        D_star,
        fewshot_sizes=[4, 6, 8],                 # L in Algorithm 1
        monte_carlo_trials=2,                    # N in Algorithm 1
        drafting_temperature=1.0,
        drafting_temperature_sampling_count=3,   # temperature sampling count
        top_k=3,                                 # K in Algorithm 1
        misaligned_pool_size=5,
        refine_fewshot_sizes=[1, 2, 4],
        refine_monte_carlo_trials=2,
        refine_temperature=1.0,
        refine_temperature_sampling_count=2,
    )

    print("BEST CRITERIA:\n", best_criteria)
    print("\nTop-5 criteria by correlation:")
    for c, s in sorted(all_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        print(f"{s:.4f}  |  {c.splitlines()[0][:80]}...")
