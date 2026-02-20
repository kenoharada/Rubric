"""
Rubric パターン定義

最適化前後のルーブリックを正規表現でマッチさせ、特徴の変化を定量化する。
無効化したい項目は enabled=False を設定してください。
"""

# ── type_id 定義 ─────────────────────────────────────────
# 各パターンが属するカテゴリ。LaTeX 表の行グループとしても使用される。
# TYPE_ORDER = ["rule_structure", "evidence_handling", "writing_quality"]
TYPE_ORDER = ["rule_structure", "evidence_handling"]  # writing_quality は一旦除外

TYPE_INFO = {
    "rule_structure": {
        "label_en": "Rule Structure",
        "label_ja": "ルール構造",
        "description_en": (
            "Structural rules that govern the scoring process: conditional "
            "branching, numeric thresholds, stepwise workflows, score caps, "
            "and boundary-resolution procedures."
        ),
        "description_ja": (
            "採点プロセスを制御する構造的ルール（条件分岐、数的閾値、"
            "手順化、スコア上限、境界解決手順など）。"
        ),
    },
    "evidence_handling": {
        "label_en": "Evidence Handling",
        "label_ja": "エビデンス評価",
        "description_en": (
            "Rules governing how evidence quality and quantity are assessed: "
            "safeguards against count inflation, requirements for concrete "
            "exemplification, elaboration depth taxonomies, and handling of "
            "off-topic or nuanced content. "
        ),
        "description_ja": (
            "エビデンスの数え方・質的評価・具体性に関するルール（水増し防止、"
            "具体例要求、精緻化段階分類、脱線制御など）。"
        ),
    },
    # "writing_quality": {
    #     "label_en": "Writing Quality",
    #     "label_ja": "文章品質",
    #     "description_en": (
    #         "Criteria evaluating surface-level language quality and discourse "
    #         "structure: organization, coherence, grammar, mechanics, and "
    #         "related features. Present in both initial and optimized rubrics, "
    #         "but optimization may attach concrete severity thresholds."
    #     ),
    #     "description_ja": (
    #         "文章構成・表面的言語品質に関する評価基準（構成、一貫性、文法、"
    #         "メカニクスなど）。初期ルーブリックにも存在するが、最適化で"
    #         "具体的なしきい値が付与されることがある。"
    #     ),
    # },
}

PATTERNS = [
    # ── Rule Structure ──────────────────────────────────────
    # 最適化後に増加しやすい: 採点の一貫性を高めるための構造的ルール群
    {
        "enabled": True,
        "pattern_id": "if_rules",
        "type_id": "rule_structure",
        "name_ja": "条件付きルール",
        "description_ja": (
            "if / when / unless / provided that 等の条件節を用いて、"
            "特定の状況における採点判断を明示的に分岐させるルール。"
            "最適化ルーブリックでは、曖昧な判断を排除するために"
            "条件分岐が大幅に増加する傾向がある。"
        ),
        "name_en": "Conditional Gating",
        "description_en": (
            "Condition-based branching rules (if / when / unless / provided that) "
            "that explicitly guide rater decisions under specific circumstances. "
            "Refined rubrics tend to add many conditional gates to reduce "
            "ambiguity in borderline situations."
        ),
        "cues_en": "if, when, unless, provided that",
        "regex": r"\bif\b|\bwhen\b|\bunless\b|\bprovided that\b",
    },
    {
        "enabled": True,
        "pattern_id": "tie_breaker_boundary",
        "type_id": "rule_structure",
        "name_ja": "境界・タイブレーク",
        "description_ja": (
            "隣接スコア帯の境界ケースの解決手順を定めるルール。"
            "例: '3 vs 4 の判断基準'、'borderline のとき lower を選ぶ' 等。"
            "最適化ルーブリックでは、評価者間一致度を上げるために"
            "境界判断の具体的指示が追加される。"
        ),
        "name_en": "Boundary / Tie-Break",
        "description_en": (
            "Rules for resolving borderline cases between adjacent score bands. "
            "Includes tie-break procedures, explicit threshold cutoffs, and "
            "'N vs N' comparisons (e.g., '3 vs 4'). Refined rubrics often "
            "add detailed boundary-resolution instructions to improve "
            "inter-rater agreement."
        ),
        "cues_en": "tie-break, borderline, threshold, N vs N, between adjacent",
        "regex": r"tie-?break|borderline|\bthreshold\b|\d\s*vs\.?\s*\d|between\s+adjacent",
    },
    {
        "enabled": True,
        "pattern_id": "stepwise_process",
        "type_id": "rule_structure",
        "name_ja": "ステップ式手順",
        "description_ja": (
            "Step 1 / Step 2… やチェックリスト形式で採点手順を順序化するルール。"
            "最適化により、自由裁量的な採点から再現性の高い手順化された"
            "ワークフローへ移行する傾向が見られる。"
        ),
        "name_en": "Stepwise Workflow",
        "description_en": (
            "Ordered step-by-step procedures (Step 1, Step 2...) or checklists "
            "that structure the scoring process into a reproducible workflow. "
            "Optimization tends to transform free-form scoring guidance into "
            "structured, sequential procedures for raters to follow."
        ),
        "cues_en": "step N, checklist, workflow, procedure, in order",
        "regex": r"\bstep\s+\d|checklist|workflow|procedure|\bin order\b",
    },
    {
        "enabled": True,
        "pattern_id": "quantitative_thresholds",
        "type_id": "rule_structure",
        "name_ja": "数量しきい値",
        "description_ja": (
            "数値による閾値や定量基準。'at least 2 facts'、'~30%' 等の"
            "具体的数値を用いてスコア帯を区切るルール。"
            "最適化前は 'some' や 'several' 等の曖昧な定性的記述のみだった"
            "基準が、最適化後に数値化されるケースが多い。"
        ),
        "name_en": "Quantitative Threshold",
        "description_en": (
            "Numeric cutoffs and quantified criteria (e.g., 'at least 2 facts', "
            "'~30% severe errors', '3 reasons') that replace vague qualitative "
            "descriptions with concrete numbers. Optimization frequently "
            "introduces numeric thresholds where the original rubric used "
            "imprecise terms like 'some' or 'several'."
        ),
        "cues_en": "at least, at most, <=, >=, N reasons/examples/sentences, N%",
        "regex": (
            r"at least|at most|<=|>=|\u2264|\u2265"
            r"|\b\d+\s*(?:reasons?|examples?|sentences?|words?|points?|facts?)\b"
            r"|\d+%|~\d+"
        ),
    },
    {
        "enabled": True,
        "pattern_id": "score_cap_demotion",
        "type_id": "rule_structure",
        "name_ja": "スコア上限・降格ルール",
        "description_ja": (
            "特定条件を満たさない場合にスコア上限を設定する、"
            "または強制的にスコアを降格させるハードルール。"
            "例: 'cannot receive 4 or higher'、'do not award 5'。"
            "最適化ルーブリックでは、過大評価を防ぐための"
            "明示的なキャップが導入される。"
        ),
        "name_en": "Score Cap / Demotion",
        "description_en": (
            "Hard constraints that cap the maximum achievable score or forcibly "
            "demote ratings when specific conditions are unmet. Examples: "
            "'cannot receive 4 or higher', 'do not award 5', 'downgrade to 2'. "
            "Refined rubrics add these guards to prevent systematic "
            "over-scoring of essays that superficially appear competent."
        ),
        "cues_en": "cannot be Score, must not receive, do not award, downgrade, demotion",
        "regex": r"cannot (?:be [Ss]core|receive|assign)|must not receive|do not award|\bdowngrade\b|\bdemotion\b",
    },
    {
        "enabled": False,
        "pattern_id": "negative_prescriptive",
        "type_id": "rule_structure",
        "name_ja": "禁止的指示",
        "description_ja": (
            "do not / must not / should not 等の禁止形で評価者の行動を制約する"
            "一般的ルール。score_cap_demotion がスコア値の上限・降格に特化する"
            "のに対し、こちらは 'do not upgrade for surface polish alone' 等の"
            "幅広い禁止事項を捕捉する。if_rules と共起しやすい。"
        ),
        "name_en": "Negative Prescriptive",
        "description_en": (
            "General prohibitions (do not / must not / should not) constraining "
            "rater behavior. Unlike score_cap_demotion which targets score-value "
            "constraints specifically, this pattern captures broader prohibitive "
            "instructions such as 'do not upgrade for surface polish alone'. "
            "Often co-occurs with conditional gating (if_rules)."
        ),
        "cues_en": "do not, must not, should not",
        "regex": r"\bdo not\b|\bmust not\b|\bshould not\b",
    },

    # ── Evidence Handling ───────────────────────────────────
    # エビデンスの質的評価と数え方に関するルール
    {
        "enabled": True,
        "pattern_id": "evidence_count_safeguard",
        "type_id": "evidence_handling",
        "name_ja": "エビデンス水増し防止",
        "description_ja": (
            "エビデンスの数え方を制約し、不当な水増しを防ぐルール群。"
            "(1) 機械的カウントの禁止 — 単純な数え上げではなく質的判断を求める "
            "指示 (do not count, not mechanically)。"
            "(2) 重複・言い換えの非カウント — repetition / restatement / rephrasing を "
            "別エビデンスとして数えない指示。"
            "最適化ルーブリックでは、多数のエビデンスを列挙するだけの"
            "エッセイが過大評価されるのを防ぐためにこれらの制約が追加される。"
            "旧 anti_mechanical_counting と repetition_noncount を統合。"
        ),
        "name_en": "Evidence Count Safeguard",
        "description_en": (
            "Rules preventing inflated evidence counts, combining two closely "
            "related aspects: (1) anti-mechanical counting — requiring qualitative "
            "judgment rather than naive tallying (e.g., 'do not count', "
            "'not mechanically'); (2) repetition / restatement non-counting — "
            "treating repeated or rephrased claims as a single piece of evidence "
            "(e.g., 'restatement', 'double-count', 'near-duplicate'). Both serve "
            "the same goal: preventing over-scoring of essays that superficially "
            "list many facts without genuine variety or depth."
        ),
        "cues_en": (
            "do not count, not mechanically, do not equate, not equivalent, "
            "repetition, restatement, double-count, rephrasing, near-duplicate"
        ),
        "regex": (
            r"do not count|not\s+mechanical|do not equate|not equivalent"
            r"|\brepetition\b|\brestatement\b|double-?count|\brephras|near-duplicate"
        ),
    },
    {
        "enabled": True,
        "pattern_id": "concrete_exemplification",
        "type_id": "evidence_handling",
        "name_ja": "具体例・例示",
        "description_ja": (
            "ルーブリックの具体性を測る2つのシグナルを捕捉する。"
            "(1) エッセイに具体例・逸話・明示的エビデンスを求める要件 "
            "(specific example, concrete detail, anecdote 等)。"
            "(2) ルーブリック自体が 'e.g.' や 'for example' を用いて "
            "採点基準を例示で具体化している箇所。"
            "最適化後のルーブリックでは、抽象的記述が具体例付きの"
            "説明に置き換わることが多く、このパターンの増加は"
            "ルーブリックの実用性向上を示唆する。"
        ),
        "name_en": "Concrete Exemplification",
        "description_en": (
            "Detects two related signals of rubric specificity: "
            "(1) requirements for essays to include concrete examples, "
            "anecdotes, or explicit evidence; "
            "(2) the rubric itself using illustrative examples "
            "(e.g., for example, for instance) to clarify scoring criteria. "
            "Refined rubrics frequently replace abstract descriptions with "
            "example-rich explanations, making this a strong indicator of "
            "rubric practicality improvement."
        ),
        "cues_en": (
            "specific example, concrete example, concrete detail, anecdote, "
            "personal experience, explicit evidence, e.g., for example, for instance"
        ),
        "regex": (
            r"specific example|concrete example|concrete detail"
            r"|\banecdote\b|personal experience|explicit evidence"
            r"|e\.g\.|for example|for instance"
        ),
    },
    {
        "enabled": False,
        "pattern_id": "offtopic_or_irrelevance",
        "type_id": "evidence_handling",
        "name_ja": "脱線・無関連への制御",
        "description_ja": (
            "off-topic や無関連内容への対応指示。エッセイが主題から逸脱した場合、"
            "または要約のみで分析を欠く場合の減点・処理ルール。"
            "最適化ルーブリックで明示的に追加されることがある。"
        ),
        "name_en": "Off-Topic / Irrelevance",
        "description_en": (
            "Rules for handling off-topic, irrelevant, or digressive essay "
            "content, as well as summary-only responses lacking analysis. "
            "Refined rubrics may add explicit off-topic handling rules "
            "that were absent in the original."
        ),
        "cues_en": "off-topic, irrelevant, digression, summary-only",
        "regex": r"off-?topic|\birrelevant\b|digression|summary-only",
    },
    {
        "enabled": False,
        "pattern_id": "elaboration_taxonomy",
        "type_id": "evidence_handling",
        "name_ja": "精緻化の段階分類",
        "description_ja": (
            "エビデンスの統合度を多段階で分類するラベル体系。"
            "list-only（列挙のみ）/ token linkage（形式的な紐付け）/ "
            "meaningful linkage（因果・推論的な紐付け）等のラベルを用いて、"
            "エビデンス活用の質を段階的に評価する。"
            "初期ルーブリックには存在せず、最適化で新規導入されることが多い。"
        ),
        "name_en": "Elaboration Taxonomy",
        "description_en": (
            "Multi-tier classification labels for depth of evidence integration. "
            "Labels like 'list-only' (mere enumeration), 'token linkage' "
            "(superficial connection), and 'meaningful linkage' (causal / "
            "inferential connection) create a graded scale of elaboration depth. "
            "These labels are typically absent in initial rubrics and introduced "
            "during optimization to help raters distinguish surface-level from "
            "genuine evidence use."
        ),
        "cues_en": "list-only, listing-only, min-elab, token linkage, meaningful linkage, partial support, weak evidence",
        "regex": r"list-only|listing-only|min-elab|token linkage|meaningful linkage|partial support|weak evidence|full-strength",
    },
    {
        "enabled": False,
        "pattern_id": "counterargument_nuance",
        "type_id": "evidence_handling",
        "name_ja": "反論・ニュアンス",
        "description_ja": (
            "反論 (counterargument)・反駁 (rebuttal)・トレードオフ・"
            "ニュアンスへの言及。高スコア帯で要求される分析的深さの"
            "指標であり、最適化ルーブリックでは上位スコアの弁別基準として"
            "導入されることがある。"
        ),
        "name_en": "Counterarg. / Nuance",
        "description_en": (
            "References to counterarguments, rebuttals, trade-offs, or nuanced "
            "analytical positions. Indicates expectations for higher-order "
            "thinking at upper score bands. Refined rubrics may introduce "
            "these as discriminators for top scores (e.g., score 5-6)."
        ),
        "cues_en": "counterargument, rebuttal, trade-off, nuance, synthesis",
        "regex": r"counterargument|rebuttal|trade-?off|\bnuanc|\bsynthesis\b",
    },

    # ── Writing Quality ─────────────────────────────────────
    # 文章構成・表面的言語品質に関する評価基準
    {
        "enabled": False,
        "pattern_id": "organization_coherence",
        "type_id": "writing_quality",
        "name_ja": "構成・一貫性",
        "description_ja": (
            "エッセイの構成 (organization)・一貫性 (coherence)・"
            "論理的流れ (logical flow)・段落構成 (paragraphing) 等、"
            "談話構造を明示的に評価する基準。"
            "初期ルーブリックにも存在するが、最適化でより"
            "具体的なしきい値と組み合わされることがある。"
        ),
        "name_en": "Organization / Coherence",
        "description_en": (
            "Criteria that explicitly evaluate discourse structure: "
            "organization, coherence, logical flow, transitions between "
            "ideas, and paragraphing. Present in both initial and optimized "
            "rubrics, but optimization may pair them with more specific "
            "quantitative thresholds."
        ),
        "cues_en": "organization, coherence, logical flow, transition, paragraphing",
        "regex": r"\borganization\b|\bcoherence\b|logical flow|\btransition|\bparagraph",
    },
    {
        "enabled": False,
        "pattern_id": "grammar_mechanics",
        "type_id": "writing_quality",
        "name_ja": "文法・メカニクス",
        "description_ja": (
            "grammar / mechanics / spelling / punctuation / syntax 等、"
            "言語形式の正確さを評価する基準。初期ルーブリックでは"
            "'errors are present' 程度の曖昧な記述が多いが、最適化後は"
            "'~30% severe-sentence threshold' 等の具体的しきい値に"
            "置き換わることがある。"
        ),
        "name_en": "Grammar / Mechanics",
        "description_en": (
            "Criteria evaluating language-form quality: grammar, mechanics, "
            "spelling, punctuation, and syntax. Initial rubrics tend to use "
            "vague descriptions ('errors are present'), while optimized "
            "rubrics may replace them with concrete severity thresholds "
            "(e.g., '~30% severe-sentence threshold')."
        ),
        "cues_en": "grammar, mechanics, spelling, punctuation, syntax",
        "regex": r"\bgrammar\b|\bmechanics\b|\bspelling\b|\bpunctuation\b|\bsyntax\b",
    },
]
