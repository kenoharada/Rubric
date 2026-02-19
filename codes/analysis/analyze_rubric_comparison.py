"""
初期ルーブリック（人手作成）と最適化後ルーブリックの定量的比較を行うスクリプト。

計測項目:
- 単語数: 初期・最適化後それぞれの単語数と、増加量（最適化後 - 初期）
- Overlap: difflib.SequenceMatcher で順序を保った共通単語列を検出し、
  最適化後ルーブリックの単語のうち初期ルーブリックと共通する割合を算出
- QWK: evaluation_results/ からベースライン（初期ルーブリック）と最適化後のテストQWKを取得

Usage:
    python analysis/analyze_rubric_comparison.py [--run_configs CONFIG1 CONFIG2 ...] [--output_dir DIR]

Examples:
    # 特定のrunを指定して実行（zero_shot_ プレフィックス付きでもなしでも可）
    python analysis/analyze_rubric_comparison.py --run_configs \
        zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4 \
        zero_shot_base_simplest_True_train100_iteration5_top3_bs4-8-12_mc4

    # 全件走査（従来通り）
    python analysis/analyze_rubric_comparison.py
"""

import os
import sys
import argparse
import difflib
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIMIZATION_RESULTS_DIR = os.path.join(BASE_DIR, 'optimization_results')
EVALUATION_RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_results')

MODEL_DISPLAY_NAMES = {
    'google_gemini-3-flash-preview': 'Gemini 3 Flash',
    'openai_gpt-5-mini': 'GPT-5 mini',
    'qwen_qwen3-next-80b-a3b-instruct': 'Qwen3 Next',
}

DATASET_DISPLAY_NAMES = {
    'asap_1': 'ASAP',
    'ASAP2': 'ASAP 2.0',
    'ets3': 'TOEFL11',
}


def read_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def read_qwk(path):
    """QWKファイルを読み込んでfloatで返す。存在しなければNone。"""
    content = read_file(path)
    if content is None:
        return None
    try:
        return float(content.strip())
    except ValueError:
        return None


def compute_word_overlap(original: str, optimized: str):
    """
    SequenceMatcherで単語レベルの共通部分列を検出。

    順序を保った一致ブロックを見つけるため、単に同じ単語の出現回数ではなく
    「元のルーブリックの構造がどれだけ保持されているか」を反映する。

    Returns:
        common_words: 共通単語数
        overlap_rate: 最適化後ルーブリックの単語のうち初期ルーブリックと共通する割合
    """
    orig_words = original.split()
    opt_words = optimized.split()

    matcher = difflib.SequenceMatcher(None, orig_words, opt_words, autojunk=False)
    matching_blocks = matcher.get_matching_blocks()

    common_word_count = sum(block.size for block in matching_blocks)

    overlap_rate = common_word_count / len(opt_words) if opt_words else 0.0

    return common_word_count, overlap_rate


def _normalize_config(config_name):
    """
    zero_shot_ プレフィックスを除去して optimization_results 側の config名に変換。
    例: "zero_shot_base_expert_True_..." -> "base_expert_True_..."
         "base_expert_True_..." -> "base_expert_True_..." (そのまま)
    """
    if config_name.startswith('zero_shot_'):
        return config_name[len('zero_shot_'):]
    return config_name


def collect_results(run_configs=None):
    """
    optimization_results/ を走査してルーブリック比較データを収集。
    同時に evaluation_results/ から対応するQWKを取得。

    Args:
        run_configs: 指定時はこれらの設定名のみ収集（zero_shot_プレフィックス付きでも可）。
                     Noneなら全件。

    Returns:
        結果dictのリスト。
    """
    results = []

    # run_configs が指定されていれば、optimization_results側の名前に正規化
    allowed_configs = None
    if run_configs:
        allowed_configs = set(_normalize_config(c) for c in run_configs)

    if not os.path.exists(OPTIMIZATION_RESULTS_DIR):
        print(f"Directory not found: {OPTIMIZATION_RESULTS_DIR}")
        return results

    for dataset in sorted(os.listdir(OPTIMIZATION_RESULTS_DIR)):
        dataset_dir = os.path.join(OPTIMIZATION_RESULTS_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            continue

        for model in sorted(os.listdir(dataset_dir)):
            model_dir = os.path.join(dataset_dir, model)
            if not os.path.isdir(model_dir):
                continue

            for config in sorted(os.listdir(model_dir)):
                if allowed_configs and config not in allowed_configs:
                    continue

                config_dir = os.path.join(model_dir, config)
                if not os.path.isdir(config_dir):
                    continue

                initial_path = os.path.join(config_dir, 'initial_rubric.txt')
                best_path = os.path.join(config_dir, 'best_rubric.txt')

                initial = read_file(initial_path)
                best = read_file(best_path)

                if initial is None or best is None:
                    continue

                # config名をパース
                # Format: base_expert_True_train100_iteration5_top3_bs4-8-12_mc4
                parts = config.split('_')
                seed_prompt = parts[1] if len(parts) > 1 else 'unknown'

                # 文字数
                init_chars = len(initial)
                opt_chars = len(best)
                char_increase = opt_chars - init_chars

                # 単語数
                init_words = len(initial.split())
                opt_words = len(best.split())

                # 単語レベル共通部分
                common_words, word_overlap = compute_word_overlap(initial, best)

                # QWKを取得
                # ベースライン: evaluation_results/{dataset}/{model}/zero_shot_no_{seed}/qwk.txt
                baseline_qwk_path = os.path.join(
                    EVALUATION_RESULTS_DIR, dataset, model,
                    f'zero_shot_no_{seed_prompt}', 'qwk.txt'
                )
                baseline_qwk = read_qwk(baseline_qwk_path)

                # 最適化後: evaluation_results/{dataset}/{model}/zero_shot_{config}/qwk.txt
                optimized_qwk_path = os.path.join(
                    EVALUATION_RESULTS_DIR, dataset, model,
                    f'zero_shot_{config}', 'qwk.txt'
                )
                optimized_qwk = read_qwk(optimized_qwk_path)

                results.append({
                    'dataset': dataset,
                    'model': model,
                    'config': config,
                    'seed_prompt': seed_prompt,
                    'init_chars': init_chars,
                    'opt_chars': opt_chars,
                    'char_increase': char_increase,
                    'init_words': init_words,
                    'opt_words': opt_words,
                    'common_words': common_words,
                    'word_overlap': word_overlap,
                    'baseline_qwk': baseline_qwk,
                    'optimized_qwk': optimized_qwk,
                })

    return results


def _format_qwk(qwk):
    """QWK値をフォーマット。Noneなら '--'。"""
    if qwk is None:
        return '--'
    return f"{qwk:.3f}"


def _format_delta(delta):
    """文字数増減をフォーマット。正なら+、負なら-を付ける。"""
    if delta >= 0:
        return f"+{delta:,}"
    else:
        return f"{delta:,}"


def _compute_delta_qwk(baseline_qwk, optimized_qwk):
    """ΔQWK を計算。どちらかがNoneならNone。"""
    if baseline_qwk is None or optimized_qwk is None:
        return None
    return optimized_qwk - baseline_qwk


def _format_delta_qwk(delta_qwk):
    """ΔQWK をフォーマット。正なら+、Noneなら '--'。"""
    if delta_qwk is None:
        return '--'
    if delta_qwk >= 0:
        return f"+{delta_qwk:.3f}"
    return f"{delta_qwk:.3f}"


def generate_latex_table(results):
    """
    メインのLaTeX表を生成（expert seed, 1行 = 1 dataset/model）。
    """
    if not results:
        return "% No results found."

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\caption{Quantitative comparison between seed and refined rubrics. '
                 r'The \textit{Seed} rubric is the human-authored rubric used as the starting point for optimization; '
                 r'\textit{Refined} is the rubric after iterative refinement. '
                 r'Word counts are shown for each, with \textit{$\Delta$\,Words} indicating the change. '
                 r'\textit{Overlap} is the percentage of refined rubric words that also appear '
                 r'in the seed rubric (order-preserving longest common subsequence). '
                 r'$\Delta$\,QWK is the improvement in test-set Quadratic Weighted Kappa '
                 r'after refinement (Refined $-$ Seed).}')
    lines.append(r'\label{tab:rubric_comparison}')
    lines.append(r'\begin{tabular}{ll rrr r r}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Model & Seed & Refined & $\Delta$\,Words & Overlap (\%) & $\Delta$\,QWK \\')
    lines.append(r'\midrule')

    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r['dataset']].append(r)

    prev_dataset = None
    dataset_order = [k for k in DATASET_DISPLAY_NAMES if k in by_dataset]
    for dataset in dataset_order:
        if prev_dataset is not None:
            lines.append(r'\midrule')
        prev_dataset = dataset

        dataset_results = sorted(by_dataset[dataset], key=lambda x: x['model'])
        for r in dataset_results:
            ds_name = DATASET_DISPLAY_NAMES.get(r['dataset'], r['dataset'])
            model_name = MODEL_DISPLAY_NAMES.get(r['model'], r['model'])
            word_delta = r['opt_words'] - r['init_words']
            delta_qwk = _compute_delta_qwk(r['baseline_qwk'], r['optimized_qwk'])
            lines.append(
                f"  {ds_name} & {model_name} & "
                f"{r['init_words']:,} & {r['opt_words']:,} & "
                f"{_format_delta(word_delta)} & "
                f"{r['word_overlap']*100:.1f} & "
                f"{_format_delta_qwk(delta_qwk)} \\\\"
            )
    lines.append(r'\midrule')
    avg_init = sum(r['init_words'] for r in results) / len(results)
    avg_opt = sum(r['opt_words'] for r in results) / len(results)
    avg_increase = avg_opt - avg_init
    avg_overlap = sum(r['word_overlap'] for r in results) / len(results) * 100
    delta_qwk_vals = [_compute_delta_qwk(r['baseline_qwk'], r['optimized_qwk'])
                      for r in results]
    delta_qwk_vals = [v for v in delta_qwk_vals if v is not None]
    avg_delta_qwk = sum(delta_qwk_vals) / len(delta_qwk_vals) if delta_qwk_vals else None
    lines.append(
        f"  \\multicolumn{{2}}{{l}}{{\\textbf{{Average}}}} & "
        f"{avg_init:,.0f} & {avg_opt:,.0f} & "
        f"{_format_delta(int(avg_increase))} & "
        f"{avg_overlap:.1f} & "
        f"{_format_delta_qwk(avg_delta_qwk)} \\\\"
    )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)


def generate_latex_table_by_seed(results):
    """
    seed prompt別（expert / simplest）のLaTeX表を生成。
    """
    if not results:
        return "% No results found."

    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\caption{Rubric comparison by seed prompt type. '
                 r'\textit{Seed} and \textit{Refined} are word counts of the initial and refined rubrics; '
                 r'\textit{$\Delta$\,Words} is the word count change. '
                 r'\textit{Overlap} is the percentage of refined rubric words that also appear '
                 r'in the seed rubric. '
                 r'$\Delta$\,QWK is the test-set QWK improvement after refinement.}')
    lines.append(r'\label{tab:rubric_comparison_by_seed}')
    lines.append(r'\begin{tabular}{lll rrr r r}')
    lines.append(r'\toprule')
    lines.append(r'Dataset & Model & Prompt & Seed & Refined & $\Delta$\,Words & Overlap (\%) & $\Delta$\,QWK \\')
    lines.append(r'\midrule')

    by_dataset = defaultdict(list)
    for r in results:
        by_dataset[r['dataset']].append(r)

    prev_dataset = None
    dataset_order = [k for k in DATASET_DISPLAY_NAMES if k in by_dataset]
    for dataset in dataset_order:
        if prev_dataset is not None:
            lines.append(r'\midrule')
        prev_dataset = dataset

        dataset_results = sorted(by_dataset[dataset], key=lambda x: (x['model'], x['seed_prompt']))
        for r in dataset_results:
            ds_name = DATASET_DISPLAY_NAMES.get(r['dataset'], r['dataset'])
            model_name = MODEL_DISPLAY_NAMES.get(r['model'], r['model'])
            word_delta = r['opt_words'] - r['init_words']
            delta_qwk = _compute_delta_qwk(r['baseline_qwk'], r['optimized_qwk'])
            lines.append(
                f"  {ds_name} & {model_name} & {r['seed_prompt']} & "
                f"{r['init_words']:,} & {r['opt_words']:,} & "
                f"{_format_delta(word_delta)} & "
                f"{r['word_overlap']*100:.1f} & "
                f"{_format_delta_qwk(delta_qwk)} \\\\"
            )

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table*}')

    return '\n'.join(lines)


def print_summary(results):
    """人間可読なサマリーを出力。"""
    print(f"\n{'='*80}")
    print(f"Rubric Comparison Summary ({len(results)} runs)")
    print(f"{'='*80}\n")

    for r in results:
        ds_name = DATASET_DISPLAY_NAMES.get(r['dataset'], r['dataset'])
        model_name = MODEL_DISPLAY_NAMES.get(r['model'], r['model'])
        print(f"{ds_name} / {model_name} / {r['seed_prompt']}")
        print(f"  Config: {r['config']}")
        print(f"  Characters: {r['init_chars']:,} -> {r['opt_chars']:,} (+{r['char_increase']:,})")
        print(f"  Words:      {r['init_words']:,} -> {r['opt_words']:,}")
        print(f"  Common words: {r['common_words']:,}")
        print(f"    Overlap (最適化後の何%が初期と共通): {r['word_overlap']*100:.1f}%")
        print(f"  QWK: {_format_qwk(r['baseline_qwk'])} -> {_format_qwk(r['optimized_qwk'])}")
        print()


def _select_best_config(results_list):
    """
    同一キーの複数configから代表的な1つを選択。
    iteration5 > mc4 > train100 > True(with_rationale) の優先度。
    """
    best_per_key = {}
    for r in results_list:
        key = (r['dataset'], r['model'], r.get('_group_key', ''))
        config = r['config']
        score = 0
        if 'iteration5' in config:
            score += 100
        if 'mc4' in config:
            score += 50
        if 'train100' in config:
            score += 25
        if '_True_' in config:
            score += 10
        if key not in best_per_key or score > best_per_key[key][1]:
            best_per_key[key] = (r, score)
    return [v[0] for v in best_per_key.values()]


def main():
    parser = argparse.ArgumentParser(
        description='初期ルーブリックと最適化後ルーブリックの定量的比較')
    parser.add_argument('--run_configs', type=str, nargs='+', default=None,
                        help='特定のrun configのみ分析。複数指定可。zero_shot_プレフィックス付きでもOK。'
                             ' (例: zero_shot_base_expert_True_train100_iteration5_top3_bs4-8-12_mc4)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='LaTeX出力先ディレクトリ (デフォルト: paper_latex/)')
    parser.add_argument('--seed_prompts', type=str, nargs='+', default=None,
                        help='seed promptでフィルタ (例: expert simplest)')
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(BASE_DIR, '..', 'paper_latex')

    results = collect_results(args.run_configs)

    if not results:
        print("No results found. Check your optimization_results directory.")
        return

    if args.seed_prompts:
        results = [r for r in results if r['seed_prompt'] in args.seed_prompts]

    print_summary(results)

    # run_configs指定時は_select_best_configをスキップ（指定したものをそのまま使う）
    use_auto_select = args.run_configs is None

    # === メイン表: expert seed, 1行 = 1 (dataset, model) ===
    expert_results = [r for r in results if r['seed_prompt'] == 'expert']
    if expert_results:
        if use_auto_select:
            for r in expert_results:
                r['_group_key'] = ''
            main_results = _select_best_config(expert_results)
        else:
            main_results = expert_results
        main_results.sort(key=lambda x: (x['dataset'], x['model']))

        latex_main = generate_latex_table(main_results)
        print("\n" + "="*80)
        print("LaTeX Table (Main Config - Expert Seed)")
        print("="*80)
        print(latex_main)

        output_path = os.path.join(output_dir, 'figures', 'tab_rubric_comparison.tex')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex_main)
        print(f"\nSaved to: {output_path}")

    # === Seed別表: expert + simplest ===
    seed_types = set(r['seed_prompt'] for r in results)
    if len(seed_types) > 1:
        if use_auto_select:
            for r in results:
                r['_group_key'] = r['seed_prompt']
            seed_results = _select_best_config(results)
        else:
            seed_results = list(results)
        seed_results.sort(key=lambda x: (x['dataset'], x['model'], x['seed_prompt']))

        latex_seed = generate_latex_table_by_seed(seed_results)
        print("\n" + "="*80)
        print("LaTeX Table (By Seed Prompt)")
        print("="*80)
        print(latex_seed)

        output_path2 = os.path.join(output_dir, 'figures', 'tab_rubric_comparison_by_seed.tex')
        with open(output_path2, 'w') as f:
            f.write(latex_seed)
        print(f"\nSaved to: {output_path2}")


if __name__ == '__main__':
    main()
