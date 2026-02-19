"""
Analysis: Effect of Iterative Refinement across Monte Carlo Runs on ASAP 2.0
Results are loaded from optimization_results/ and evaluation_results/ directories.
"""

import os
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPT_DIR = os.path.join(BASE_DIR, 'optimization_results', 'ASAP2')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation_results', 'ASAP2')

MODELS = [
    ('openai_gpt-5-mini', 'GPT-5-mini'),
    ('qwen_qwen3-next-80b-a3b-instruct', 'Qwen3-80B-A3B'),
    ('google_gemini-3-flash-preview', 'Gemini-3-flash'),
]
MC_RUNS = [1, 2, 4]
CONFIG_TEMPLATE = 'base_expert_True_train100_iteration5_top3_bs4-8-12_mc{mc}'


def read_lines(filepath):
    """Read a file and return stripped non-empty lines."""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_train_qwk(model_dir, mc):
    """Load train_qwk.txt: one QWK value per line (step 0 through step N)."""
    config = CONFIG_TEMPLATE.format(mc=mc)
    path = os.path.join(OPT_DIR, model_dir, config, 'train_qwk.txt')
    return [float(v) for v in read_lines(path)]


def load_train_accuracy(model_dir, mc):
    """Load train_accuracy.txt: one accuracy value per line."""
    config = CONFIG_TEMPLATE.format(mc=mc)
    path = os.path.join(OPT_DIR, model_dir, config, 'train_accuracy.txt')
    return [float(v) for v in read_lines(path)]


def load_top_candidates(model_dir, mc):
    """Load train_score_top_candidates.txt and parse into structured data.
    Returns dict: {step: [(top_k, accuracy, qwk), ...]}
    """
    config = CONFIG_TEMPLATE.format(mc=mc)
    path = os.path.join(OPT_DIR, model_dir, config, 'train_score_top_candidates.txt')
    lines = read_lines(path)
    results = {}
    i = 0
    while i < len(lines) - 1:
        # Parse accuracy line
        acc_match = re.match(r'Step (\d+), Top (\d+) Candidate, Training Accuracy: (.+)', lines[i])
        qwk_match = re.match(r'Step (\d+), Top (\d+) Candidate, Training QWK: (.+)', lines[i + 1])
        if acc_match and qwk_match:
            step = int(acc_match.group(1))
            top_k = int(acc_match.group(2))
            acc = float(acc_match.group(3))
            qwk = float(qwk_match.group(3))
            if step not in results:
                results[step] = []
            results[step].append((top_k, acc, qwk))
        i += 2
    return results


def load_test_results(model_dir, mc, method):
    """Load test accuracy and QWK from evaluation_results."""
    config = CONFIG_TEMPLATE.format(mc=mc)
    eval_dir = os.path.join(EVAL_DIR, model_dir, f'{method}_{config}')
    if not os.path.isdir(eval_dir):
        return None, None
    acc_path = os.path.join(eval_dir, 'accuracy.txt')
    qwk_path = os.path.join(eval_dir, 'qwk.txt')
    acc = float(read_lines(acc_path)[0]) if os.path.exists(acc_path) else None
    qwk = float(read_lines(qwk_path)[0]) if os.path.exists(qwk_path) else None
    return acc, qwk


# ==============================
# Load all data
# ==============================
train_qwk_data = {}   # (model_dir, mc) -> [qwk_step0, qwk_step1, ...]
train_acc_data = {}
top_candidates_data = {}
test_data = {}         # (model_dir, mc, method) -> (acc, qwk)

for model_dir, display_name in MODELS:
    for mc in MC_RUNS:
        train_qwk_data[(model_dir, mc)] = load_train_qwk(model_dir, mc)
        train_acc_data[(model_dir, mc)] = load_train_accuracy(model_dir, mc)
        top_candidates_data[(model_dir, mc)] = load_top_candidates(model_dir, mc)
        for method in ['zero_shot', 'few_shot']:
            test_data[(model_dir, mc, method)] = load_test_results(model_dir, mc, method)

# ==============================
# FIGURE 1: Training QWK trajectories
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

colors = {1: '#2196F3', 2: '#FF9800', 4: '#4CAF50'}
markers = {1: 'o', 2: 's', 4: 'D'}

for idx, (model_dir, display_name) in enumerate(MODELS):
    ax = axes[idx]
    for mc in MC_RUNS:
        data = train_qwk_data[(model_dir, mc)]
        # Skip step 0 (initial evaluation)
        steps = list(range(1, len(data)))
        ax.plot(steps, data[1:], color=colors[mc], marker=markers[mc], linewidth=2,
                markersize=8, label=f'MC={mc}', alpha=0.9)

    ax.set_title(display_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration Step', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Training QWK', fontsize=12)
    ax.set_xticks(steps)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.0, 0.85)

fig.suptitle('Effect of Monte Carlo Runs on Training QWK During Iterative Refinement (ASAP 2.0)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig_iterative_refinement_qwk.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(BASE_DIR, 'fig_iterative_refinement_qwk.png'), bbox_inches='tight', dpi=300)
plt.close()

# ==============================
# FIGURE 2: Training Accuracy trajectories
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for idx, (model_dir, display_name) in enumerate(MODELS):
    ax = axes[idx]
    for mc in MC_RUNS:
        data = train_acc_data[(model_dir, mc)]
        # Skip step 0 (initial evaluation)
        steps = list(range(1, len(data)))
        ax.plot(steps, data[1:], color=colors[mc], marker=markers[mc], linewidth=2,
                markersize=8, label=f'MC={mc}', alpha=0.9)

    ax.set_title(display_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration Step', fontsize=12)
    if idx == 0:
        ax.set_ylabel('Training Accuracy', fontsize=12)
    ax.set_xticks(steps)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.2, 0.75)

fig.suptitle('Effect of Monte Carlo Runs on Training Accuracy During Iterative Refinement (ASAP 2.0)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig_iterative_refinement_acc.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(BASE_DIR, 'fig_iterative_refinement_acc.png'), bbox_inches='tight', dpi=300)
plt.close()

# ==============================
# FIGURE 3: Test set QWK comparison (bar chart)
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

model_display_names = [d for _, d in MODELS]
x = np.arange(len(MODELS))
width = 0.25

for ax_idx, method in enumerate(['zero_shot', 'few_shot']):
    ax = axes[ax_idx]
    for i, mc in enumerate(MC_RUNS):
        vals = []
        for model_dir, _ in MODELS:
            _, qwk = test_data[(model_dir, mc, method)]
            vals.append(qwk if qwk is not None else 0)
        bars = ax.bar(x + i * width, vals, width, label=f'MC={mc}',
                      color=colors[mc], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_title(f'Test QWK ({method.replace("_", "-")})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Test QWK', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_display_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, 0.9)

fig.suptitle('Test Set QWK by Monte Carlo Runs (ASAP 2.0)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig_test_qwk_by_mc.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(BASE_DIR, 'fig_test_qwk_by_mc.png'), bbox_inches='tight', dpi=300)
plt.close()

# ==============================
# FIGURE 4: Top-k candidate spread (GPT-5-mini, MC=1 vs MC=4)
# ==============================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
topk_colors = {1: '#D32F2F', 2: '#7B1FA2', 3: '#00796B'}
topk_styles = {1: ('o', '-'), 2: ('s', '--'), 3: ('D', ':')}

gpt5_dir = MODELS[0][0]
for ax_idx, mc in enumerate([1, 4]):
    ax = axes[ax_idx]
    candidates = top_candidates_data[(gpt5_dir, mc)]
    all_steps = sorted(candidates.keys())

    for k in [1, 2, 3]:
        qwk_vals = []
        for step in all_steps:
            step_candidates = candidates[step]
            match = [c for c in step_candidates if c[0] == k]
            qwk_vals.append(match[0][2] if match else None)
        marker, linestyle = topk_styles[k]
        ax.plot(all_steps, qwk_vals, marker=marker, linestyle=linestyle,
                color=topk_colors[k], linewidth=2, markersize=8, label=f'Top-{k}')

    # Fill between top-1 and top-3
    top1_vals = [candidates[s][0][2] for s in all_steps]
    top3_vals = []
    for s in all_steps:
        match = [c for c in candidates[s] if c[0] == 3]
        top3_vals.append(match[0][2] if match else top1_vals[all_steps.index(s)])
    ax.fill_between(all_steps, top3_vals, top1_vals, alpha=0.15, color='gray')

    ax.set_title(f'GPT-5-mini (MC={mc}): Top-k Candidate Spread', fontsize=13, fontweight='bold')
    ax.set_xlabel('Iteration Step', fontsize=12)
    ax.set_ylabel('Training QWK', fontsize=12)
    ax.set_xticks(all_steps)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 0.65)

fig.suptitle('Top-k Candidate Diversity: MC=1 vs MC=4', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig_topk_spread.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(BASE_DIR, 'fig_topk_spread.png'), bbox_inches='tight', dpi=300)
plt.close()

# ==============================
# FIGURE 5: QWK improvement (delta from step 0)
# ==============================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for idx, (model_dir, display_name) in enumerate(MODELS):
    ax = axes[idx]
    for mc in MC_RUNS:
        data = train_qwk_data[(model_dir, mc)]
        # Skip step 0, show delta from step 0
        steps = list(range(1, len(data)))
        deltas = [d - data[0] for d in data[1:]]
        ax.plot(steps, deltas, color=colors[mc], marker=markers[mc], linewidth=2,
                markersize=8, label=f'MC={mc}', alpha=0.9)

    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_title(display_name, fontsize=14, fontweight='bold')
    ax.set_xlabel('Iteration Step', fontsize=12)
    if idx == 0:
        ax.set_ylabel('QWK Improvement (delta from Step 0)', fontsize=12)
    ax.set_xticks(steps)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

fig.suptitle('QWK Improvement Over Iterations by Monte Carlo Runs (ASAP 2.0)',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'fig_qwk_improvement.pdf'), bbox_inches='tight', dpi=300)
plt.savefig(os.path.join(BASE_DIR, 'fig_qwk_improvement.png'), bbox_inches='tight', dpi=300)
plt.close()

# ==============================
# Summary Table (printed)
# ==============================
print("=" * 100)
print("SUMMARY: Effect of Iterative Refinement by Monte Carlo Runs (ASAP 2.0)")
print("=" * 100)

print("\n--- Training QWK: Step 1 -> Final (Step 5) ---")
for model_dir, display_name in MODELS:
    print(f"\n{display_name}:")
    for mc in MC_RUNS:
        data = train_qwk_data[(model_dir, mc)]
        delta = data[-1] - data[1]
        print(f"  MC={mc}: {data[1]:.4f} -> {data[-1]:.4f} (delta: +{delta:.4f})")

print("\n--- Test QWK (zero-shot / few-shot) ---")
for model_dir, display_name in MODELS:
    print(f"\n{display_name}:")
    for mc in MC_RUNS:
        _, zs_qwk = test_data[(model_dir, mc, 'zero_shot')]
        _, fs_qwk = test_data[(model_dir, mc, 'few_shot')]
        print(f"  MC={mc}: zero-shot={zs_qwk:.4f}, few-shot={fs_qwk:.4f}")

# ==============================
# LaTeX Tables
# ==============================
print("\n" + "=" * 100)
print("LaTeX Table: Training QWK Progression")
print("=" * 100)

print(r"""
\begin{table}[t]
\centering
\small
\caption{Training QWK across iterations with different Monte Carlo sampling runs on ASAP 2.0.
Higher MC runs increase search diversity per iteration.}
\label{tab:mc_runs_training}
\begin{tabular}{ll|ccccc}
\toprule
\textbf{Model} & \textbf{MC} & \textbf{Step 1} & \textbf{Step 2} & \textbf{Step 3} & \textbf{Step 4} & \textbf{Step 5} \\
\midrule""")

for model_dir, display_name in MODELS:
    for i, mc in enumerate(MC_RUNS):
        data = train_qwk_data[(model_dir, mc)]
        model_col = f'\\multirow{{3}}{{*}}{{{display_name}}}' if i == 0 else ''
        vals = ' & '.join([f'{v:.3f}' for v in data[1:]])
        print(f"{model_col} & {mc} & {vals} \\\\")
    print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

print("\n" + "=" * 100)
print("LaTeX Table: Test QWK by MC Runs")
print("=" * 100)

print(r"""
\begin{table}[t]
\centering
\small
\caption{Test QWK on ASAP 2.0 with rubrics optimized using different numbers of Monte Carlo runs.}
\label{tab:mc_runs_test}
\begin{tabular}{ll|cc}
\toprule
\textbf{Model} & \textbf{MC} & \textbf{Zero-shot QWK} & \textbf{Few-shot QWK} \\
\midrule""")

for model_dir, display_name in MODELS:
    for i, mc in enumerate(MC_RUNS):
        model_col = f'\\multirow{{3}}{{*}}{{{display_name}}}' if i == 0 else ''
        _, zs_qwk = test_data[(model_dir, mc, 'zero_shot')]
        _, fs_qwk = test_data[(model_dir, mc, 'few_shot')]
        print(f"{model_col} & {mc} & {zs_qwk:.3f} & {fs_qwk:.3f} \\\\")
    print(r"\midrule")

print(r"""\bottomrule
\end{tabular}
\end{table}""")

print(f"\nAll figures saved to {BASE_DIR}/")
print("- fig_iterative_refinement_qwk.pdf/png")
print("- fig_iterative_refinement_acc.pdf/png")
print("- fig_test_qwk_by_mc.pdf/png")
print("- fig_topk_spread.pdf/png")
print("- fig_qwk_improvement.pdf/png")
