"""
Visualize step-wise QWK improvements (deltas) as grouped bar charts.
X-axis groups: Δ(step0→step1), Δ(step1→step2), ...
Bars within each group: one per dataset, colored differently.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- ACL-style figure settings ----------
line_width_pt = 219
line_height_pt = line_width_pt * (5 / 8)
fig_size = (line_width_pt / 72.0, line_height_pt / 72.0)
font_size_pt = 11
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": font_size_pt,
    "axes.titlesize": font_size_pt - 1,
    "axes.labelsize": font_size_pt - 1,
    "legend.fontsize": font_size_pt - 3.5,
    "legend.title_fontsize": font_size_pt - 3.5,
    "xtick.labelsize": font_size_pt - 1,
    "ytick.labelsize": font_size_pt - 2,
    "mathtext.fontset": "stix",
    "figure.figsize": fig_size,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0,
    "pdf.use14corefonts": False,
    "pdf.fonttype": 42,
})

# ---------- configuration ----------
RESULT_DIR = "./optimization_results"
RUN_PATTERN = re.compile(
    r"^base_expert_(?P<rationale>True|False)"
    r"_train(?P<train_size>\d+)"
    r"_iteration(?P<iteration>\d+)"
    r"_top(?P<top_k>\d+)"
    r"_bs(?P<batch_sizes>[\d\-]+)"
    r"_mc(?P<mc>\d+)$"
)

DATASET_DISPLAY = {
    "asap_1": "ASAP",
    "ASAP2": "ASAP 2.0",
    "ets3": "TOEFL 11",
}

MODEL_DISPLAY = {
    "openai_gpt-5-mini": "GPT-5-mini",
    "google_gemini-3-flash-preview": "Gemini-3-Flash",
    "qwen_qwen3-next-80b-a3b-instruct": "Qwen3-80B",
    "openai_gpt-4.1": "GPT-4.1",
    "google_gemini-2.5-pro": "Gemini-2.5-Pro",
    "google_gemini-2.5-flash": "Gemini-2.5-Flash",
}

# ---------- filter ----------
DATASETS_TO_PLOT = [
    "asap_1",
    "ASAP2",
    "ets3",
]

MODELS_TO_PLOT = [
    "openai_gpt-5-mini",
    "google_gemini-3-flash-preview",
    "qwen_qwen3-next-80b-a3b-instruct",
    # "openai_gpt-4.1",
    # "google_gemini-2.5-pro",
    # "google_gemini-2.5-flash",
]

MC_TO_PLOT = [
    # 1,
    # 2,
    4,
    # 8,
]

RATIONALE_TO_PLOT = [
    "True",
    # "False",
]

TRAIN_SIZES_TO_PLOT = [
    # 10,
    # 20,
    # 50,
    100,
]

TOP_KS_TO_PLOT = [
    3,
]

BATCH_SIZES_TO_PLOT = [
    "4-8-12",
    # "2-4-8",
]

# Step pairs to show: each tuple is (from_step, to_step)
STEP_PAIRS = [(0, 1), (1, 3), (3, 5)]

# Okabe-Ito colorblind-friendly palette
DATASET_COLORS = {
    "asap_1": "#0072B2",   # blue
    "ASAP2": "#E69F00",    # orange
    "ets3": "#009E73",     # green
}

# ---------- data collection ----------
records = []
for dataset in sorted(os.listdir(RESULT_DIR)):
    ds_path = os.path.join(RESULT_DIR, dataset)
    if not os.path.isdir(ds_path):
        continue
    for model in sorted(os.listdir(ds_path)):
        model_path = os.path.join(ds_path, model)
        if not os.path.isdir(model_path):
            continue
        for run_name in sorted(os.listdir(model_path)):
            m = RUN_PATTERN.match(run_name)
            if m is None:
                continue
            qwk_path = os.path.join(model_path, run_name, "train_qwk.txt")
            if not os.path.exists(qwk_path):
                continue
            with open(qwk_path) as f:
                qwk_values = [float(line.strip()) for line in f if line.strip()]
            if len(qwk_values) < 2:
                continue
            records.append({
                "dataset": dataset,
                "model": model,
                "rationale": m.group("rationale"),
                "train_size": int(m.group("train_size")),
                "iteration": int(m.group("iteration")),
                "top_k": int(m.group("top_k")),
                "batch_sizes": m.group("batch_sizes"),
                "mc": int(m.group("mc")),
                "qwk_values": qwk_values,
                "n_steps": len(qwk_values) - 1,
            })

df = pd.DataFrame(records)
print(f"Found {len(df)} expert optimization runs")

# ---------- Filter ----------
df = df[df["n_steps"] >= max(t for _, t in STEP_PAIRS)]
df = df[df["dataset"].isin(DATASETS_TO_PLOT)]
df = df[df["model"].isin(MODELS_TO_PLOT)]
df = df[df["mc"].isin(MC_TO_PLOT)]
df = df[df["rationale"].isin(RATIONALE_TO_PLOT)]
df = df[df["train_size"].isin(TRAIN_SIZES_TO_PLOT)]
df = df[df["top_k"].isin(TOP_KS_TO_PLOT)]
df = df[df["batch_sizes"].isin(BATCH_SIZES_TO_PLOT)]

print(f"After filtering: {len(df)} runs")
print(df[["dataset", "model", "rationale", "train_size", "mc", "batch_sizes", "top_k", "n_steps"]].to_string())

if len(df) == 0:
    print("No data to plot after filtering.")
    exit()

# ---------- compute deltas ----------
# For each step pair (a, b), compute qwk[b] - qwk[a]
# Then average across all runs (models × MC) per dataset
delta_by_dataset = {}  # dataset -> array of length len(STEP_PAIRS)

for dataset in DATASETS_TO_PLOT:
    ds_df = df[df["dataset"] == dataset]
    if len(ds_df) == 0:
        continue
    all_deltas = []
    for _, row in ds_df.iterrows():
        qwk = row["qwk_values"]
        deltas = [max(qwk[a:b+1]) - max(qwk[:a+1]) for a, b in STEP_PAIRS if b < len(qwk)]
        if len(deltas) == len(STEP_PAIRS):
            all_deltas.append(deltas)
    if all_deltas:
        mean_deltas = np.mean(all_deltas, axis=0)
        delta_by_dataset[dataset] = mean_deltas

print("\nMean step-wise QWK deltas:")
for ds, deltas in delta_by_dataset.items():
    print(f"  {DATASET_DISPLAY.get(ds, ds)}: {['%.3f' % d for d in deltas]}")

# ---------- plotting ----------
datasets_with_data = [ds for ds in DATASETS_TO_PLOT if ds in delta_by_dataset]
n_datasets = len(datasets_with_data)

n_pairs = len(STEP_PAIRS)
x_labels = [f"$s_{{{a}}}\\!\\to\\!s_{{{b}}}$" for a, b in STEP_PAIRS]
x = np.arange(n_pairs)
bar_width = 0.8 / n_datasets

fig, ax = plt.subplots(figsize=fig_size)

for j, dataset in enumerate(datasets_with_data):
    deltas = delta_by_dataset[dataset]
    offset = (j - (n_datasets - 1) / 2) * bar_width
    color = DATASET_COLORS.get(dataset, f"C{j}")
    bars = ax.bar(
        x + offset,
        deltas,
        bar_width,
        label=DATASET_DISPLAY.get(dataset, dataset),
        color=color,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    # Annotate bar values
    # for bar, val in zip(bars, deltas):
    #     ax.text(
    #         bar.get_x() + bar.get_width() / 2,
    #         bar.get_height() + 0.003,
    #         f"{val:.2f}",
    #         ha="center",
    #         va="bottom",
    #         fontsize=font_size_pt - 4,
    #         color=color,
    #         fontweight="bold",
    #     )

ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.set_ylabel("QWK Improvement")
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_axisbelow(True)

ax.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.9,
    edgecolor="lightgray",
)

plt.tight_layout()

out_base = "fig_stepwise_improvement"
plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
plt.savefig(f"{out_base}.png", bbox_inches="tight", dpi=300)
print(f"\nSaved {out_base}.pdf / .png")
plt.show()
