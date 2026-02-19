"""
Visualize train QWK improvement over optimization iterations,
grouped by Monte Carlo (MC) run count.
Step 0 is shown as a horizontal dotted line (human expert baseline).
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- ACL-style figure settings ----------
line_width_pt = 219
line_height_pt = line_width_pt * (5 / 8)
fig_size = (line_width_pt / 72.0, line_height_pt / 72.0)
font_size_pt = 12
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": font_size_pt,
    "axes.titlesize": font_size_pt - 1,
    "axes.labelsize": font_size_pt - 1,
    "legend.fontsize": font_size_pt - 3.5,
    "legend.title_fontsize": font_size_pt - 3.5,
    "xtick.labelsize": font_size_pt - 2,
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
# Run name regex:
# base_expert_{rationale}_train{N}_iteration{N}_top{N}_bs{sizes}_mc{N}
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
    "ets3": "TOEFL 11 (3-class)",
}

MODEL_DISPLAY = {
    "openai_gpt-5-mini": "GPT-5-mini",
    "google_gemini-3-flash-preview": "Gemini-3-Flash",
    "qwen_qwen3-next-80b-a3b-instruct": "Qwen3-80B",
    "openai_gpt-4.1": "GPT-4.1",
    "google_gemini-2.5-pro": "Gemini-2.5-Pro",
    "google_gemini-2.5-flash": "Gemini-2.5-Flash",
}

# ---------- filter: datasets and models to plot ----------
# Comment/uncomment to select which datasets and models to include
DATASETS_TO_PLOT = [
    "asap_1",
    "ASAP2",
    "ets3",
]

MODELS_TO_PLOT = [
    # "openai_gpt-5-mini",
    "google_gemini-3-flash-preview",
    # "qwen_qwen3-next-80b-a3b-instruct",
    # "openai_gpt-4.1",
    # "google_gemini-2.5-pro",
    # "google_gemini-2.5-flash",
]

MC_TO_PLOT = [
    1,
    # 2,
    4,
    # 8,
]

# Okabe-Ito colorblind-friendly palette
MC_COLORS = {1: "#0072B2", 2: "#E69F00", 4: "#009E73", 8: "#D55E00"}
MC_MARKERS = {1: "o", 2: "s", 4: "D", 8: "^"}

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
            # Skip runs where step 0 only
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
                "n_steps": len(qwk_values) - 1,  # excluding step 0
            })

df = pd.DataFrame(records)
print(f"Found {len(df)} expert optimization runs")
print(df[["dataset", "model", "rationale", "train_size", "mc", "n_steps"]].to_string())

# ---------- Filter ----------
df = df[df["n_steps"] >= 3]
df = df[df["dataset"].isin(DATASETS_TO_PLOT)]
df = df[df["model"].isin(MODELS_TO_PLOT)]
df = df[df["mc"].isin(MC_TO_PLOT)]

# ---------- plotting ----------
# Group by (dataset, model, rationale, train_size) — each group gets a subplot
# Within each group, lines are distinguished by MC count

group_keys = df.groupby(["dataset", "model"]).groups.keys()
group_keys = sorted(group_keys)

n_groups = len(group_keys)
if n_groups == 0:
    print("No data to plot.")
    exit()

# Layout: one figure per dataset, subplots per model
datasets_available = sorted(df["dataset"].unique())

for dataset in datasets_available:
    ds_df = df[df["dataset"] == dataset]
    models_in_ds = [m for m in MODELS_TO_PLOT if m in ds_df["model"].values]
    n_models = len(models_in_ds)

    fig_w = fig_size[0] * n_models
    fig_h = fig_size[1]
    fig, axes = plt.subplots(1, n_models, figsize=(fig_w, fig_h), squeeze=False)

    for col, model in enumerate(models_in_ds):
        ax = axes[0, col]
        sub_df = ds_df[ds_df["model"] == model]

        # Group by MC, and for each MC pick the longest run
        # (if there are multiple with same mc, pick the one with most steps)
        mc_groups = sub_df.groupby("mc")

        # Baseline: average Step 0 QWK across all MC runs
        all_step0 = [row["qwk_values"][0] for _, row in sub_df.iterrows()]
        baseline_qwk = np.mean(all_step0)
        ax.axhline(
            y=baseline_qwk,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="Step 0 (Human Expert Rubric)",
            zorder=1,
        )

        for mc_val, mc_df in sorted(mc_groups, key=lambda x: x[0]):
            # Pick the run with most steps for this MC
            best_row = mc_df.loc[mc_df["n_steps"].idxmax()]
            qwk_vals = best_row["qwk_values"]
            steps = list(range(len(qwk_vals)))

            color = MC_COLORS.get(mc_val, "black")
            marker = MC_MARKERS.get(mc_val, "x")

            # Plot optimization curve (steps 1..N)
            ax.plot(
                steps[1:],
                qwk_vals[1:],
                color=color,
                marker=marker,
                markersize=3,
                linewidth=1.2,
                label=f"MC={mc_val}",
                zorder=2,
            )

        ax.set_xlabel("Optimization Step")
        if col == 0:
            ax.set_ylabel("Train QWK")
        # ax.set_title(MODEL_DISPLAY.get(model, model))
        max_steps = max(len(r) for r in sub_df["qwk_values"])
        ax.set_xticks(range(1, max_steps))
        ax.grid(axis="y", alpha=0.3)

    # fig.suptitle(
    #     f"Iterative Refinement — {DATASET_DISPLAY.get(dataset, dataset)}",
    #     fontsize=font_size_pt,
    #     y=1.02,
    # )

    # Shared legend: MC lines first row, baseline second row
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Reorder for column-major legend: [MC=1, baseline, MC=4]
    # ncol=2 fills col-major → row 1: MC=1, MC=4 / row 2: baseline
    reorder = [1, 0, 2]
    handles = [handles[i] for i in reorder]
    labels = [labels[i] for i in reorder]
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    plt.tight_layout()

    out_base = f"fig_iterative_refinement_mc_{dataset}"
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_base}.png", bbox_inches="tight", dpi=300)
    print(f"Saved {out_base}.pdf / .png")
    plt.show()
