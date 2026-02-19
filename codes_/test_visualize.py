import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# plt.style.use("ggplot")

line_width_pt = 219
line_height_pt = line_width_pt * (5/8)
fig_size = (line_width_pt / 72.0, line_height_pt / 72.0)  # inches
font_size_pt = 10
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": font_size_pt,
    "axes.titlesize": font_size_pt,
    "axes.labelsize": font_size_pt,
    "legend.fontsize": font_size_pt,
    "xtick.labelsize": font_size_pt,
    "ytick.labelsize": font_size_pt,
    "mathtext.fontset": "stix",
    "figure.figsize": fig_size,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.pad_inches": 0,
    "pdf.use14corefonts": False,
    "pdf.fonttype": 42,
})

# --- Dummy data ---
cats = ["A", "B", "C", "D", "E"]
vals = np.array([12, 18, 7, 15, 11])

# --- Plot ---
fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

ax.bar(cats, vals, label="Values")
ax.set_title("Dummy Bar Chart")
ax.set_xlabel("A figure with a caption")
ax.set_ylabel("Value")
ax.legend()

# 保存（fig_sizeを維持してラベル/凡例も含める）
fig.savefig("figure.pdf")