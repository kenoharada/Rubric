#!/usr/bin/env python3
"""
パターン概要の TikZ ダイアグラムを生成する。

pattern_config.py の enabled パターンを読み込み、
カテゴリ別にカラーボックスで図示した TikZ コードを出力する。

使用例:
  python make_pattern_overview_figure.py
  python make_pattern_overview_figure.py -o ../paper_latex/figures/pattern_overview.tex
"""

import argparse
from pathlib import Path

from pattern_config import PATTERNS

# パターンごとの (表示名, キーワード説明)
# カード幅に収まるよう短くする
PATTERN_SHORT = {
    "if_rules":                 ("Conditional Gating",      "if, when, unless"),
    "tie_breaker_boundary":     ("Boundary / Tie-Break",    "tie-break, borderline, N vs N"),
    "stepwise_process":         ("Stepwise Workflow",       "Step 1\\ldots, checklist, procedure"),
    "quantitative_thresholds":  ("Quantitative Threshold",  "at least N, N\\%, {\\texttildelow}N"),
    "score_cap_demotion":       ("Score Cap / Demotion",    "cannot receive, do not award"),
    "negative_prescriptive":    ("Negative Prescriptive",   "do not, must not, should not"),
    "evidence_count_safeguard": ("Count Safeguard",         "restatement, repetition"),
    "concrete_exemplification": ("Concrete Exemplification","e.g., specific example"),
    "offtopic_or_irrelevance":  ("Off-Topic / Irrelevance", "off-topic, irrelevant"),
    "elaboration_taxonomy":     ("Elaboration Taxonomy",    "list-only, token linkage"),
    "counterargument_nuance":   ("Counterarg. / Nuance",    "rebuttal, trade-off, synthesis"),
    "organization_coherence":   ("Organization / Coherence","organization, coherence"),
    "grammar_mechanics":        ("Grammar / Mechanics",     "grammar, spelling, punctuation"),
}


def get_enabled_patterns():
    return [p for p in PATTERNS if p.get("enabled", True)]


def generate_tikz() -> str:
    patterns = get_enabled_patterns()

    # ── TikZ ヘッダ ──
    lines: list[str] = []
    lines.append(r"\begin{figure}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tikzpicture}[")
    lines.append(r"  font=\small\sffamily,")
    lines.append(r"  every node/.style={inner sep=0pt, outer sep=0pt},")
    lines.append(r"]")
    lines.append("")
    lines.append(r"% ── colors ──")
    lines.append(r"\definecolor{PatHdr}{HTML}{2B6CB0}")
    lines.append(r"\definecolor{PatFrame}{HTML}{90BEE0}")
    lines.append(r"\definecolor{PatBg}{HTML}{EBF4FA}")
    lines.append("")

    # ── レイアウト定数 ──
    box_w = 7.0     # cm
    card_h = 0.40   # cm
    card_gap = 0.08  # cm
    hdr_h = 0.38    # cm
    pad = 0.10      # cm

    n = len(patterns)
    this_h = hdr_h + pad + n * card_h + (n - 1) * card_gap + pad

    x0 = 0.0
    y_top = 0.0
    y_bot = y_top - this_h

    lines.append(r"% ── Patterns ──")

    # 背景ボックス
    lines.append(
        rf"\fill[PatBg, rounded corners=3pt]"
        rf" ({x0:.2f},{y_bot:.2f})"
        rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
    )
    # 外枠
    lines.append(
        rf"\draw[PatFrame, rounded corners=3pt, line width=0.6pt]"
        rf" ({x0:.2f},{y_bot:.2f})"
        rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
    )

    # ヘッダ
    lines.append(r"\begin{scope}")
    lines.append(
        rf"  \clip[rounded corners=3pt]"
        rf" ({x0:.2f},{y_bot:.2f})"
        rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
    )
    lines.append(
        rf"  \fill[PatHdr]"
        rf" ({x0:.2f},{y_top - hdr_h:.2f})"
        rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
    )
    lines.append(r"\end{scope}")

    # ヘッダテキスト
    cx = x0 + box_w / 2
    cy = y_top - hdr_h / 2
    lines.append(
        rf"\node[anchor=center, text=white, font=\small\bfseries\sffamily]"
        rf" at ({cx:.2f},{cy:.2f}) {{Rubric Refinement Patterns}};"
    )

    # パターンカード
    card_y = y_top - hdr_h - pad
    card_pad_x = 0.10
    text_pad = 0.10

    for pi, p in enumerate(patterns):
        pid = p["pattern_id"]
        short = PATTERN_SHORT.get(pid)
        if short:
            name, cues = short
        else:
            name, cues = p["name_en"], p.get("cues_en", "")

        cy_top = card_y
        cy_bot = card_y - card_h
        cy_mid = (cy_top + cy_bot) / 2

        # カード背景
        lines.append(
            rf"\fill[white, rounded corners=2pt]"
            rf" ({x0 + card_pad_x:.2f},{cy_bot:.2f})"
            rf" rectangle ({x0 + box_w - card_pad_x:.2f},{cy_top:.2f});"
        )
        lines.append(
            rf"\draw[PatFrame, rounded corners=2pt, line width=0.4pt]"
            rf" ({x0 + card_pad_x:.2f},{cy_bot:.2f})"
            rf" rectangle ({x0 + box_w - card_pad_x:.2f},{cy_top:.2f});"
        )

        # パターン名 (左寄せ)
        tx_l = x0 + card_pad_x + text_pad
        lines.append(
            rf"\node[anchor=west, font=\footnotesize\bfseries]"
            rf" at ({tx_l:.2f},{cy_mid:.2f}) {{{name}}};"
        )
        # キーワード (右寄せ)
        tx_r = x0 + box_w - card_pad_x - text_pad
        lines.append(
            rf"\node[anchor=east, font={{\fontsize{{7.5}}{{9}}\selectfont}}, text=black!50]"
            rf" at ({tx_r:.2f},{cy_mid:.2f}) {{{cues}}};"
        )

        card_y = cy_bot - card_gap

    lines.append("")

    # ── 閉じ ──
    lines.append(r"\end{tikzpicture}")
    lines.append(
        r"\caption{Overview of regex-based patterns used to quantify "
        r"rubric changes through iterative refinement. "
        r"Each pattern is detected by case-insensitive keyword matching "
        r"against the rubric text.}"
    )
    lines.append(r"\label{fig:pattern_overview}")
    lines.append(r"\end{figure}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="パターン概要 TikZ ダイアグラムを生成する",
    )
    parser.add_argument("--output", "-o", default=None, help="出力 .tex ファイルパス")
    args = parser.parse_args()

    tikz = generate_tikz()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(tikz, encoding="utf-8")
        print(f"TikZ figure → {out_path}")
    else:
        print(tikz)


if __name__ == "__main__":
    main()
