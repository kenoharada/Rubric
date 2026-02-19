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
from collections import defaultdict
from pathlib import Path

from pattern_config import PATTERNS, TYPE_ORDER, TYPE_INFO

# ── カテゴリ配色 (TikZ color 定義用) ─────────────────────

TYPE_STYLES = {
    "rule_structure": {
        "header_bg": "RuleHdr",   # 定義は TikZ 側
        "card_frame": "RuleFrame",
        "card_bg": "RuleBg",
    },
    "evidence_handling": {
        "header_bg": "EvidHdr",
        "card_frame": "EvidFrame",
        "card_bg": "EvidBg",
    },
}

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

    pat_by_type: dict[str, list[dict]] = defaultdict(list)
    for p in patterns:
        pat_by_type[p["type_id"]].append(p)

    active_types = [t for t in TYPE_ORDER if pat_by_type.get(t)]

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
    lines.append(r"\definecolor{RuleHdr}{HTML}{2B6CB0}")
    lines.append(r"\definecolor{RuleFrame}{HTML}{90BEE0}")
    lines.append(r"\definecolor{RuleBg}{HTML}{EBF4FA}")
    lines.append(r"\definecolor{EvidHdr}{HTML}{B7791F}")
    lines.append(r"\definecolor{EvidFrame}{HTML}{E2C97A}")
    lines.append(r"\definecolor{EvidBg}{HTML}{FEFCF3}")
    lines.append("")

    # ── レイアウト定数 (上下配置・フル幅・1行カード) ──
    box_w = 7.0     # cm — ボックス幅 (カラム幅に合わせる)
    card_h = 0.40   # cm — パターンカード高さ (1行)
    card_gap = 0.08  # cm — カード間
    hdr_h = 0.38    # cm — ヘッダ高さ
    pad = 0.10      # cm — ヘッダ↔カード, カード↔底辺
    gap_y = 0.20    # cm — カテゴリ間の縦ギャップ

    # カテゴリボックス高さ算出
    def box_height(n_pats):
        return hdr_h + pad + n_pats * card_h + (n_pats - 1) * card_gap + pad

    # ── カテゴリを上から下へ描画 ──
    y_cursor = 0.0  # 上端から開始

    for ti, type_id in enumerate(active_types):
        pats = pat_by_type[type_id]
        styles = TYPE_STYLES.get(type_id, TYPE_STYLES["rule_structure"])
        info = TYPE_INFO[type_id]
        n = len(pats)
        this_h = box_height(n)

        x0 = 0.0
        y_top = -y_cursor
        y_bot = y_top - this_h

        lines.append(f"% ── {info['label_en']} ──")

        # 背景ボックス
        lines.append(
            rf"\fill[{styles['card_bg']}, rounded corners=3pt]"
            rf" ({x0:.2f},{y_bot:.2f})"
            rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
        )
        # 外枠
        lines.append(
            rf"\draw[{styles['card_frame']}, rounded corners=3pt, line width=0.6pt]"
            rf" ({x0:.2f},{y_bot:.2f})"
            rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
        )

        # ヘッダ (角丸の上半分 — clip で実現)
        lines.append(r"\begin{scope}")
        lines.append(
            rf"  \clip[rounded corners=3pt]"
            rf" ({x0:.2f},{y_bot:.2f})"
            rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
        )
        lines.append(
            rf"  \fill[{styles['header_bg']}]"
            rf" ({x0:.2f},{y_top - hdr_h:.2f})"
            rf" rectangle ({x0 + box_w:.2f},{y_top:.2f});"
        )
        lines.append(r"\end{scope}")

        # ヘッダテキスト
        cx = x0 + box_w / 2
        cy = y_top - hdr_h / 2
        lines.append(
            rf"\node[anchor=center, text=white, font=\small\bfseries\sffamily]"
            rf" at ({cx:.2f},{cy:.2f}) {{{info['label_en']}}};"
        )

        # パターンカード (1行: 名前=左, キーワード=右)
        card_y = y_top - hdr_h - pad
        card_pad_x = 0.10
        text_pad = 0.10  # カード内パディング

        for pi, p in enumerate(pats):
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
                rf"\draw[{styles['card_frame']}, rounded corners=2pt, line width=0.4pt]"
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

        y_cursor += this_h + gap_y

    # ── 閉じ ──
    lines.append(r"\end{tikzpicture}")
    lines.append(
        r"\caption{Overview of regex-based patterns used to quantify "
        r"rubric changes through iterative refinement. "
        r"\textsc{Rule Structure} patterns capture procedural and "
        r"structural constraints that guide the scoring process; "
        r"\textsc{Evidence Handling} patterns capture rules governing "
        r"how evidence quality and quantity are assessed. "
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
