#!/usr/bin/env python3
"""
Rubric テキスト上のパターンマッチ箇所をハイライト表示する Figure を生成する。

best_rubric.txt を読み込み、pattern_config.py の有効パターンにマッチした部分を
パターンごとに異なる背景色でハイライトした PDF/PNG を出力する。

使用例:
  python make_pattern_highlight_on_rubric.py \
    --rubric ../optimization_results/ASAP2/openai_gpt-5-mini/base_expert_True_train100_iteration5_top3_bs4-8-12_mc4/best_rubric.txt

  python make_pattern_highlight_on_rubric.py \
    --rubric ../optimization_results/ASAP2/openai_gpt-5-mini/base_expert_True_train100_iteration5_top3_bs4-8-12_mc4/best_rubric.txt \
    -o ../../paper_latex/figures/rubric_highlight.pdf \
    --max-lines 40 --font-size 7
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

from pattern_config import PATTERNS


# ── パターンごとの背景色 (Material Design 200) ───────────
PATTERN_COLORS = {
    # Rule Structure
    "if_rules":                 "#FFF59D",  # yellow
    "tie_breaker_boundary":     "#90CAF9",  # blue
    "stepwise_process":         "#A5D6A7",  # green
    "quantitative_thresholds":  "#FFAB91",  # orange
    "score_cap_demotion":       "#CE93D8",  # purple
    # Evidence Handling
    "evidence_count_safeguard": "#FFCC80",  # amber
    "concrete_exemplification": "#80CBC4",  # teal
    # Disabled (kept for completeness)
    "negative_prescriptive":    "#EF9A9A",
    "offtopic_or_irrelevance":  "#B0BEC5",
    "elaboration_taxonomy":     "#BCAAA4",
    "counterargument_nuance":   "#F48FB1",
    "organization_coherence":   "#90CAF9",
    "grammar_mechanics":        "#E6EE9C",
}

PATTERN_SHORT_NAMES = {p["pattern_id"]: p["name_en"] for p in PATTERNS}


# ── ヘルパー関数 ──────────────────────────────────────────

def get_enabled_patterns():
    return [p for p in PATTERNS if p.get("enabled", True)]


def find_all_matches(text: str, patterns: list[dict]) -> list[dict]:
    """テキスト中の全パターンマッチを検出し位置付きで返す。"""
    matches = []
    for p in patterns:
        for m in re.finditer(p["regex"], text, re.IGNORECASE):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "pattern_id": p["pattern_id"],
                "matched_text": m.group(),
            })
    return sorted(matches, key=lambda x: (x["start"], -(x["end"] - x["start"])))


def assign_colors(text: str, matches: list[dict]):
    """各文字位置にパターン色・ID を割り当て (config 順で先着優先)。"""
    colors = [None] * len(text)
    pids = [None] * len(text)

    enabled = get_enabled_patterns()
    priority = {p["pattern_id"]: i for i, p in enumerate(enabled)}
    ordered = sorted(matches, key=lambda m: priority.get(m["pattern_id"], 999))

    for m in ordered:
        pid = m["pattern_id"]
        c = PATTERN_COLORS.get(pid)
        if not c:
            continue
        for i in range(m["start"], min(m["end"], len(text))):
            if colors[i] is None:
                colors[i] = c
                pids[i] = pid
    return colors, pids


def wrap_lines(text: str, char_colors: list, width: int):
    """テキストを行分割 + 折り返し。色配列も対応させる。"""
    raw = text.split("\n")
    lines, lcolors = [], []
    offset = 0
    for rl in raw:
        ll = len(rl)
        lc = char_colors[offset: offset + ll]
        if ll <= width:
            lines.append(rl)
            lcolors.append(lc)
        else:
            pos = 0
            while pos < ll:
                end = min(pos + width, ll)
                if end < ll:
                    sp = rl.rfind(" ", pos, end)
                    if sp > pos:
                        end = sp + 1
                lines.append(rl[pos:end])
                lcolors.append(lc[pos:end])
                pos = end
        offset += ll + 1
    return lines, lcolors


def split_line_segments(line_text: str, line_colors: list):
    """行テキストを (text, color_or_None) セグメントに分割。"""
    if not line_text:
        return [("", None)]
    # 色配列が短い場合はパディング
    lc = list(line_colors)
    if len(lc) < len(line_text):
        lc += [None] * (len(line_text) - len(lc))
    segments = []
    curr_color = lc[0]
    curr_text = ""
    for ch, col in zip(line_text, lc):
        if col == curr_color:
            curr_text += ch
        else:
            if curr_text:
                segments.append((curr_text, curr_color))
            curr_color = col
            curr_text = ch
    if curr_text:
        segments.append((curr_text, curr_color))
    return segments


# ── 描画 ─────────────────────────────────────────────────

def render_figure(
    lines: list[str],
    line_colors: list[list],
    used_pids: set[str],
    output_path: Path,
    font_size: float = 6.5,
    title: str = "",
):
    """ハイライト付きルーブリックを描画し PDF/PNG で保存する。

    セグメント単位でテキストを描画し、各テキストの実測幅をもとに
    ハイライト矩形を正確に配置する。
    """
    n_lines = len(lines)
    max_chars = max((len(l) for l in lines), default=80)

    # figure サイズ推定 (monospace 文字幅 ≈ 0.6 × font_size pt)
    cw_in = font_size * 0.6 / 72
    lh_in = font_size * 2.0 / 72

    margin_l, margin_r = 0.3, 0.3
    margin_t = 0.4 if title else 0.15
    margin_b = 0.7
    fig_w = max(max_chars * cw_in + margin_l + margin_r, 4.0)
    fig_h = max(n_lines * lh_in + margin_t + margin_b, 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_ylim(n_lines - 0.5, -0.5)  # y=0 → 先頭行
    ax.axis("off")

    # レンダラー初期化 & データ座標変換
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    # ── monospace 1文字幅を実測 (alpha=0 で描画後に削除) ──
    probe = ax.text(0, 0, "M" * 80, fontfamily="monospace",
                    fontsize=font_size, alpha=0)
    fig.canvas.draw()
    bb = probe.get_window_extent(renderer)
    pts = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    cw = (pts[1, 0] - pts[0, 0]) / 80  # 1 文字のデータ座標幅
    probe.remove()

    # xlim をテキスト幅に合わせて設定
    ax.set_xlim(-0.5, max_chars * cw + 0.5)

    # ── 各行をセグメント単位で描画 ──
    text_kw = dict(fontfamily="monospace", fontsize=font_size,
                   va="center", ha="left")

    for li in range(n_lines):
        segments = split_line_segments(lines[li], line_colors[li])
        x = 0.0  # データ座標

        for seg_text, seg_color in segments:
            if not seg_text:
                continue

            # テキスト描画
            t = ax.text(x, li, seg_text, zorder=2, **text_kw)

            # テキスト幅を実測
            ext = t.get_window_extent(renderer)
            d0, d1 = inv.transform([[ext.x0, ext.y0], [ext.x1, ext.y1]])
            seg_w = d1[0] - d0[0]

            # 色付きセグメントにはハイライト矩形
            if seg_color:
                ax.add_patch(
                    Rectangle(
                        (x - cw * 0.05, li - 0.42),
                        seg_w + cw * 0.1,
                        0.84,
                        facecolor=seg_color, edgecolor="none", zorder=1,
                    )
                )

            x += seg_w

    # ── タイトル ──
    if title:
        ax.set_title(title, fontsize=font_size + 2, fontweight="bold",
                     fontfamily="sans-serif", pad=8)

    # ── 凡例 ──
    handles = []
    for p in get_enabled_patterns():
        if p["pattern_id"] in used_pids:
            handles.append(
                mpatches.Patch(
                    facecolor=PATTERN_COLORS[p["pattern_id"]],
                    edgecolor="gray", linewidth=0.5,
                    label=PATTERN_SHORT_NAMES[p["pattern_id"]],
                )
            )
    if handles:
        ax.legend(
            handles=handles, loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(handles), 4),
            fontsize=font_size, frameon=True,
            edgecolor="gray", fancybox=True,
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Figure saved → {output_path}")


# ── メイン ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rubric テキストのパターンマッチ箇所をハイライト Figure で出力",
    )
    parser.add_argument("--rubric", "-r", required=True,
                        help="ルーブリックファイルのパス")
    parser.add_argument("--output", "-o", default=None,
                        help="出力 .pdf/.png パス")
    parser.add_argument("--max-lines", type=int, default=None,
                        help="表示する最大行数")
    parser.add_argument("--wrap-width", type=int, default=100,
                        help="折り返し幅 (文字数, デフォルト: 100)")
    parser.add_argument("--font-size", type=float, default=6.5,
                        help="フォントサイズ (pt, デフォルト: 6.5)")
    parser.add_argument("--title", default="", help="図タイトル")
    args = parser.parse_args()

    rubric_path = Path(args.rubric)
    if not rubric_path.exists():
        print(f"Error: {rubric_path} が見つかりません")
        return

    text = rubric_path.read_text(encoding="utf-8")
    patterns = get_enabled_patterns()
    print(f"Loaded {len(patterns)} enabled patterns")

    # マッチ検出 & 色割り当て
    matches = find_all_matches(text, patterns)
    char_colors, char_pids = assign_colors(text, matches)
    used_pids = {pid for pid in char_pids if pid is not None}

    # 統計表示
    pid_counts = Counter(m["pattern_id"] for m in matches)
    print(f"\nMatch summary ({len(matches)} total matches):")
    for p in patterns:
        cnt = pid_counts.get(p["pattern_id"], 0)
        if cnt > 0:
            print(f"  {PATTERN_SHORT_NAMES[p['pattern_id']]:<30s}: {cnt:>4d}")

    # 行分割 + 折り返し
    lines, lcolors = wrap_lines(text, char_colors, args.wrap_width)
    if args.max_lines and len(lines) > args.max_lines:
        lines = lines[:args.max_lines]
        lcolors = lcolors[:args.max_lines]

    print(f"\nRendering {len(lines)} lines ...")

    # 出力パス
    if args.output:
        out = Path(args.output)
    else:
        out = rubric_path.parent / f"{rubric_path.stem}_highlighted.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    render_figure(lines, lcolors, used_pids, out,
                  font_size=args.font_size, title=args.title)


if __name__ == "__main__":
    main()
