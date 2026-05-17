#!/usr/bin/env python3
"""Appendix掲載のregexカウント手順が期待通り動くかを確認するテスト。"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


PATTERNS = {
    "Conditional Gating":
        r"\bif\b|\bwhen\b|\bunless\b|\bprovided that\b",
    "Quantitative Threshold": (
        r"at least|at most|<=|>=|\u2264|\u2265"
        r"|\b\d+\s*(?:reasons?|examples?|sentences?|words?|points?|facts?)\b"
        r"|\d+%|~\d+"
    ),
    "Concrete Exemplification":
        r"e\.g\.|for example|for instance",
    "Score Cap / Demotion": (
        r"cannot (?:be [Ss]core|receive|assign)|must not receive|"
        r"do not award|\bdowngrade\b|\bdemotion\b"
    ),
    "Boundary / Tie-Break": (
        r"tie-?break|borderline|\bthreshold\b|"
        r"\d\s*vs\.?\s*\d|between\s+adjacent"
    ),
    "Stepwise Workflow":
        r"\bstep\s+\d|checklist|workflow|procedure|\bin order\b",
}


def count_matches(text: str, regex: str) -> int:
    return len(re.findall(regex, text, flags=re.IGNORECASE))


def compute_counts() -> dict[str, tuple[int, int, int]]:
    initial_text = Path("initial_rubric.txt").read_text(encoding="utf-8")
    refined_text = Path("best_rubric.txt").read_text(encoding="utf-8")

    rows: dict[str, tuple[int, int, int]] = {}
    for name, regex in PATTERNS.items():
        before = count_matches(initial_text, regex)
        after = count_matches(refined_text, regex)
        rows[name] = (before, after, after - before)
    return rows


def main() -> None:
    initial = "This rubric is general. For example, support should be clear."
    refined = (
        "If the essay gives at least 2 reasons and 3 examples, when each "
        "example is developed, do not award a 6 unless it addresses the "
        "threshold between adjacent scores. Step 1: apply the checklist in "
        "order. For instance, judge 4 vs 5 with this procedure."
    )
    expected = {
        "Conditional Gating": (0, 3, 3),
        "Quantitative Threshold": (0, 3, 3),
        "Concrete Exemplification": (1, 1, 0),
        "Score Cap / Demotion": (0, 1, 1),
        "Boundary / Tie-Break": (0, 3, 3),
        "Stepwise Workflow": (0, 4, 4),
    }

    old_cwd = Path.cwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            Path("initial_rubric.txt").write_text(initial, encoding="utf-8")
            Path("best_rubric.txt").write_text(refined, encoding="utf-8")
            actual = compute_counts()
        finally:
            os.chdir(old_cwd)

    assert actual == expected, f"unexpected counts: {actual!r}"
    print("OK: Appendix regex counting procedure produced expected counts.")
    for name, values in actual.items():
        print(f"{name}: before={values[0]}, after={values[1]}, delta={values[2]}")


if __name__ == "__main__":
    main()
