"""
Analyze eval_500.json and print paper-ready numbers.

Outputs:
  - Per-round CHAIR_i and CHAIR_s with relative reductions
  - LaTeX table rows ready to paste into Table 1
  - Caption-length and mention-count stats per round
  - Diminishing-returns delta analysis

Usage:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --input results/eval_500.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def load_results(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def print_summary(data: dict) -> None:
    summary = data["summary"]
    rounds = summary["per_round"]

    print("=" * 70)
    print("HEADLINE NUMBERS")
    print("=" * 70)
    print(f"Total images evaluated: {summary['num_samples']}")
    print(f"Wall-clock time:        {summary['wall_time_sec']:.0f}s "
          f"({summary['wall_time_sec'] / 60:.1f} min)")
    print()

    # Per-round table
    print(f"{'Round':<10}{'CHAIR_i':>10}{'CHAIR_s':>10}"
          f"{'#mentions':>12}{'#hallu':>10}{'#captions':>12}")
    print("-" * 64)
    for k in ["round_0", "round_1", "round_2"]:
        r = rounds[k]
        print(f"{k:<10}"
              f"{r['chair_i']:>10.4f}"
              f"{r['chair_s']:>10.4f}"
              f"{r['n_mentions']:>12d}"
              f"{r['n_hallucinations']:>10d}"
              f"{r['n_captions']:>12d}")
    print()

    # Relative reductions
    r0, r1, r2 = rounds["round_0"], rounds["round_1"], rounds["round_2"]
    print("=" * 70)
    print("RELATIVE REDUCTIONS (vs Round 0)")
    print("=" * 70)
    for label, r in [("Round 1", r1), ("Round 2", r2)]:
        chair_i_rel = (r0["chair_i"] - r["chair_i"]) / r0["chair_i"] * 100
        chair_s_rel = (r0["chair_s"] - r["chair_s"]) / r0["chair_s"] * 100
        print(f"{label}:  CHAIR_i {chair_i_rel:+.1f}%   CHAIR_s {chair_s_rel:+.1f}%")
    print()

    # Diminishing-returns analysis
    print("=" * 70)
    print("DIMINISHING RETURNS (per-round delta)")
    print("=" * 70)
    transitions = [("0->1", r0, r1), ("1->2", r1, r2)]
    for label, a, b in transitions:
        d_chair_s = (a["chair_s"] - b["chair_s"]) * 100  # in pp
        d_chair_i = (a["chair_i"] - b["chair_i"]) * 100
        d_hallu = a["n_hallucinations"] - b["n_hallucinations"]
        print(f"  Round {label}:  ΔCHAIR_s = {d_chair_s:+.2f}pp   "
              f"ΔCHAIR_i = {d_chair_i:+.2f}pp   "
              f"hallucinations removed = {d_hallu}")
    print()


def print_latex_table(data: dict) -> None:
    rounds = data["summary"]["per_round"]
    print("=" * 70)
    print("LATEX TABLE 1 — paste into your report")
    print("=" * 70)
    print()
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Method & CHAIR$_i$ $\downarrow$ & CHAIR$_s$ $\downarrow$ \\")
    print(r"\midrule")

    labels = {
        "round_0": "Qwen2.5-VL-7B (raw)",
        "round_1": "+ re-prompt $K{=}1$",
        "round_2": "+ re-prompt $K{=}2$",
    }
    for k, lbl in labels.items():
        r = rounds[k]
        print(f"{lbl} & {r['chair_i']*100:.1f}\\% & {r['chair_s']*100:.1f}\\% \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()


def print_caption_stats(data: dict) -> None:
    print("=" * 70)
    print("CAPTION-LENGTH AND MENTION STATS PER ROUND")
    print("=" * 70)
    rounds_data = [[], [], []]
    mention_counts = [[], [], []]
    for img in data["per_image"]:
        for k, r in enumerate(img["rounds"]):
            if k < 3:
                rounds_data[k].append(len(r["response"].split()))
                mention_counts[k].append(len(r["mentioned_coco_objects"]))

    print(f"{'Round':<10}{'mean_words':>12}{'median_words':>14}"
          f"{'mean_mentions':>16}")
    print("-" * 52)
    for k in range(3):
        if rounds_data[k]:
            mw = mean(rounds_data[k])
            md = median(rounds_data[k])
            mm = mean(mention_counts[k])
            print(f"round_{k}".ljust(10)
                  + f"{mw:>12.1f}{md:>14.1f}{mm:>16.2f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/eval_500.json")
    args = parser.parse_args()

    data = load_results(Path(args.input))
    print_summary(data)
    print_latex_table(data)
    print_caption_stats(data)


if __name__ == "__main__":
    main()
