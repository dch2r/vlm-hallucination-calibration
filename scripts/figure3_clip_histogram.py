"""
Figure 3: per-round hallucination calibration results.

Two panels:
  Left  - CHAIR_i and CHAIR_s line plot across rounds 0/1/2.
  Right - absolute hallucinations removed per re-prompting transition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/eval_500.json")
    parser.add_argument("--output_dir", type=str, default="figures")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.input) as f:
        data = json.load(f)

    rounds = data["summary"]["per_round"]
    keys = ["round_0", "round_1", "round_2"]
    n = data["summary"]["num_samples"]

    chair_i = [rounds[k]["chair_i"] * 100 for k in keys]
    chair_s = [rounds[k]["chair_s"] * 100 for k in keys]
    hallu = [rounds[k]["n_hallucinations"] for k in keys]
    round_labels = ["Round 0\n(raw)", "Round 1\n(re-prompt)", "Round 2\n(re-prompt)"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 3.8))

    # --- Left panel: CHAIR metrics across rounds ---
    x = np.arange(len(keys))
    ax_left.plot(x, chair_s, "-o", color="#C44E52", linewidth=2,
                 markersize=8, label=r"CHAIR$_s$ (caption-level)")
    ax_left.plot(x, chair_i, "-s", color="#4C72B0", linewidth=2,
                 markersize=8, label=r"CHAIR$_i$ (mention-level)")
    # Labels above CHAIR_s, also above CHAIR_i (no longer below)
    for xi, yi in zip(x, chair_s):
        ax_left.annotate(f"{yi:.1f}%", (xi, yi), xytext=(0, 9),
                         textcoords="offset points", ha="center", fontsize=10,
                         color="#C44E52", fontweight="bold")
    for xi, yi in zip(x, chair_i):
        ax_left.annotate(f"{yi:.1f}%", (xi, yi), xytext=(0, 9),
                         textcoords="offset points", ha="center", fontsize=10,
                         color="#4C72B0", fontweight="bold")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(round_labels)
    ax_left.set_ylabel("Hallucination rate (%)")
    ax_left.set_title("(a) Hallucination rate per round", fontsize=11)
    ax_left.set_ylim(0, max(chair_s) * 1.25)
    ax_left.legend(loc="upper right", frameon=False, fontsize=9)
    ax_left.grid(axis="y", alpha=0.3)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # --- Right panel: hallucinations removed per transition ---
    transitions = ["Round 0 → 1", "Round 1 → 2"]
    removed = [hallu[0] - hallu[1], hallu[1] - hallu[2]]
    bars = ax_right.bar(transitions, removed, color=["#55A868", "#A0A0A0"],
                        edgecolor="black", linewidth=0.5, width=0.55)
    # Pluralize correctly: 1 -> "1 fix", >1 -> "N fixes"
    for rect, v in zip(bars, removed):
        label = f"{v} fix" if v == 1 else f"{v} fixes"
        ax_right.text(rect.get_x() + rect.get_width() / 2,
                      rect.get_height() + 0.15,
                      label, ha="center", va="bottom", fontsize=11,
                      fontweight="bold")
    ax_right.set_ylabel("Hallucinations eliminated")
    ax_right.set_title("(b) Diminishing returns", fontsize=11)
    ax_right.set_ylim(0, max(removed) * 1.4)
    ax_right.grid(axis="y", alpha=0.3)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.suptitle(
        f"Per-round calibration on Qwen2.5-VL-7B (MS COCO val 2014, n={n})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out / "fig3_calibration.pdf", bbox_inches="tight")
    plt.savefig(out / "fig3_calibration.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'fig3_calibration.pdf'}")
    print(f"Saved: {out / 'fig3_calibration.png'}")


if __name__ == "__main__":
    main()
