"""
Figure: entity flow through the calibration pipeline.

Per round, shows three quantities aggregated over all 500 images:
  1. COCO objects mentioned in caption
  2. Entities CLIP flagged as ungrounded
  3. Mentioned objects not in COCO ground truth (true hallucinations)

For images that converged early (pipeline stopped before K=2 because
CLIP found nothing to flag), we carry forward the final round's
counts to later rounds. This matches the aggregation logic used in
experiments/run_eval.py for the headline Table 1 numbers, ensuring
the figure agrees with the paper's reported metrics.

Output: figures/fig_entity_flow.{pdf,png}
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

    # Aggregate per-round counts with carry-forward for early-converged images.
    n_mentions = [0, 0, 0]
    n_flagged = [0, 0, 0]
    n_hallu = [0, 0, 0]

    for img in data["per_image"]:
        rounds = img["rounds"]
        # For each of the 3 logical rounds, use the actual round if it exists,
        # otherwise carry forward the last available round.
        for k in range(3):
            r = rounds[k] if k < len(rounds) else rounds[-1]
            n_mentions[k] += len(r["mentioned_coco_objects"])
            n_flagged[k] += r.get("num_flagged_by_clip", 0)
            n_hallu[k] += len(r["hallucinated_coco_objects"])

    n = data["summary"]["num_samples"]
    rounds_lbl = ["Round 0\n(raw)", "Round 1\n(re-prompt)", "Round 2\n(re-prompt)"]
    x = np.arange(len(rounds_lbl))
    width = 0.26

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - width, n_mentions, width,
                label="COCO objects mentioned",
                color="#4C72B0", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x,         n_flagged,  width,
                label="CLIP-flagged as ungrounded",
                color="#DD8452", edgecolor="black", linewidth=0.5)
    b3 = ax.bar(x + width, n_hallu,    width,
                label="CHAIR-identified hallucinations",
                color="#C44E52", edgecolor="black", linewidth=0.5)

    for bars, vals in [(b1, n_mentions), (b2, n_flagged), (b3, n_hallu)]:
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max(n_mentions) * 0.012,
                    str(v), ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(rounds_lbl)
    ax.set_ylabel(f"Total count across {n} images")
    ax.set_title(
        "Entity flow through the calibration pipeline",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_ylim(0, max(n_mentions) * 1.18)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out / "fig_entity_flow.pdf", bbox_inches="tight")
    plt.savefig(out / "fig_entity_flow.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'fig_entity_flow.pdf'}")
    print(f"Saved: {out / 'fig_entity_flow.png'}")

    print()
    print("=== Counts (with carry-forward for early-converged images) ===")
    for k, lbl in enumerate(["Round 0", "Round 1", "Round 2"]):
        print(f"  {lbl}: mentions={n_mentions[k]}, "
              f"CLIP-flagged={n_flagged[k]}, "
              f"true hallucinations (CHAIR)={n_hallu[k]}")

    # Cross-check against summary block.
    print()
    print("=== Cross-check vs Table 1 summary ===")
    summary = data["summary"]["per_round"]
    for k in range(3):
        s_mentions = summary[f"round_{k}"]["n_mentions"]
        s_hallu = summary[f"round_{k}"]["n_hallucinations"]
        ok_m = "✓" if s_mentions == n_mentions[k] else "✗"
        ok_h = "✓" if s_hallu == n_hallu[k] else "✗"
        print(f"  Round {k}: figure mentions={n_mentions[k]} vs summary={s_mentions} {ok_m}    "
              f"figure hallu={n_hallu[k]} vs summary={s_hallu} {ok_h}")


if __name__ == "__main__":
    main()
