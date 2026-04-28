"""
Re-score eval_500.json's captions with the corrected CHAIR matcher.

We do NOT re-run the VLM. We use the captions already in eval_500.json
and re-apply the new (more conservative) CHAIR synonym map.

Output: results/eval_500_rescored.json (same schema as input)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics.chair import CHAIRMetric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/eval_500.json")
    parser.add_argument("--output", type=str, default="results/eval_500_rescored.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Build CHAIR metric with COCO categories from the data.
    # Collect all GT objects across the eval to build the category set.
    all_gt = set()
    for img in data["per_image"]:
        all_gt.update(img["gt_objects"])
    chair = CHAIRMetric(coco_categories=all_gt)

    per_round = [[], [], []]
    for img in data["per_image"]:
        gt = set(img["gt_objects"])
        new_rounds = []
        for r in img["rounds"]:
            cr = chair.score_one(r["response"], gt)
            new_round = dict(r)
            new_round["mentioned_coco_objects"] = sorted(cr.mentioned_objects)
            new_round["hallucinated_coco_objects"] = sorted(cr.hallucinated_objects)
            new_round["has_hallucination"] = cr.has_hallucination
            new_rounds.append(new_round)
        # carry-forward to fill 3 rounds for aggregation
        per_round_chair_results = []
        for r in new_rounds:
            cr = chair.score_one(r["response"], gt)
            per_round_chair_results.append(cr)
        for k in range(3):
            per_round[k].append(per_round_chair_results[k] if k < len(per_round_chair_results)
                                else per_round_chair_results[-1])
        img["rounds"] = new_rounds

    # Recompute summary
    new_summary = {
        "num_samples": data["summary"]["num_samples"],
        "wall_time_sec": data["summary"]["wall_time_sec"],
        "per_round": {},
    }
    for k in range(3):
        agg = chair.score_dataset(per_round[k])
        new_summary["per_round"][f"round_{k}"] = {
            "chair_i": round(agg.chair_i, 4),
            "chair_s": round(agg.chair_s, 4),
            "n_captions": agg.n_captions,
            "n_mentions": agg.n_mentions,
            "n_hallucinations": agg.n_hallucinations,
        }
    data["summary"] = new_summary
    data["config"]["rescored"] = True
    data["config"]["rescored_with"] = "src/metrics/chair.py (conservative synonym map, post-fix)"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {args.output}")
    print()
    print("=== Original vs Rescored ===")
    print(f"{'Round':<10}{'old CHAIR_s':>14}{'new CHAIR_s':>14}{'old hallu':>12}{'new hallu':>12}")
    print("-" * 62)
    with open(args.input) as f:
        old = json.load(f)
    for k in range(3):
        old_r = old["summary"]["per_round"][f"round_{k}"]
        new_r = new_summary["per_round"][f"round_{k}"]
        print(f"round_{k}".ljust(10)
              + f"{old_r['chair_s']:>14.4f}{new_r['chair_s']:>14.4f}"
              + f"{old_r['n_hallucinations']:>12d}{new_r['n_hallucinations']:>12d}")


if __name__ == "__main__":
    main()
