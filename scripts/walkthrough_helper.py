"""
Re-run entity extraction + CLIP scoring on a single COCO image's Round 0
caption. Used to populate the walkthrough subsection with real per-entity
CLIP scores that weren't saved in eval_500.json.

Usage:
  python scripts/walkthrough_helper.py --image_id 168714
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.coco_loader import COCOLoader
from src.modules.entity_extractor import EntityExtractor
from src.modules.clip_scorer import CLIPGroundingScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_id", type=int, required=True)
    parser.add_argument("--input", type=str, default="results/eval_500.json")
    parser.add_argument("--tau", type=float, default=0.22)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    img = next((x for x in data["per_image"] if x["image_id"] == args.image_id), None)
    if img is None:
        print(f"image_id {args.image_id} not in results")
        sys.exit(1)

    loader = COCOLoader()
    sample = loader.get_sample(args.image_id)

    extractor = EntityExtractor()
    scorer = CLIPGroundingScorer()

    print(f"=== Walkthrough for image_id={args.image_id} ===")
    print(f"GT objects (COCO): {sorted(sample.gt_objects)}")
    print(f"Image saved at: {sample.image_path}")
    print()

    for r in img["rounds"]:
        caption = r["response"]
        print(f"--- Round {r['round']} ---")
        print(f"Caption: {caption}")
        print()

        entities = extractor.extract(caption)
        if not entities:
            print("  No entities extracted.")
            print()
            continue

        phrases = [e.text for e in entities]
        scores = scorer.score_entities(sample.image, phrases)
        for e, s in zip(entities, scores):
            tag = "HALLU" if s.score < args.tau else "  OK "
            print(f"  [{tag}] {e.text:35s}  CLIP={s.score:+.4f}")
        print()
        print(f"  COCO mentions: {r['mentioned_coco_objects']}")
        print(f"  CHAIR hallucinations: {r['hallucinated_coco_objects']}")
        print()


if __name__ == "__main__":
    main()
