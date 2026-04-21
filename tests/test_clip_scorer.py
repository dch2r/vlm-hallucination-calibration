"""Smoke test for CLIPGroundingScorer."""

import sys
from pathlib import Path

# Add project root to sys.path so we can import src.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from src.modules.clip_scorer import CLIPGroundingScorer


def make_test_image() -> Image.Image:
    """Create a tiny synthetic image: solid blue background with a red square."""
    img = Image.new("RGB", (224, 224), color=(50, 80, 200))  # blue background
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 160, 160], fill=(220, 30, 30))    # red square
    return img


def main() -> None:
    print("Loading CLIP scorer (this downloads ~600MB on first run)...")
    scorer = CLIPGroundingScorer()
    print(f"  device = {scorer.device}")

    image = make_test_image()
    entities = [
        "a red square",
        "a blue background",
        "a yellow giraffe",
        "an elephant",
        "a person riding a bicycle",
    ]

    print("\nGrounding scores (higher = more visually present):")
    print("-" * 50)
    results = scorer.score_entities(image, entities)
    for r in sorted(results, key=lambda x: x.score, reverse=True):
        marker = " <-- expected high" if r.entity in ("a red square", "a blue background") else ""
        print(f"  {r.entity:30s}  {r.score:+.4f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
