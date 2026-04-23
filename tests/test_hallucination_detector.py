"""Smoke test for HallucinationDetector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from src.modules.hallucination_detector import HallucinationDetector


def make_test_image() -> Image.Image:
    """Solid blue background with a red square in the center."""
    img = Image.new("RGB", (224, 224), color=(50, 80, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 160, 160], fill=(220, 30, 30))
    return img


def main() -> None:
    print("Initializing detector (CLIP weights cached from previous run)...")
    detector = HallucinationDetector()
    image = make_test_image()

    # A caption that mixes truly-grounded entities (red square, blue background)
    # with invented ones (yellow giraffe, elephant, person on a bicycle).
    caption = (
        "A red square is shown in front of a blue background, "
        "next to a yellow giraffe and an elephant. "
        "A person is riding a bicycle past the scene."
    )

    print(f"\nCaption: {caption}\n")

    for policy, kwargs in [
        ("absolute",   {"tau": 0.22}),
        ("percentile", {"percentile": 40.0}),
    ]:
        print(f"--- policy={policy} {kwargs} ---")
        result = detector.detect(image, caption, policy=policy, **kwargs)
        print(f"  effective cutoff: {result.threshold_used:+.4f}")
        for v in result.verdicts:
            tag = "HALLU" if v.is_hallucination else "  OK "
            print(f"  [{tag}] {v.entity:30s} {v.score:+.4f}")
        print(f"  => flagged {len(result.hallucinated)} / {len(result.verdicts)}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
