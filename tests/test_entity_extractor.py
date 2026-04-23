"""Smoke test for EntityExtractor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modules.entity_extractor import EntityExtractor


def main() -> None:
    extractor = EntityExtractor()

    captions = [
        # Typical VLM caption with a few objects.
        "A man is riding a bicycle next to a yellow giraffe in a kitchen.",
        # Caption with duplicates and pronouns.
        "The cat sat on the mat. It was a black cat with a red collar.",
        # Short caption.
        "A dog.",
        # Empty / whitespace.
        "   ",
        # Realistic COCO-style caption.
        "Two people are standing in front of a large refrigerator, "
        "and a microwave sits on the counter next to several dishes.",
    ]

    for i, caption in enumerate(captions, 1):
        print(f"\n[{i}] Caption: {caption!r}")
        entities = extractor.extract(caption)
        if not entities:
            print("    (no entities extracted)")
            continue
        for e in entities:
            print(f"    - {e.text!r:30s} (raw: {e.raw!r}, chars {e.start_char}..{e.end_char})")

    print("\nDone.")


if __name__ == "__main__":
    main()
