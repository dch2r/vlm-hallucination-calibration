"""
Manual audit of CHAIR-flagged hallucinations.

For each Round-0 hallucination in eval_500.json, prints:
  - image_id
  - GT objects (COCO)
  - Round 0 caption with the flagged object highlighted
  - Brief context (the surrounding 5 words)

Then offers an interactive prompt to label each one as:
  R = Real hallucination (object truly not in image and not COCO-listed)
  S = CHAIR substring artifact (compound noun like 'hot dog' -> 'dog')
  G = COCO annotation gap (object visible in image, just not labeled)
  C = Color/word collision (e.g., 'orange' the color flagged as fruit)
  A = Ambiguous / can't tell

Saves labels to results/audit_round0.json.

Usage:
    python scripts/audit_hallucinations.py            # interactive
    python scripts/audit_hallucinations.py --report   # show summary of saved labels
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABEL_OPTIONS = {
    "r": "real",
    "s": "substring_artifact",
    "g": "annotation_gap",
    "c": "color_collision",
    "a": "ambiguous",
    "o": "other",
}


def find_context(caption: str, term: str, window: int = 5) -> str:
    """Return the substring around `term` in `caption`."""
    pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
    m = pattern.search(caption)
    if not m:
        return f"[{term}] (term not found verbatim — likely synonym/lemma match)"
    start = max(0, m.start() - 40)
    end = min(len(caption), m.end() + 40)
    snippet = caption[start:end]
    # Bold-mark the term
    snippet = pattern.sub(f"**{term.upper()}**", snippet)
    return snippet


def open_image_in_finder(image_id: int):
    """Try to open the COCO image in Preview for visual inspection."""
    import subprocess
    path = Path.home() / ".cache/coco/val2014" / f"COCO_val2014_{image_id:012d}.jpg"
    if path.exists():
        subprocess.run(["open", str(path)])
    else:
        print(f"  (image not yet cached at {path})")


def interactive(args):
    with open(args.input) as f:
        data = json.load(f)

    out_path = Path(args.output)
    labels = {}
    if out_path.exists():
        with open(out_path) as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} prior labels from {out_path}")

    cases = []
    for img in data["per_image"]:
        r0 = img["rounds"][0]
        for h in r0["hallucinated_coco_objects"]:
            key = f"{img['image_id']}::{h}"
            if key in labels:
                continue
            cases.append({
                "key": key,
                "image_id": img["image_id"],
                "object": h,
                "gt": img["gt_objects"],
                "caption": r0["response"],
            })

    print(f"\n{len(cases)} unlabeled cases to review. (Already labeled: {len(labels)})")
    print(f"Labels: " + ", ".join(f"[{k}]={v}" for k, v in LABEL_OPTIONS.items()))
    print(f"Type the letter (r/s/g/c/a/o), 'i' to open image, or 'q' to save and quit.\n")

    for i, case in enumerate(cases):
        print("=" * 70)
        print(f"[{i+1}/{len(cases)}]  image_id={case['image_id']}  flagged={case['object']!r}")
        print(f"  GT: {case['gt']}")
        print(f"  Context: {find_context(case['caption'], case['object'])}")
        print(f"  Full caption: {case['caption']}")
        print()

        while True:
            ans = input("Label [r/s/g/c/a/o, i=image, q=quit]: ").strip().lower()
            if ans == "q":
                _save(out_path, labels)
                print(f"Saved {len(labels)} labels to {out_path}")
                return
            if ans == "i":
                open_image_in_finder(case["image_id"])
                continue
            if ans in LABEL_OPTIONS:
                labels[case["key"]] = LABEL_OPTIONS[ans]
                _save(out_path, labels)
                break
            print("  Unknown option. Try again.")

    _save(out_path, labels)
    print(f"\nDone. {len(labels)} total labels saved to {out_path}")


def _save(path: Path, labels: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2)


def report(args):
    out_path = Path(args.output)
    if not out_path.exists():
        print(f"No audit file at {out_path}")
        return

    with open(out_path) as f:
        labels = json.load(f)

    from collections import Counter
    counts = Counter(labels.values())

    print(f"=== Audit summary ({len(labels)} labels) ===")
    total = len(labels)
    for label, count in counts.most_common():
        pct = count / total * 100
        print(f"  {label:25s}  {count:3d}  ({pct:5.1f}%)")
    print()
    print(f"Real hallucinations: {counts.get('real', 0)} / {total}  "
          f"({counts.get('real', 0) / total * 100:.1f}%)")
    print(f"Matcher artifacts:   {counts.get('substring_artifact', 0) + counts.get('color_collision', 0)} "
          f"/ {total}")
    print(f"COCO annotation gaps: {counts.get('annotation_gap', 0)} / {total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="results/eval_500.json")
    parser.add_argument("--output", type=str, default="results/audit_round0.json")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    if args.report:
        report(args)
    else:
        interactive(args)


if __name__ == "__main__":
    main()
