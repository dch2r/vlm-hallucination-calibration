"""
End-to-end smoke test for the calibration pipeline.

Loads Qwen2.5-VL-7B (int4) + CLIP scorer + entity extractor + detector,
runs one image through the full K=2 re-prompting loop, and prints the
per-round trajectory.

Requires GPU (Colab T4 or NOTS).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.modules.hallucination_detector import HallucinationDetector
from src.modules.vlm_backbone import Qwen25VLBackbone
from src.pipeline.calibration_pipeline import CalibrationPipeline


SAMPLE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/pipeline-cat-chonk.jpeg"
)
SAMPLE_PATH = Path("/tmp/vlm_pipeline_test.jpg")


def get_sample_image() -> Image.Image:
    if not SAMPLE_PATH.exists():
        print(f"Downloading sample image to {SAMPLE_PATH}...")
        urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return Image.open(SAMPLE_PATH).convert("RGB")


def main() -> None:
    print("=== Initializing pipeline ===")
    t0 = time.time()
    vlm = Qwen25VLBackbone(quantize_int4=True)
    detector = HallucinationDetector()
    pipeline = CalibrationPipeline(
        vlm=vlm,
        detector=detector,
        max_rounds=2,
        policy="percentile",
        percentile=30.0,
        max_new_tokens=64,
    )
    print(f"Pipeline ready in {time.time() - t0:.1f}s\n")

    image = get_sample_image()

    print("=== Running calibration loop ===")
    t0 = time.time()
    result = pipeline.run(
        image=image,
        initial_prompt=(
            "Describe this image in detail, mentioning all objects you see, "
            "including any vehicles, people, or animals."
        ),
    )
    print(f"Run complete in {time.time() - t0:.1f}s\n")

    # Per-round breakdown.
    for r in result.rounds:
        print(f"--- Round {r.round_index} ---")
        print(f"  Response: {r.response}")
        print(f"  Hallucinated: {r.num_hallucinated} / {len(r.detection.verdicts)}")
        if r.detection.verdicts:
            for v in r.detection.verdicts:
                tag = "HALLU" if v.is_hallucination else "  OK "
                print(f"    [{tag}] {v.entity:30s} {v.score:+.4f}")
        print()

    print("=== Summary ===")
    rates = result.hallucination_rate_per_round()
    for k, rate in enumerate(rates):
        print(f"  Round {k}: hallucination rate = {rate * 100:.1f}%")
    if result.converged_at is not None:
        print(f"  Converged at round {result.converged_at}")
    else:
        print(f"  Did not converge within {result.max_rounds} rounds")

    print("\nDone.")


if __name__ == "__main__":
    main()
