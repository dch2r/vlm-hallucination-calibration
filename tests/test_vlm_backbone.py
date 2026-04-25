"""
Smoke test for the Qwen2.5-VL backbone.

This test downloads ~5GB (int4-quantized weights) on first run and requires
a CUDA-capable GPU with at least ~7 GB free VRAM. Run it on Colab T4 or
NOTS, not on the Mac.

Usage:
    python tests/test_vlm_backbone.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from src.modules.vlm_backbone import Qwen25VLBackbone


# Public COCO sample image (a cat, well-known test asset).
SAMPLE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/"
    "resolve/main/pipeline-cat-chonk.jpeg"
)
SAMPLE_PATH = Path("/tmp/vlm_test_image.jpg")


def get_sample_image() -> Image.Image:
    if not SAMPLE_PATH.exists():
        print(f"Downloading sample image to {SAMPLE_PATH}...")
        urlretrieve(SAMPLE_URL, SAMPLE_PATH)
    return Image.open(SAMPLE_PATH).convert("RGB")


def main() -> None:
    print("=== Loading Qwen2.5-VL-7B (int4) ===")
    t0 = time.time()
    backbone = Qwen25VLBackbone(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        device="cuda",
        quantize_int4=True,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s\n")

    image = get_sample_image()

    prompts = [
        "Describe this image in one sentence.",
        "List the objects you see in this image, separated by commas.",
    ]

    for prompt in prompts:
        print(f"--- Prompt: {prompt!r} ---")
        t0 = time.time()
        resp = backbone.generate(image, prompt, max_new_tokens=64)
        gen_time = time.time() - t0
        print(f"  Response: {resp.text}")
        print(
            f"  ({resp.num_output_tokens} tokens in {gen_time:.1f}s, "
            f"{resp.num_output_tokens / gen_time:.1f} tok/s)"
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
