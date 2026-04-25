"""
Vision-Language Model backbone wrapper.

Provides a single ``VLMBackbone`` interface that wraps a pretrained VLM
(Qwen2.5-VL by default) and exposes a ``generate(image, prompt)`` method
returning the model's free-form text response.

The backbone is designed to be backbone-agnostic at the call site: the rest
of the pipeline (entity extraction, CLIP scoring, re-prompting) does not
care which underlying model is used. To swap backbones, write a new
subclass and pass it to the pipeline.

Memory notes (T4, 15GB VRAM):
    * Qwen2.5-VL-7B int4  : ~5-6 GB  (recommended for T4)
    * Qwen2.5-VL-7B bf16  : ~15 GB   (tight, may OOM)
    * Qwen2.5-VL-3B fp16  : ~7 GB    (smaller fallback)

The model is lazy-loaded inside ``__init__`` only when ``device`` is
explicitly cuda. This lets us import the module on a CPU-only machine
(e.g. a laptop) for syntax checking without triggering a 16GB download.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from PIL import Image


@dataclass
class VLMResponse:
    """Single VLM generation result."""
    text: str                  # Decoded response text (no special tokens)
    prompt: str                # The user prompt that was sent
    backbone: str              # Model identifier, e.g. "Qwen/Qwen2.5-VL-7B-Instruct"
    num_input_tokens: int      # Length of input ids (image+text)
    num_output_tokens: int     # Length of generated ids


class VLMBackbone:
    """Abstract backbone interface — subclasses implement ``generate``."""

    name: str = "abstract"

    def generate(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> VLMResponse:
        raise NotImplementedError


class Qwen25VLBackbone(VLMBackbone):
    """
    Qwen2.5-VL backbone with optional int4 quantization.

    Example:
        from PIL import Image
        backbone = Qwen25VLBackbone(quantize_int4=True)
        img = Image.open("test.jpg").convert("RGB")
        resp = backbone.generate(img, "Describe this image in one sentence.")
        print(resp.text)
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        quantize_int4: bool = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ) -> None:
        # Imports are scoped here so this module can be imported on CPU-only
        # machines without pulling in heavy CUDA-only deps at import time.
        from transformers import (
            AutoProcessor,
            Qwen2VLForConditionalGeneration,
        )

        self.model_id = model_id
        self.device = device
        self.name = f"qwen25vl({'int4' if quantize_int4 else 'bf16'})"

        # Try to use the dedicated Qwen2_5 class if available; fall back to
        # the Qwen2 class which loads Qwen2.5 weights identically.
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model_cls = Qwen2_5_VLForConditionalGeneration
        except ImportError:
            model_cls = Qwen2VLForConditionalGeneration

        load_kwargs = {"device_map": device, "trust_remote_code": False}

        if quantize_int4:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.model = model_cls.from_pretrained(model_id, **load_kwargs)
        self.model.eval()

    def _load_image(self, image: Union[Image.Image, str]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def generate(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> VLMResponse:
        img = self._load_image(image)

        # Qwen uses an OpenAI-style chat message list with image placeholders.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template -> textual prompt with <image> placeholders.
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
            )

        # Strip the input prompt tokens from the output.
        input_len = inputs.input_ids.shape[1]
        output_ids = generated_ids[:, input_len:]

        decoded = self.processor.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return VLMResponse(
            text=decoded,
            prompt=prompt,
            backbone=self.name,
            num_input_tokens=int(input_len),
            num_output_tokens=int(output_ids.shape[1]),
        )
