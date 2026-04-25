"""
Vision-Language Model backbone wrapper.

Provides a single ``VLMBackbone`` interface that wraps a pretrained VLM
(Qwen2.5-VL by default) and exposes a ``generate(image, prompt)`` method
returning the model's free-form text response.

Memory notes (T4, 15GB VRAM):
    * Qwen2.5-VL-7B int4  : ~5-6 GB  (recommended for T4)
    * Qwen2.5-VL-7B bf16  : ~15 GB   (tight, may OOM)
    * Qwen2.5-VL-3B fp16  : ~7 GB    (smaller fallback)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from PIL import Image


@dataclass
class VLMResponse:
    """Single VLM generation result."""
    text: str
    prompt: str
    backbone: str
    num_input_tokens: int
    num_output_tokens: int


class VLMBackbone:
    """Abstract backbone interface."""

    name: str = "abstract"

    def generate(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> VLMResponse:
        raise NotImplementedError


def _resolve_qwen25_class():
    """
    Locate the correct model class for Qwen2.5-VL.

    transformers 4.45+ exposes ``Qwen2_5_VLForConditionalGeneration``, but the
    public top-level import path is unreliable across versions. We try the
    most specific submodule first, then the top-level, then fall back to the
    older Qwen2-VL class (which loads Qwen2.5 weights but warns).
    """
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )
        return Qwen2_5_VLForConditionalGeneration
    except Exception:
        pass
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    except Exception:
        pass
    from transformers import Qwen2VLForConditionalGeneration
    return Qwen2VLForConditionalGeneration


class Qwen25VLBackbone(VLMBackbone):
    """Qwen2.5-VL backbone with optional int4 quantization."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        quantize_int4: bool = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ) -> None:
        from transformers import AutoProcessor

        self.model_id = model_id
        self.device = device
        self.name = f"qwen25vl({'int4' if quantize_int4 else 'bf16'})"

        model_cls = _resolve_qwen25_class()
        print(f"[VLMBackbone] Using model class: {model_cls.__name__}")

        # device_map="auto" is required when using bitsandbytes 4-bit
        # quantization so accelerate places quantized layers correctly
        # without trying to initialize byte-typed weights on CPU first.
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": False,
        }

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[img],
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to the same device as the model's first parameter.
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else 1.0,
            )

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]

        decoded = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return VLMResponse(
            text=decoded,
            prompt=prompt,
            backbone=self.name,
            num_input_tokens=int(input_len),
            num_output_tokens=int(output_ids.shape[1]),
        )
