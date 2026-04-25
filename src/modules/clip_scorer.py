"""
CLIP-based grounding scorer.

For each (image, entity-phrase) pair, computes the cosine similarity
between the CLIP image embedding and the CLIP text embedding of the
phrase. Returns a list of (entity, score) pairs.

Compatible with both transformers <4.49 (returns tensors) and
transformers >=4.49 (returns BaseModelOutputWithPooling wrappers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import torch
from PIL import Image


@dataclass
class EntityScore:
    """One entity with its grounding score."""
    entity: str
    score: float


def _to_tensor(out) -> torch.Tensor:
    """
    Normalize the return value of CLIP's get_image_features /
    get_text_features so we always work with a 2-D tensor.

    transformers <4.49: returns Tensor[B, D] directly.
    transformers >=4.49: returns BaseModelOutputWithPooling with
                         .pooler_output (preferred) or .last_hidden_state.
    """
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        # Mean-pool over sequence dimension as a fallback.
        return out.last_hidden_state.mean(dim=1)
    raise TypeError(f"Unrecognized CLIP output type: {type(out)}")


class CLIPGroundingScorer:
    """Score how well each entity phrase is grounded in an image."""

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str = None,
        prompt_template: str = "a photo of {entity}",
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.prompt_template = prompt_template

        print("Loading CLIP scorer (this downloads ~600MB on first run)...")
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        print(f"  device = {device}")

    @torch.no_grad()
    def score_entities(
        self,
        image: Union[Image.Image, str],
        entities: List[str],
    ) -> List[EntityScore]:
        if not entities:
            return []

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        prompts = [self.prompt_template.format(entity=e) for e in entities]

        # Image features.
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = _to_tensor(self.model.get_image_features(**img_inputs))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Text features.
        txt_inputs = self.processor(
            text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        text_features = _to_tensor(self.model.get_text_features(**txt_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity between the single image vec and each text vec.
        sims = (image_features @ text_features.T).squeeze(0).cpu().tolist()
        if isinstance(sims, float):
            sims = [sims]

        return [EntityScore(entity=e, score=float(s)) for e, s in zip(entities, sims)]
