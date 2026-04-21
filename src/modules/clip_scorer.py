"""
CLIP-based cross-modal grounding scorer.

Computes the cosine similarity between a CLIP image embedding and a CLIP
text embedding for each candidate entity phrase. Higher scores indicate
that the entity is visually grounded in the image. We use these scores
to flag hallucinated entities at a configurable threshold.

This module is the first stage of our verification pipeline. It treats
the VLM as a black box and operates entirely on the generated text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@dataclass
class EntityScore:
    """Single entity with its CLIP grounding score."""
    entity: str
    score: float


class CLIPGroundingScorer:
    """
    Wraps a HuggingFace CLIP model and exposes a simple
    .score_entities(image, entities) interface.

    Example:
        scorer = CLIPGroundingScorer()
        results = scorer.score_entities(
            image=Image.open("photo.jpg").convert("RGB"),
            entities=["a man", "a bicycle", "a red car"],
        )
        for r in results:
            print(f"{r.entity:20s} -> {r.score:.4f}")
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        prompt_template: str = "a photo of {entity}",
    ) -> None:
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score_entities(
        self,
        image: Union[Image.Image, str],
        entities: Iterable[str],
    ) -> List[EntityScore]:
        """
        Compute a CLIP cosine-similarity grounding score for each entity.

        Args:
            image:    PIL Image (RGB) or path to an image file.
            entities: Iterable of short noun phrases (e.g. "a red car").

        Returns:
            List of EntityScore in the same order as the input entities.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        entity_list = list(entities)
        if not entity_list:
            return []

        prompts = [self.prompt_template.format(entity=e) for e in entity_list]

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get raw embeddings
        image_features = self.model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )
        text_features = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # Normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # image_features: [1, D]  text_features: [N, D]
        sims = (text_features @ image_features.T).squeeze(-1)  # [N]

        return [
            EntityScore(entity=ent, score=float(sim))
            for ent, sim in zip(entity_list, sims.tolist())
        ]
