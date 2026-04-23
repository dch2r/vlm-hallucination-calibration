"""
Hallucination detection via entity-level CLIP grounding verification.

Combines EntityExtractor (text -> noun phrases) with CLIPGroundingScorer
(image + phrases -> similarity scores) and applies a threshold policy
to flag individual entities as grounded or hallucinated.

Two threshold policies are supported:

  * ``absolute``  : flag entities with CLIP score below a fixed tau.
  * ``percentile``: flag the bottom P% of entities within THIS caption.

The percentile policy is self-calibrating per-caption and robust to the
narrow, image-dependent range of raw CLIP similarities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from PIL import Image

from .clip_scorer import CLIPGroundingScorer, EntityScore
from .entity_extractor import Entity, EntityExtractor


ThresholdPolicy = Literal["absolute", "percentile"]


@dataclass
class EntityVerdict:
    """One entity with its grounding score and hallucination verdict."""
    entity: str               # Cleaned phrase, e.g. "yellow giraffe"
    raw: str                  # Original noun chunk from the caption
    start_char: int
    end_char: int
    score: float              # CLIP cosine similarity (higher = more grounded)
    is_hallucination: bool


@dataclass
class DetectionResult:
    """Full detection output for a single (image, caption) pair."""
    caption: str
    verdicts: List[EntityVerdict]
    policy: ThresholdPolicy
    threshold_used: float     # Effective cutoff (absolute tau OR computed percentile)

    @property
    def hallucinated(self) -> List[EntityVerdict]:
        return [v for v in self.verdicts if v.is_hallucination]

    @property
    def grounded(self) -> List[EntityVerdict]:
        return [v for v in self.verdicts if not v.is_hallucination]


class HallucinationDetector:
    """
    End-to-end: (image, caption) -> per-entity grounding verdicts.

    Example:
        detector = HallucinationDetector()
        result = detector.detect(
            image=Image.open("kitchen.jpg").convert("RGB"),
            caption="A man stands near a refrigerator and a yellow giraffe.",
            policy="percentile",
            percentile=30.0,
        )
        for v in result.verdicts:
            tag = "HALLU" if v.is_hallucination else "OK"
            print(f"  [{tag}] {v.entity:20s} {v.score:+.4f}")
    """

    def __init__(
        self,
        extractor: Optional[EntityExtractor] = None,
        scorer: Optional[CLIPGroundingScorer] = None,
    ) -> None:
        self.extractor = extractor if extractor is not None else EntityExtractor()
        self.scorer = scorer if scorer is not None else CLIPGroundingScorer()

    def detect(
        self,
        image: Union[Image.Image, str],
        caption: str,
        policy: ThresholdPolicy = "percentile",
        tau: float = 0.22,
        percentile: float = 30.0,
    ) -> DetectionResult:
        """
        Run extraction -> scoring -> thresholding on a single caption.

        Args:
            image:      PIL Image (RGB) or path to an image file.
            caption:    Free-form text produced by a VLM.
            policy:     ``"absolute"`` or ``"percentile"``.
            tau:        Cutoff for absolute policy. Scores below tau are flagged.
            percentile: For percentile policy, flag the bottom ``percentile``%
                        of entities (by score) in this caption.

        Returns:
            DetectionResult with one EntityVerdict per extracted entity.
        """
        entities: List[Entity] = self.extractor.extract(caption)
        if not entities:
            return DetectionResult(
                caption=caption, verdicts=[], policy=policy, threshold_used=float("nan")
            )

        # Score every entity against the image in one CLIP forward pass.
        phrases = [e.text for e in entities]
        scored: List[EntityScore] = self.scorer.score_entities(image, phrases)

        # Pair entities with their scores (order is preserved by score_entities).
        pairs = list(zip(entities, scored))

        threshold_used = self._compute_threshold(
            scores=[s.score for s in scored],
            policy=policy,
            tau=tau,
            percentile=percentile,
        )

        verdicts = [
            EntityVerdict(
                entity=ent.text,
                raw=ent.raw,
                start_char=ent.start_char,
                end_char=ent.end_char,
                score=sc.score,
                is_hallucination=sc.score < threshold_used,
            )
            for ent, sc in pairs
        ]

        return DetectionResult(
            caption=caption,
            verdicts=verdicts,
            policy=policy,
            threshold_used=threshold_used,
        )

    @staticmethod
    def _compute_threshold(
        scores: List[float],
        policy: ThresholdPolicy,
        tau: float,
        percentile: float,
    ) -> float:
        """Resolve the effective cutoff score under the chosen policy."""
        if policy == "absolute":
            return tau

        if policy == "percentile":
            if not scores:
                return float("nan")
            if not 0.0 <= percentile <= 100.0:
                raise ValueError(
                    f"percentile must be in [0, 100], got {percentile}"
                )
            # Manual percentile (avoid numpy import for this tiny helper).
            sorted_scores = sorted(scores)
            # Index of the cutoff. A score STRICTLY below this is flagged.
            # percentile=30 -> flag bottom 30% -> cutoff at index ceil(0.3 * N).
            n = len(sorted_scores)
            k = max(1, int(round(percentile / 100.0 * n)))
            # Everything below sorted_scores[k] is flagged. To make the check
            # ``score < threshold_used`` flag exactly k entities, we set the
            # threshold just above sorted_scores[k - 1].
            cutoff_value = sorted_scores[k - 1]
            # Tiny epsilon so ``< threshold_used`` includes the cutoff value itself.
            return cutoff_value + 1e-9

        raise ValueError(f"Unknown policy: {policy!r}")
