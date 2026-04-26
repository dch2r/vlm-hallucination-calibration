"""
CHAIR (Caption Hallucination Assessment with Image Relevance) metric.

Reference:
    Rohrbach et al., "Object Hallucination in Image Captioning", EMNLP 2018.

Two metrics:
  CHAIR_i:  hallucinated_objects / total_mentioned_objects
            (fraction of mentions that are hallucinated)
  CHAIR_s:  hallucinated_captions / total_captions
            (fraction of captions that contain at least one hallucination)

Both are computed against MS COCO's 80 ground-truth object categories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set

import spacy


# Synonym map: phrases the model might say -> canonical COCO category name.
# Keep this conservative; the CHAIR paper uses a richer hand-curated map but
# this covers the common collisions seen in VLM outputs.
SYNONYMS = {
    # people
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "kid": "person", "guy": "person", "lady": "person",
    "people": "person", "men": "person", "women": "person", "children": "person",
    # vehicles
    "automobile": "car", "vehicle": "car", "taxi": "car",
    "motorbike": "motorcycle", "bike": "bicycle",
    "plane": "airplane", "jet": "airplane",
    # animals
    "puppy": "dog", "kitten": "cat", "calf": "cow", "bull": "cow",
    "horses": "horse", "dogs": "dog", "cats": "cat",
    # furniture
    "tv": "tv", "television": "tv", "monitor": "tv",
    "couch": "couch", "sofa": "couch",
    "dining table": "dining table", "table": "dining table",
    # food
    "donut": "donut", "doughnut": "donut",
    "hot dog": "hot dog", "hotdog": "hot dog",
    # misc
    "cellphone": "cell phone", "mobile phone": "cell phone", "phone": "cell phone",
    "remote control": "remote",
    "fridge": "refrigerator",
}


@dataclass
class CHAIRResult:
    """Per-caption CHAIR result + counts for aggregate computation."""
    mentioned_objects: Set[str]       # COCO objects extracted from caption
    hallucinated_objects: Set[str]    # mentioned but not in ground truth
    grounded_objects: Set[str]        # mentioned and in ground truth
    has_hallucination: bool           # True if any hallucinated object


@dataclass
class AggregateCHAIR:
    """Dataset-level CHAIR scores."""
    chair_i: float                    # mention-level rate
    chair_s: float                    # caption-level rate
    n_captions: int
    n_mentions: int
    n_hallucinations: int


class CHAIRMetric:
    """
    Compute CHAIR_i and CHAIR_s for VLM-generated captions vs COCO GT.

    Example:
        chair = CHAIRMetric(coco_categories=loader.categories)
        result = chair.score_one(caption="A man rides a bicycle past a bus stop.",
                                 gt_objects={"person", "bicycle"})
        print(result.hallucinated_objects)  # -> {"bus stop"} if "bus stop" in COCO
    """

    def __init__(
        self,
        coco_categories: Iterable[str],
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.coco_categories: Set[str] = {c.lower() for c in coco_categories}
        self.nlp = spacy.load(spacy_model)

        # Precompute a lookup: any phrase (lemmatized) -> canonical category.
        # This handles "men" -> "person" via SYNONYMS *and* lemmatization
        # ("dogs" -> "dog") which is already a COCO name.
        self._phrase_to_category = {}
        for cat in self.coco_categories:
            self._phrase_to_category[cat] = cat
        for syn, cat in SYNONYMS.items():
            if cat.lower() in self.coco_categories:
                self._phrase_to_category[syn.lower()] = cat.lower()

    def _extract_coco_objects(self, caption: str) -> Set[str]:
        """Return the set of COCO categories mentioned in the caption."""
        if not caption.strip():
            return set()

        doc = self.nlp(caption.lower())
        mentioned: Set[str] = set()

        # Pass 1: noun chunks (handles multi-word categories like "dining table").
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            # Strip leading determiners.
            tokens = text.split()
            if tokens and tokens[0] in {"a", "an", "the", "this", "that"}:
                tokens = tokens[1:]
            cleaned = " ".join(tokens)
            if cleaned in self._phrase_to_category:
                mentioned.add(self._phrase_to_category[cleaned])

        # Pass 2: individual lemmatized tokens (catches "dogs" -> "dog").
        for tok in doc:
            if tok.pos_ not in {"NOUN", "PROPN"}:
                continue
            lemma = tok.lemma_.lower()
            if lemma in self._phrase_to_category:
                mentioned.add(self._phrase_to_category[lemma])

        return mentioned

    def score_one(self, caption: str, gt_objects: Set[str]) -> CHAIRResult:
        """Score a single (caption, gt_objects) pair."""
        gt = {o.lower() for o in gt_objects}
        mentioned = self._extract_coco_objects(caption)
        hallucinated = mentioned - gt
        grounded = mentioned & gt
        return CHAIRResult(
            mentioned_objects=mentioned,
            hallucinated_objects=hallucinated,
            grounded_objects=grounded,
            has_hallucination=len(hallucinated) > 0,
        )

    def score_dataset(self, results: List[CHAIRResult]) -> AggregateCHAIR:
        """Aggregate per-caption results into dataset-level CHAIR_i, CHAIR_s."""
        n_caps = len(results)
        if n_caps == 0:
            return AggregateCHAIR(0.0, 0.0, 0, 0, 0)

        n_mentions = sum(len(r.mentioned_objects) for r in results)
        n_hallu = sum(len(r.hallucinated_objects) for r in results)
        n_caps_with_hallu = sum(1 for r in results if r.has_hallucination)

        return AggregateCHAIR(
            chair_i=n_hallu / n_mentions if n_mentions > 0 else 0.0,
            chair_s=n_caps_with_hallu / n_caps,
            n_captions=n_caps,
            n_mentions=n_mentions,
            n_hallucinations=n_hallu,
        )
