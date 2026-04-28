"""
CHAIR (Caption Hallucination Assessment with Image Relevance) metric.

Reference:
    Rohrbach et al., "Object Hallucination in Image Captioning", EMNLP 2018.

Two metrics:
  CHAIR_i:  hallucinated_objects / total_mentioned_objects
  CHAIR_s:  hallucinated_captions / total_captions

Both are computed against MS COCO's 80 ground-truth object categories.

Known limitations of this implementation (discussed in our paper):
  1. Substring matching: when a caption contains a compound noun whose
     constituent is a COCO category (e.g., "hot dog" -> "dog",
     "teddy bear" -> "bear", "passenger cars" -> "car"), the lemma-
     based matcher may produce false positive mentions.
  2. Color/fruit collision: "orange" is both a color and a COCO fruit
     category. Captions like "an orange cat" can produce false
     mentions of the fruit.
  3. COCO annotation gaps: some images in MS COCO val 2014 have
     unlabeled but visible objects (e.g., a laptop in the corner
     that COCO annotators didn't label). When a VLM correctly names
     such an object, CHAIR will flag it as a hallucination.

We do NOT try to fix these in the matcher itself — instead, we
quantify their impact via a manual audit reported in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Set

import spacy


# Synonym map: phrases the model might say -> canonical COCO category.
# Conservative: only direct linguistic equivalents and plurals.
SYNONYMS = {
    # ---- people ----
    "man": "person", "woman": "person", "boy": "person", "girl": "person",
    "child": "person", "kid": "person", "guy": "person", "lady": "person",
    "people": "person", "men": "person", "women": "person", "children": "person",
    "individual": "person", "individuals": "person",
    "person": "person", "persons": "person",
    # ---- vehicles ----
    "bike": "bicycle", "bicycle": "bicycle", "bicycles": "bicycle",
    "motorbike": "motorcycle", "motorcycle": "motorcycle", "motorcycles": "motorcycle",
    "plane": "airplane", "jet": "airplane", "airplane": "airplane", "airplanes": "airplane",
    "car": "car", "cars": "car",
    "bus": "bus", "buses": "bus",
    "truck": "truck", "trucks": "truck",
    "boat": "boat", "boats": "boat",
    "train": "train", "trains": "train",
    # ---- animals ----
    "puppy": "dog", "puppies": "dog", "dogs": "dog", "dog": "dog",
    "kitten": "cat", "kittens": "cat", "cats": "cat", "cat": "cat",
    "calf": "cow", "calves": "cow", "cows": "cow", "cow": "cow", "bull": "cow",
    "horses": "horse", "horse": "horse",
    "birds": "bird", "bird": "bird",
    "elephants": "elephant", "elephant": "elephant",
    "bears": "bear", "bear": "bear",
    "giraffes": "giraffe", "giraffe": "giraffe",
    "zebras": "zebra", "zebra": "zebra",
    "sheep": "sheep",
    # ---- electronics ----
    "tv": "tv", "television": "tv", "televisions": "tv",
    "cellphone": "cell phone", "cell phone": "cell phone",
    "cellphones": "cell phone", "cell phones": "cell phone",
    "remote control": "remote", "remote": "remote",
    "laptops": "laptop", "laptop": "laptop",
    "keyboards": "keyboard", "keyboard": "keyboard",
    # ---- furniture ----
    "couch": "couch", "couches": "couch", "sofa": "couch", "sofas": "couch",
    "dining table": "dining table",
    "chairs": "chair", "chair": "chair",
    "beds": "bed", "bed": "bed",
    # ---- food ----
    "donut": "donut", "doughnut": "donut", "donuts": "donut", "doughnuts": "donut",
    "hot dog": "hot dog", "hotdog": "hot dog", "hotdogs": "hot dog", "hot dogs": "hot dog",
    "pizzas": "pizza", "pizza": "pizza",
    "cakes": "cake", "cake": "cake",
    "sandwiches": "sandwich", "sandwich": "sandwich",
    "apples": "apple", "apple": "apple",
    "oranges": "orange", "orange": "orange",
    "bananas": "banana", "banana": "banana",
    "broccoli": "broccoli",
    "carrots": "carrot", "carrot": "carrot",
    # ---- containers / kitchen ----
    "fridge": "refrigerator", "refrigerators": "refrigerator", "refrigerator": "refrigerator",
    "bottles": "bottle", "bottle": "bottle",
    "cups": "cup", "cup": "cup",
    "wine glass": "wine glass", "wine glasses": "wine glass",
    "bowls": "bowl", "bowl": "bowl",
    "forks": "fork", "fork": "fork",
    "knives": "knife", "knife": "knife",
    "spoons": "spoon", "spoon": "spoon",
    # ---- accessories ----
    "handbags": "handbag", "handbag": "handbag", "purse": "handbag",
    "ties": "tie", "tie": "tie",
    "suitcases": "suitcase", "suitcase": "suitcase",
    "umbrellas": "umbrella", "umbrella": "umbrella",
    "backpacks": "backpack", "backpack": "backpack",
    # ---- sports ----
    "frisbees": "frisbee", "frisbee": "frisbee",
    "skis": "skis", "ski": "skis",
    "snowboards": "snowboard", "snowboard": "snowboard",
    "kites": "kite", "kite": "kite",
    "baseball bats": "baseball bat", "baseball bat": "baseball bat",
    "baseball gloves": "baseball glove", "baseball glove": "baseball glove",
    "skateboards": "skateboard", "skateboard": "skateboard",
    "surfboards": "surfboard", "surfboard": "surfboard",
    "tennis rackets": "tennis racket", "tennis racket": "tennis racket",
    # ---- outdoor ----
    "traffic lights": "traffic light", "traffic light": "traffic light",
    "fire hydrants": "fire hydrant", "fire hydrant": "fire hydrant",
    "stop signs": "stop sign", "stop sign": "stop sign",
    "parking meters": "parking meter", "parking meter": "parking meter",
    "benches": "bench", "bench": "bench",
    # ---- bathroom ----
    "toilets": "toilet", "toilet": "toilet",
    "sinks": "sink", "sink": "sink",
    # ---- misc ----
    "books": "book", "book": "book",
    "vases": "vase", "vase": "vase",
    "scissors": "scissors",
    "teddy bears": "teddy bear", "teddy bear": "teddy bear",
    "hair driers": "hair drier", "hair drier": "hair drier", "hairdryer": "hair drier",
    "toothbrushes": "toothbrush", "toothbrush": "toothbrush",
    "clocks": "clock", "clock": "clock",
    "potted plants": "potted plant", "potted plant": "potted plant", "houseplant": "potted plant",
}


@dataclass
class CHAIRResult:
    mentioned_objects: Set[str]
    hallucinated_objects: Set[str]
    grounded_objects: Set[str]
    has_hallucination: bool


@dataclass
class AggregateCHAIR:
    chair_i: float
    chair_s: float
    n_captions: int
    n_mentions: int
    n_hallucinations: int


class CHAIRMetric:
    def __init__(
        self,
        coco_categories: Iterable[str],
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.coco_categories: Set[str] = {c.lower() for c in coco_categories}
        self.nlp = spacy.load(spacy_model)

        self._phrase_to_category = {}
        for cat in self.coco_categories:
            self._phrase_to_category[cat] = cat
        for syn, cat in SYNONYMS.items():
            if cat.lower() in self.coco_categories:
                self._phrase_to_category[syn.lower()] = cat.lower()

    def _extract_coco_objects(self, caption: str) -> Set[str]:
        if not caption.strip():
            return set()

        doc = self.nlp(caption.lower())
        mentioned: Set[str] = set()

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            tokens = text.split()
            if tokens and tokens[0] in {"a", "an", "the", "this", "that"}:
                tokens = tokens[1:]
            cleaned = " ".join(tokens)
            if cleaned in self._phrase_to_category:
                mentioned.add(self._phrase_to_category[cleaned])

        for tok in doc:
            if tok.pos_ not in {"NOUN", "PROPN"}:
                continue
            lemma = tok.lemma_.lower()
            if lemma in self._phrase_to_category:
                mentioned.add(self._phrase_to_category[lemma])

        return mentioned

    def score_one(self, caption: str, gt_objects: Set[str]) -> CHAIRResult:
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
