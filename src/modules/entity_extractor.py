"""
Entity extraction from VLM-generated text.

Uses spaCy noun-chunk detection to extract candidate object entities
(noun phrases) from a caption or answer. Filters out determiners,
pronouns, and a stoplist of abstract / meta / locative nouns that are
not visually depictable (e.g., "image", "scene", "front", "kind").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import spacy
from spacy.language import Language


_LEADING_DETERMINERS = {
    "a", "an", "the",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no",
}

# Pronouns we never want as entities.
_PRONOUN_POS = {"PRON"}

# Abstract / meta / locative / quantitative nouns that pass the noun-chunk
# filter but are not visually depictable. CLIP gives them garbage scores.
_NON_VISUAL_STOPLIST = {
    # Meta-nouns about the image itself
    "image", "picture", "photo", "photograph", "scene", "shot", "frame",
    "view", "background", "foreground", "setting",
    # Locative / positional
    "front", "back", "side", "top", "bottom", "middle", "center",
    "left", "right", "edge", "corner", "above", "below",
    # Abstract / generic
    "kind", "type", "way", "thing", "something", "anything", "nothing",
    "part", "piece", "area", "place", "position", "direction",
    # Quantifiers / mixers
    "mix", "lot", "lots", "number", "amount", "variety", "selection",
    "group", "pair", "couple",
    # Distance / temporal
    "distance", "time", "moment", "winter", "summer", "spring", "autumn",
    # Actions-as-nouns that slip through
    "appearance", "presence", "look",
}

_MIN_LEN = 2


@dataclass
class Entity:
    text: str
    raw: str
    start_char: int
    end_char: int


class EntityExtractor:
    """Extract noun-phrase entities from free-form text."""

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        nlp: Optional[Language] = None,
        stoplist: Optional[set] = None,
    ) -> None:
        self.spacy_model = spacy_model
        self.nlp = nlp if nlp is not None else spacy.load(spacy_model)
        self.stoplist = stoplist if stoplist is not None else _NON_VISUAL_STOPLIST

    def extract(self, text: str) -> List[Entity]:
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        seen_lower: set[str] = set()
        entities: List[Entity] = []

        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in _PRONOUN_POS:
                continue

            cleaned = self._strip_leading_determiners(chunk.text).strip()
            if len(cleaned) < _MIN_LEN:
                continue

            # Drop the chunk if its head noun (or the whole cleaned phrase)
            # is in our non-visual stoplist.
            head = chunk.root.lemma_.lower()
            if head in self.stoplist or cleaned.lower() in self.stoplist:
                continue

            key = cleaned.lower()
            if key in seen_lower:
                continue
            seen_lower.add(key)

            entities.append(
                Entity(
                    text=cleaned,
                    raw=chunk.text,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                )
            )

        return entities

    @staticmethod
    def _strip_leading_determiners(phrase: str) -> str:
        tokens = phrase.split()
        if tokens and tokens[0].lower() in _LEADING_DETERMINERS:
            tokens = tokens[1:]
        return " ".join(tokens)
