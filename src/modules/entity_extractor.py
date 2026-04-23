"""
Entity extraction from VLM-generated text.

Uses spaCy noun-chunk detection to extract candidate object entities
(noun phrases) from a caption or answer. These entities are the units
we score with CLIP and, in our full pipeline, the units we selectively
regenerate when flagged as hallucinated.

We deliberately keep this module simple and syntactic: any downstream
re-scoring (CLIP grounding, threshold-based detection) happens elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import spacy
from spacy.language import Language


# Determiners to strip from the front of a noun chunk, e.g. "a red car" -> "red car".
# We keep the rest of the chunk (adjectives, compound nouns) intact.
_LEADING_DETERMINERS = {
    "a", "an", "the",
    "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no",
}

# Pronouns we never want as entities.
_PRONOUN_POS = {"PRON"}

# Minimum character length after cleanup.
_MIN_LEN = 2


@dataclass
class Entity:
    """A candidate entity extracted from text."""
    text: str           # Cleaned phrase, e.g. "red car"
    raw: str            # Original noun chunk, e.g. "a red car"
    start_char: int     # Character offset in the source text
    end_char: int


class EntityExtractor:
    """
    Extracts noun-phrase entities from free-form text.

    Example:
        extractor = EntityExtractor()
        entities = extractor.extract(
            "A man is riding a bicycle next to a yellow giraffe."
        )
        for e in entities:
            print(e.text)
        # -> man
        # -> bicycle
        # -> yellow giraffe
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        nlp: Optional[Language] = None,
    ) -> None:
        self.spacy_model = spacy_model
        self.nlp = nlp if nlp is not None else spacy.load(spacy_model)

    def extract(self, text: str) -> List[Entity]:
        """Extract cleaned noun-phrase entities from a caption or answer."""
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
        seen_lower: set[str] = set()
        entities: List[Entity] = []

        for chunk in doc.noun_chunks:
            # Drop pronoun-headed chunks ("he", "it", "they riding a bike").
            if chunk.root.pos_ in _PRONOUN_POS:
                continue

            cleaned = self._strip_leading_determiners(chunk.text)
            cleaned = cleaned.strip()

            if len(cleaned) < _MIN_LEN:
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
        """Remove a single leading determiner if present."""
        tokens = phrase.split()
        if tokens and tokens[0].lower() in _LEADING_DETERMINERS:
            tokens = tokens[1:]
        return " ".join(tokens)
