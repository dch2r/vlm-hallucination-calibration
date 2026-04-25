"""Selective entity regeneration for hallucinated VLM outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.modules.hallucination_detector import EntityVerdict


# Imperative re-prompt: starts with a clear instruction, lists the bad
# entities, then asks for a NEW description rather than quoting the old one.
DEFAULT_REPROMPT_TEMPLATE = (
    "Look at the image carefully and write a short, factual caption. "
    "Important constraints:\n"
    "- DO NOT mention any of these unverified entities: {hallucinated}.\n"
    "- It is fine to mention these verified entities: {grounded}.\n"
    "- Only describe what you can clearly see.\n"
    "- Keep the caption to 1-2 sentences.\n\n"
    "Write the new caption now:"
)


@dataclass
class RePromptInstruction:
    text: str
    hallucinated_entities: List[str]
    grounded_entities: List[str]
    previous_response: str


class SelectiveRegenerator:
    """Builds corrective re-prompts that target only hallucinated entities."""

    def __init__(self, template: str = DEFAULT_REPROMPT_TEMPLATE) -> None:
        self.template = template

    def build(
        self,
        previous_response: str,
        verdicts: List[EntityVerdict],
    ) -> RePromptInstruction:
        hallucinated = [v.entity for v in verdicts if v.is_hallucination]
        grounded = [v.entity for v in verdicts if not v.is_hallucination]

        text = self.template.format(
            hallucinated=self._format_list(hallucinated),
            grounded=self._format_list(grounded),
        )

        return RePromptInstruction(
            text=text,
            hallucinated_entities=hallucinated,
            grounded_entities=grounded,
            previous_response=previous_response,
        )

    @staticmethod
    def _format_list(items: List[str]) -> str:
        if not items:
            return "(none)"
        if len(items) == 1:
            return f'"{items[0]}"'
        quoted = [f'"{x}"' for x in items]
        return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"
