"""
Selective entity regeneration for hallucinated VLM outputs.

This module implements the core novelty of our pipeline. When the
hallucination detector flags a subset of entities in a generated caption
as ungrounded, we do NOT discard the entire response. Instead, we
construct a corrective re-prompt that:

  1. Lists the specific entities that were not visually verified.
  2. Asks the VLM to regenerate ONLY a faithful description, preserving
     the grounded entities.

This is conceptually different from prior work (MARINE, VCD, M3ID) which
regenerates the full caption from scratch and risks losing correctly
grounded content. By listing the hallucinated entities explicitly, we
give the VLM precise guidance about what to fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.modules.hallucination_detector import EntityVerdict


# Default corrective re-prompt template. ``{hallucinated}`` is a comma-
# separated list of phrases; ``{grounded}`` is similar for verified entities.
DEFAULT_REPROMPT_TEMPLATE = (
    "Your previous response described this image as: \"{previous}\".\n\n"
    "However, a visual verification step found that the following entities "
    "are NOT clearly visible in the image: {hallucinated}.\n\n"
    "The following entities WERE verified as visible: {grounded}.\n\n"
    "Please rewrite a faithful, concise description of the image. Keep the "
    "verified entities, do not mention the unverified ones, and do not "
    "introduce new objects unless you are certain they are present."
)


@dataclass
class RePromptInstruction:
    """Output of the regenerator: the re-prompt string and supporting state."""
    text: str                         # The corrective prompt to send to the VLM
    hallucinated_entities: List[str]
    grounded_entities: List[str]
    previous_response: str


class SelectiveRegenerator:
    """
    Builds corrective re-prompts that target only hallucinated entities.

    Example:
        regen = SelectiveRegenerator()
        instruction = regen.build(
            previous_response="A man rides a bicycle past a bus stop.",
            verdicts=detection_result.verdicts,
        )
        new_response = vlm.generate(image, instruction.text)
    """

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
            previous=previous_response.strip(),
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
