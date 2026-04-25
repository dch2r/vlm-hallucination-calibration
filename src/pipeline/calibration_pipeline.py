"""End-to-end calibration pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from PIL import Image

from src.modules.hallucination_detector import (
    DetectionResult,
    HallucinationDetector,
    ThresholdPolicy,
)
from src.modules.vlm_backbone import VLMBackbone
from src.pipeline.selective_regenerator import SelectiveRegenerator


@dataclass
class RoundResult:
    round_index: int
    prompt: str
    response: str
    detection: DetectionResult
    num_hallucinated: int
    num_grounded: int


@dataclass
class CalibrationResult:
    image_id: Optional[str]
    initial_prompt: str
    rounds: List[RoundResult] = field(default_factory=list)
    converged_at: Optional[int] = None
    max_rounds: int = 2

    @property
    def final_response(self) -> str:
        return self.rounds[-1].response if self.rounds else ""

    @property
    def initial_response(self) -> str:
        return self.rounds[0].response if self.rounds else ""

    def hallucination_rate_per_round(self) -> List[float]:
        rates = []
        for r in self.rounds:
            n = len(r.detection.verdicts)
            rates.append(r.num_hallucinated / n if n > 0 else 0.0)
        return rates


class CalibrationPipeline:
    """Run the full hallucination calibration loop end-to-end."""

    def __init__(
        self,
        vlm: VLMBackbone,
        detector: HallucinationDetector,
        regenerator: Optional[SelectiveRegenerator] = None,
        max_rounds: int = 2,
        policy: ThresholdPolicy = "percentile",
        tau: float = 0.22,
        percentile: float = 30.0,
        max_new_tokens: int = 128,
        reprompt_temperature: float = 0.4,
    ) -> None:
        self.vlm = vlm
        self.detector = detector
        self.regenerator = regenerator if regenerator is not None else SelectiveRegenerator()
        self.max_rounds = max_rounds
        self.policy = policy
        self.tau = tau
        self.percentile = percentile
        self.max_new_tokens = max_new_tokens
        # Round 0 is greedy (deterministic baseline). Re-prompts use a
        # mild sampling temperature so the model doesn't greedy-copy
        # the previous response that's quoted in the corrective prompt.
        self.reprompt_temperature = reprompt_temperature

    def run(
        self,
        image: Union[Image.Image, str],
        initial_prompt: str = "Describe this image in one sentence.",
        image_id: Optional[str] = None,
    ) -> CalibrationResult:
        result = CalibrationResult(
            image_id=image_id,
            initial_prompt=initial_prompt,
            max_rounds=self.max_rounds,
        )

        current_prompt = initial_prompt
        for k in range(self.max_rounds + 1):
            # Round 0 = greedy; subsequent rounds use sampling so re-prompt
            # actually produces a different response.
            temperature = 0.0 if k == 0 else self.reprompt_temperature

            vlm_response = self.vlm.generate(
                image=image,
                prompt=current_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
            )
            response_text = vlm_response.text

            detection = self.detector.detect(
                image=image,
                caption=response_text,
                policy=self.policy,
                tau=self.tau,
                percentile=self.percentile,
            )

            num_hallu = len(detection.hallucinated)
            num_ok = len(detection.grounded)

            result.rounds.append(
                RoundResult(
                    round_index=k,
                    prompt=current_prompt,
                    response=response_text,
                    detection=detection,
                    num_hallucinated=num_hallu,
                    num_grounded=num_ok,
                )
            )

            if num_hallu == 0:
                if result.converged_at is None:
                    result.converged_at = k
                break

            if k < self.max_rounds:
                instruction = self.regenerator.build(
                    previous_response=response_text,
                    verdicts=detection.verdicts,
                )
                current_prompt = instruction.text

        return result
