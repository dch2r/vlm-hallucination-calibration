"""
End-to-end calibration pipeline.

Wraps a VLM backbone with the verification stack (entity extraction +
CLIP scoring + hallucination detection) and a selective re-prompting
loop. Records the full per-round trajectory so we can compute staged
metrics (round 0 vs round 1 vs round 2) for the final report.

Pipeline flow:

    image, prompt
        |
        v
    [round 0]  VLM.generate -> response_0
                              -> detect hallucinations
                              -> if none: STOP
                              -> else: build re-prompt
        |
        v
    [round 1]  VLM.generate(re-prompt) -> response_1
                                       -> detect hallucinations
                                       -> if none: STOP
                                       -> else: build re-prompt
        |
        v
    [round 2]  VLM.generate(re-prompt) -> response_2  (final)

The pipeline always returns a complete trajectory, including all
intermediate responses and per-round verdicts. This is what enables
staged evaluation across rounds 0, 1, 2.
"""

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
    """One round in the calibration trajectory."""
    round_index: int                       # 0 = initial, 1 = after first re-prompt, ...
    prompt: str                            # Prompt sent to the VLM this round
    response: str                          # VLM output this round
    detection: DetectionResult             # Per-entity verdicts on the response
    num_hallucinated: int
    num_grounded: int


@dataclass
class CalibrationResult:
    """Full trajectory for a single (image, prompt) calibration run."""
    image_id: Optional[str]
    initial_prompt: str
    rounds: List[RoundResult] = field(default_factory=list)
    converged_at: Optional[int] = None     # Round index where hallucinations hit zero, else None
    max_rounds: int = 2

    @property
    def final_response(self) -> str:
        return self.rounds[-1].response if self.rounds else ""

    @property
    def initial_response(self) -> str:
        return self.rounds[0].response if self.rounds else ""

    def hallucination_rate_per_round(self) -> List[float]:
        """Fraction of entities flagged as hallucinated, per round."""
        rates = []
        for r in self.rounds:
            n = len(r.detection.verdicts)
            rates.append(r.num_hallucinated / n if n > 0 else 0.0)
        return rates


class CalibrationPipeline:
    """
    Run the full hallucination calibration loop end-to-end.

    Args:
        vlm:          VLMBackbone instance (e.g. Qwen25VLBackbone).
        detector:     HallucinationDetector instance.
        regenerator:  SelectiveRegenerator instance.
        max_rounds:   Maximum number of re-prompting rounds (K). The
                      professor's feedback specifies measuring at K=0, 1, 2,
                      so the default is 2.
        policy:       Threshold policy for hallucination detection.
        tau:          Cutoff for ``absolute`` policy.
        percentile:   Cutoff for ``percentile`` policy.

    Example:
        pipeline = CalibrationPipeline(vlm, detector, regenerator)
        result = pipeline.run(image, "Describe this image.")
        for r in result.rounds:
            print(f"Round {r.round_index}: {r.response}")
            print(f"  hallucinated: {r.num_hallucinated}/{len(r.detection.verdicts)}")
    """

    def __init__(
        self,
        vlm: VLMBackbone,
        detector: HallucinationDetector,
        regenerator: Optional[SelectiveRegenerator] = None,
        max_rounds: int = 2,
        policy: ThresholdPolicy = "percentile",
        tau: float = 0.22,
        percentile: float = 30.0,
        max_new_tokens: int = 96,
    ) -> None:
        self.vlm = vlm
        self.detector = detector
        self.regenerator = regenerator if regenerator is not None else SelectiveRegenerator()
        self.max_rounds = max_rounds
        self.policy = policy
        self.tau = tau
        self.percentile = percentile
        self.max_new_tokens = max_new_tokens

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

        # Round 0 — raw VLM output.
        current_prompt = initial_prompt
        for k in range(self.max_rounds + 1):
            vlm_response = self.vlm.generate(
                image=image,
                prompt=current_prompt,
                max_new_tokens=self.max_new_tokens,
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

            # Convergence check: stop early if nothing was flagged.
            if num_hallu == 0:
                if result.converged_at is None:
                    result.converged_at = k
                break

            # Otherwise, build a re-prompt and continue (unless we've hit max).
            if k < self.max_rounds:
                instruction = self.regenerator.build(
                    previous_response=response_text,
                    verdicts=detection.verdicts,
                )
                current_prompt = instruction.text

        return result
