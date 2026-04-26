"""
Run the calibration pipeline over a sample of MS COCO val images,
compute CHAIR_i / CHAIR_s per round, and save results to JSON.

Usage:
    # Real run on Colab/NOTS GPU:
    python experiments/run_eval.py --num_samples 500 --output results/eval_main.json

    # Mock run on Mac CPU (no VLM/CLIP, hardcoded captions):
    python experiments/run_eval.py --num_samples 5 --mock --output results/eval_mock.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.coco_loader import COCOLoader, COCOSample
from src.metrics.chair import CHAIRMetric, CHAIRResult


# ---------- mock pipeline for CPU testing ----------

class MockPipeline:
    """
    Pretends to be a CalibrationPipeline. Returns hardcoded responses so we
    can wire up the eval harness on a CPU-only machine without loading Qwen.
    """

    def __init__(self):
        self.max_rounds = 2

    def run(self, image, initial_prompt: str = "", image_id=None):
        from src.pipeline.calibration_pipeline import CalibrationResult, RoundResult
        from src.modules.hallucination_detector import (
            DetectionResult, EntityVerdict,
        )

        # Three fake rounds with progressively cleaner captions.
        fake_responses = [
            f"A person stands near a bicycle next to a bus stop and a red car.",
            f"A person stands near a bicycle next to a bus stop.",
            f"A person stands near a bicycle.",
        ]
        rounds = []
        for k, resp in enumerate(fake_responses):
            verdicts = [
                EntityVerdict(entity="person", raw="A person", start_char=0, end_char=8,
                              score=0.28, is_hallucination=False),
                EntityVerdict(entity="bicycle", raw="a bicycle", start_char=0, end_char=8,
                              score=0.27, is_hallucination=False),
            ]
            if k == 0:
                verdicts.append(EntityVerdict(entity="bus stop", raw="a bus stop",
                                              start_char=0, end_char=8, score=0.18,
                                              is_hallucination=True))
                verdicts.append(EntityVerdict(entity="red car", raw="a red car",
                                              start_char=0, end_char=8, score=0.19,
                                              is_hallucination=True))
            elif k == 1:
                verdicts.append(EntityVerdict(entity="bus stop", raw="a bus stop",
                                              start_char=0, end_char=8, score=0.18,
                                              is_hallucination=True))
            n_h = sum(1 for v in verdicts if v.is_hallucination)
            n_g = sum(1 for v in verdicts if not v.is_hallucination)
            rounds.append(RoundResult(
                round_index=k,
                prompt=initial_prompt if k == 0 else "...mock re-prompt...",
                response=resp,
                detection=DetectionResult(
                    caption=resp, verdicts=verdicts,
                    policy="absolute", threshold_used=0.22,
                ),
                num_hallucinated=n_h,
                num_grounded=n_g,
            ))
        result = CalibrationResult(
            image_id=str(image_id) if image_id else "mock",
            initial_prompt=initial_prompt,
            rounds=rounds,
            converged_at=2,
            max_rounds=2,
        )
        return result


# ---------- main eval loop ----------

def build_real_pipeline():
    """Build the real Qwen + CLIP pipeline (only used in non-mock mode)."""
    from src.modules.hallucination_detector import HallucinationDetector
    from src.modules.vlm_backbone import Qwen25VLBackbone
    from src.pipeline.calibration_pipeline import CalibrationPipeline
    vlm = Qwen25VLBackbone(quantize_int4=True)
    detector = HallucinationDetector()
    return CalibrationPipeline(
        vlm=vlm, detector=detector,
        max_rounds=2, policy="absolute", tau=0.22, max_new_tokens=96,
    )


def evaluate(args) -> None:
    print(f"[eval] num_samples={args.num_samples}  mock={args.mock}  policy=absolute")

    pipeline = MockPipeline() if args.mock else build_real_pipeline()
    loader = COCOLoader()
    chair = CHAIRMetric(coco_categories=loader.categories)

    # Per-round CHAIR result accumulators.
    per_round_results: List[List[CHAIRResult]] = [[], [], []]
    per_image_records = []

    t_start = time.time()
    for i, sample in enumerate(loader.iter_samples(n=args.num_samples, seed=args.seed)):
        t0 = time.time()
        cal_result = pipeline.run(
            image=sample.image,
            initial_prompt=args.prompt,
            image_id=str(sample.image_id),
        )

        per_round_chair = []
        for k, r in enumerate(cal_result.rounds):
            cr = chair.score_one(r.response, sample.gt_objects)
            per_round_chair.append(cr)
            if k < 3:
                per_round_results[k].append(cr)

        # If pipeline converged early, fill remaining rounds with the final round's result.
        for k in range(len(cal_result.rounds), 3):
            per_round_results[k].append(per_round_chair[-1])

        record = {
            "image_id": sample.image_id,
            "gt_objects": sorted(sample.gt_objects),
            "rounds": [
                {
                    "round": r.round_index,
                    "response": r.response,
                    "mentioned_coco_objects": sorted(per_round_chair[k].mentioned_objects),
                    "hallucinated_coco_objects": sorted(per_round_chair[k].hallucinated_objects),
                    "has_hallucination": per_round_chair[k].has_hallucination,
                    "num_flagged_by_clip": r.num_hallucinated,
                }
                for k, r in enumerate(cal_result.rounds)
            ],
            "converged_at": cal_result.converged_at,
            "elapsed_sec": round(time.time() - t0, 2),
        }
        per_image_records.append(record)

        print(f"[{i+1}/{args.num_samples}] image_id={sample.image_id} "
              f"R0_hallu={record['rounds'][0]['hallucinated_coco_objects']} "
              f"final_hallu={record['rounds'][-1]['hallucinated_coco_objects']} "
              f"({record['elapsed_sec']}s)")

    elapsed = time.time() - t_start

    summary = {
        "num_samples": len(per_image_records),
        "wall_time_sec": round(elapsed, 1),
        "per_round": {},
    }
    for k in range(3):
        agg = chair.score_dataset(per_round_results[k])
        summary["per_round"][f"round_{k}"] = {
            "chair_i": round(agg.chair_i, 4),
            "chair_s": round(agg.chair_s, 4),
            "n_captions": agg.n_captions,
            "n_mentions": agg.n_mentions,
            "n_hallucinations": agg.n_hallucinations,
        }

    output = {
        "config": {
            "num_samples": args.num_samples,
            "mock": args.mock,
            "prompt": args.prompt,
            "seed": args.seed,
        },
        "summary": summary,
        "per_image": per_image_records,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Summary (n={summary['num_samples']}, time={elapsed:.1f}s) ===")
    for k, r in summary["per_round"].items():
        print(f"  {k}: CHAIR_i={r['chair_i']:.3f}  CHAIR_s={r['chair_s']:.3f}  "
              f"({r['n_hallucinations']} hallu / {r['n_mentions']} mentions)")
    print(f"\nResults written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mock", action="store_true",
                        help="Use mock VLM (CPU only, no GPU/weights needed)")
    parser.add_argument("--prompt", type=str,
                        default="Describe this image in one sentence.")
    parser.add_argument("--output", type=str, default="results/eval.json")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
