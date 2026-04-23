# Calibrating Generative Hallucinations in Vision-Language Models

**Authors:** Diwesh Chaurasia, Jeevan Balasubramaniam
**Institution:** Department of Computer Science, Rice University

A modular, training-free pipeline for detecting and calibrating object
hallucinations in Vision-Language Models (VLMs) through CLIP-based
cross-modal grounding verification and iterative re-prompting.

## Key Contributions

1. **Entity-level selective regeneration** — unlike prior work that
   regenerates full captions, we mask and replace only hallucinated
   phrases while preserving grounded content.
2. **Per-round diminishing returns analysis** — we introduce the
   delta-CHAIR_k metric to measure calibration gain at each
   re-prompting round, addressing a gap in prior work.
3. **Black-box compatible** — treats the VLM as an inference API;
   no retraining, no logit access.

## Architecture

    Image + Prompt -> VLM -> Response -> Entity Extraction -> CLIP Scoring
                             ^                                      |
                             |                                      v
                             +------ Selective Re-Prompt <-- Detection
                                        (up to K=2 rounds)

## Setup

Requires Python >= 3.10.

    git clone https://github.com/dch2r/vlm-hallucination-calibration.git
    cd vlm-hallucination-calibration
    bash scripts/setup.sh
    source venv/bin/activate

## Running Tests

Each module has a standalone smoke test you can run to verify the install:

    python tests/test_clip_scorer.py
    python tests/test_entity_extractor.py
    python tests/test_hallucination_detector.py

First run of CLIP test downloads ~600MB of model weights.

## Running on Rice NOTS (GPU)

On NOTS, request an interactive GPU session before running:

    srun --partition=gpu --gres=gpu:1 --time=01:00:00 --mem=32G --pty bash
    module load python/3.11
    cd ~/vlm-hallucination-calibration
    source venv/bin/activate
    python tests/test_clip_scorer.py

## Repository Structure

    src/
        modules/       Core building blocks (CLIP, entity extraction, detector)
        pipeline/      End-to-end calibration pipeline
    experiments/       Evaluation scripts
    scripts/           Setup and utility scripts
    tests/             Unit tests for each module
    results/           Generated results (gitignored)
    notebooks/         Exploration notebooks

