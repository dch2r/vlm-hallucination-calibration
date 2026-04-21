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
   ΔCHAIR_k metric to measure calibration gain at each re-prompting
   round, addressing a gap in prior work.
3. **Black-box compatible** — treats the VLM as an inference API;
   no retraining, no logit access.

## Architecture
src/
modules/          Core building blocks (CLIP, entity extraction, etc.)
pipeline/         End-to-end calibration pipeline
experiments/        Evaluation scripts
scripts/            Utility / data-prep scripts
tests/              Unit tests for each module
results/            Generated results (gitignored)
notebooks/          Exploration notebooks
## Status

Actively under development. See `experiments/` for current evaluation runs.
