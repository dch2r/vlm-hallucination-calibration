#!/usr/bin/env bash
# Setup script for the VLM hallucination calibration project.
# Usage: bash scripts/setup.sh
#
# Assumes Python >= 3.10 is available as `python3` or as a module
# (on NOTS, you may need `module load python/3.11` first).

set -e  # Exit on any error

echo "=== Python version ==="
python3 --version

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt

echo "=== Downloading spaCy English model ==="
python -m spacy download en_core_web_sm

echo "=== Verifying install ==="
python -c "
import torch, transformers, spacy, numpy as np
print('numpy:', np.__version__)
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)
print('spacy:', spacy.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
nlp = spacy.load('en_core_web_sm')
print('spaCy model loaded OK')
"

echo ""
echo "=== Setup complete ==="
echo "To activate the environment in future sessions, run:"
echo "  source venv/bin/activate"
