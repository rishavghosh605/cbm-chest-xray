#!/usr/bin/env bash
# Downloads everything the notebooks need that isn't bundled in this repo:
# BiomedCLIP (via the Hugging Face CLI, warming the local HF cache) and the
# CheXpert-v1.0-small images/labels (via a Python conversion step — see
# scripts/convert_chexpert.py for why a plain download isn't enough).
#
# Run this from the notebooks/ directory — the same working directory the
# notebooks themselves run from, since all paths below are relative to it.
#
# Usage:
#   cd notebooks/
#   bash ../scripts/download_data.sh
set -euo pipefail

if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli not found. Install it first: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

echo "== Step 1: BiomedCLIP (single checkpoint, both towers bundled) =="
# open_clip's create_model_from_pretrained("hf-hub:...") would download this
# automatically on first use, but pulling it explicitly via the CLI first
# avoids first-run failures/timeouts inside the notebook and confirms the
# download succeeded before you start training.
huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

echo ""
echo "== Step 2: CheXpert-v1.0-small (images + labels) =="
if [ -f "CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv" ]; then
    echo "Already present at CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv — skipping."
else
    python ../scripts/convert_chexpert.py
fi

echo ""
echo "Done. Verify with:"
echo "  ls CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv"
