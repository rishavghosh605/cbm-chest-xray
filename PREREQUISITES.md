# Prerequisites

Steps to get from a clean checkout to a runnable `notebooks/02_cbm_pipeline.ipynb`. This repo ships code, curated concepts, and results — not the raw CheXpert images or the BiomedCLIP checkpoint, both of which are large and have their own licenses.

## 1. Install dependencies

```
pip install -r requirements.txt
pip install -U "huggingface_hub[cli]" datasets pyarrow
```

The second line is only for the data-download step below (`datasets`/`pyarrow` read the CheXpert parquet mirror; `huggingface_hub[cli]` gives you `huggingface-cli`) — it's not needed to run the notebooks themselves once the data is in place.

## 2. Download BiomedCLIP and CheXpert

```
cd notebooks/
bash ../scripts/download_data.sh
```

This runs two steps:

- **BiomedCLIP** — `huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`. This is a single Hugging Face repo containing one bundled checkpoint (`open_clip_pytorch_model.bin`) with both the vision and text towers already combined — `open_clip.create_model_from_pretrained("hf-hub:...")` (what both notebooks call) would download this automatically on first use, but running it via the CLI first warms the cache and surfaces network issues before you're mid-notebook.

  **Note on "two towers":** if you've previously also downloaded `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract` separately, that's a different, standalone repo (the underlying PubMedBERT text model on its own) — it is **not** referenced anywhere in either notebook and isn't required for this pipeline. The single bundled BiomedCLIP checkpoint above is sufficient.

- **CheXpert-v1.0-small** — the notebooks expect a specific on-disk layout (`CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv` plus matching image files — the doubled folder name is intentional, not a typo, and comes from how the original Stanford zip was extracted). Stanford's own CheXpert release is gated behind a signed-URL request form, not a plain download. `scripts/convert_chexpert.py` instead pulls the community Hugging Face mirror [`danjacobellis/chexpert`](https://huggingface.co/datasets/danjacobellis/chexpert) and converts it into that exact layout — see the script's docstring for the label-encoding remap this requires and how it was verified against the actual parquet rows.

  This step downloads ~11GB and writes ~223K individual JPEG files, so it takes a while and needs the disk space. `danjacobellis/chexpert` is a third-party mirror, not the official Stanford release — the image bytes are Hugging Face's own JPEG re-encoding, so treat this as functionally equivalent to the original CheXpert-v1.0-small, not guaranteed byte-identical. It's not authenticated/gated at the time of writing, so no Hugging Face login should be required for either download; run `huggingface-cli login` first only if you hit a rate limit.

## 3. Verify

```
ls notebooks/CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv
python3 -c "import pandas as pd; print(pd.read_csv('notebooks/CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv').shape)"
```

Expect `(223414, 19)` — the same shape `02_cbm_pipeline.ipynb`'s own first cell prints.

## 4. Run the notebooks

```
jupyter notebook notebooks/02_cbm_pipeline.ipynb
```

Run top to bottom once. Step 4 (image embedding) is the only cached step — see the notebook's own header cell for cache/re-run details.
