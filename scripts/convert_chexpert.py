"""
Materializes the CheXpert-v1.0-small folder structure notebooks/02_cbm_pipeline.ipynb
expects, from the Hugging Face dataset mirror `danjacobellis/chexpert`.

Why this exists: that HF dataset stores images embedded in parquet and encodes
the 14 pathology labels as HF class_label integers (0=unlabeled, 1=uncertain,
2=absent, 3=present) instead of the original CheXpert convention (blank/-1/0/1).
Downloading it alone does not reproduce the on-disk layout the notebook reads —
this script does that conversion. Verified directly against the actual parquet
rows (not just the dataset's README):
  - The dataset's own "Path" column already contains the full relative path,
    e.g. "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg" —
    this exactly matches what notebooks/02_cbm_pipeline.ipynb's first cell
    prints when it reads the real Stanford CSV.
  - The notebook's Config cell sets IMG_ROOT = Path("CheXpert-v1.0-small") and
    RAW_CSV_PATH = "CheXpert-v1.0-small/CheXpert-v1.0-small/train.csv" — the
    doubled "CheXpert-v1.0-small/CheXpert-v1.0-small/..." nesting is real and
    intentional (an artifact of how the original zip was extracted), not a bug.
    This script reproduces that exact nesting so no notebook path needs to change.
  - The 14 pathology columns use HF class_label ints; mapped back here to
    1 (present), 0 (absent), -1 (uncertain), NaN (unlabeled/not mentioned).
  - Frontal/Lateral is mapped back to the literal strings "Frontal"/"Lateral",
    since the notebook does a direct string comparison on this column.

Requires: pip install datasets pyarrow pillow tqdm pandas
Usage (run from notebooks/, the same directory the notebook itself runs from):
    python ../scripts/convert_chexpert.py
"""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]

# Verified against the dataset's own class_label schema (its README.md) and spot-checked
# against actual parquet rows: No Finding=3 ("present") paired with Cardiomegaly=0
# ("unlabeled") on row 0, Cardiomegaly=1 ("uncertain") on row 1, etc.
LABEL_MAP = {0: float("nan"), 1: -1.0, 2: 0.0, 3: 1.0}  # unlabeled, uncertain, absent, present
FRONTAL_LATERAL_MAP = {0: "Frontal", 1: "Lateral"}
SEX_MAP = {0: "Male", 1: "Female"}
AP_PA_MAP = {0: "AP", 1: "PA", 2: ""}

# IMG_ROOT in the notebook — this script must be run from the same working directory
# the notebook runs from, so this relative path resolves the same way for both.
IMG_ROOT = Path("CheXpert-v1.0-small")


def main():
    print("Loading danjacobellis/chexpert (train split) — this streams ~11GB from Hugging Face...")
    ds = load_dataset("danjacobellis/chexpert", split="train")

    rows = []
    print(f"Writing {len(ds)} images under {IMG_ROOT}/ ...")
    for example in tqdm(ds, total=len(ds)):
        rel_path = example["Path"]  # already "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
        img_path = IMG_ROOT / rel_path  # -> CheXpert-v1.0-small/CheXpert-v1.0-small/train/... (intentional double nesting)
        img_path.parent.mkdir(parents=True, exist_ok=True)

        if not img_path.exists():
            example["image"].save(img_path)  # PIL.Image via the datasets Image feature

        row = {
            "Path": rel_path,
            "Sex": SEX_MAP.get(example["Sex"], example["Sex"]),
            "Age": example["Age"],
            "Frontal/Lateral": FRONTAL_LATERAL_MAP.get(example["Frontal/Lateral"], example["Frontal/Lateral"]),
            "AP/PA": AP_PA_MAP.get(example["AP/PA"], example["AP/PA"]),
        }
        for col in LABEL_COLS:
            row[col] = LABEL_MAP.get(example[col], example[col])
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = IMG_ROOT / "CheXpert-v1.0-small" / "train.csv"  # matches RAW_CSV_PATH in the notebook exactly
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(df)} rows)")
    print("Expected to match the notebook's own printed shape: Raw CSV: (223414, 19)")


if __name__ == "__main__":
    main()
