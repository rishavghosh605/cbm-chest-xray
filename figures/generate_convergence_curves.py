"""
Regenerates figures/convergence_curves.png by parsing the per-epoch training
logs already saved in notebooks/02_cbm_pipeline.ipynb's cell outputs.

This is the direct visual evidence behind "The fix and resolution" in the
README: the two continuous-scaling variants were still climbing well past
the original 100-epoch budget, which is why the first pass under-trained
them relative to the binarized variant.

Usage (from the figures/ directory):
    python generate_convergence_curves.py
"""
import json
import re

import matplotlib.pyplot as plt

NOTEBOOK_PATH = "../notebooks/02_cbm_pipeline.ipynb"
EPOCH_LINE = re.compile(r"epoch\s+(\d+)\s+loss\s+[\d.]+\s+val AUROC\s+([\d.]+)")
VARIANT_LINE = re.compile(r"Training CBM probe — variant: (\w+)")
BASELINE_LINE = re.compile(r"Training baseline probe")


def extract_curves(notebook_path):
    nb = json.load(open(notebook_path))
    curves = {}
    current_variant = None
    for cell in nb["cells"]:
        if "VARIANT_TRAIN_CONFIG" not in "".join(cell.get("source", [])):
            continue
        for out in cell.get("outputs", []):
            text = "".join(out.get("text", []))
            variant_match = VARIANT_LINE.search(text)
            if variant_match:
                current_variant = variant_match.group(1)
                curves.setdefault(current_variant, ([], []))
            if BASELINE_LINE.search(text):
                current_variant = None  # baseline has its own epoch log; not part of this comparison
            epoch_match = EPOCH_LINE.search(text)
            if epoch_match and current_variant:
                epoch, auroc = int(epoch_match.group(1)), float(epoch_match.group(2))
                curves[current_variant][0].append(epoch)
                curves[current_variant][1].append(auroc)
    return curves


curves = extract_curves(NOTEBOOK_PATH)

label_map = {
    "binarized": "Binarized (K-means), 100-epoch budget",
    "minmax_per_image": "Min-max per-image, 2000-epoch budget",
    "minmax_per_concept": "Min-max per-concept, 2000-epoch budget",
}

fig, ax = plt.subplots(figsize=(8, 5))
for variant, (epochs, aurocs) in curves.items():
    ax.plot(epochs, aurocs, label=label_map.get(variant, variant), linewidth=2)

ax.axvline(100, color="gray", linestyle="--", linewidth=1, label="Original fixed epoch budget (100)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation AUROC")
ax.set_title("Continuous concept scores were still under-trained at epoch 100")
ax.legend(fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("convergence_curves.png", dpi=150)
print("Saved convergence_curves.png")
