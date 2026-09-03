"""
Regenerates figures/auroc_comparison.png from results/concept_scaling_comparison.csv.

Usage (from the figures/ directory):
    python generate_auroc_comparison.py
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/concept_scaling_comparison.csv")

label_map = {
    "baseline_no_bottleneck": "Baseline\n(no bottleneck)",
    "binarized": "Binarized\n(K-means)",
    "minmax_per_image": "Min-max\n(per-image)",
    "minmax_per_concept": "Min-max\n(per-concept)",
}
order = ["baseline_no_bottleneck", "binarized", "minmax_per_image", "minmax_per_concept"]
df = df.set_index("variant").loc[order].reset_index()

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar([label_map[v] for v in df["variant"]], df["test_macro_auroc"], color="#4C72B0", width=0.6)
ax.set_ylabel("Test macro AUROC")
ax.set_ylim(0, 0.8)
ax.set_title("Concept encoding vs. test AUROC (14-label CheXpert)")
for bar, val in zip(bars, df["test_macro_auroc"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("auroc_comparison.png", dpi=150)
print("Saved auroc_comparison.png")
