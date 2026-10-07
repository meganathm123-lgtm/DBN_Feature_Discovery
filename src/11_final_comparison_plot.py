from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

input_file = RESULTS_DIR / "final_experimental_results.csv"
output_file = RESULTS_DIR / "final_comparison_figure.png"


# --------------------------------------------------
# Load final experimental results
# --------------------------------------------------
df = pd.read_csv(input_file)

models = df["Representation"]


# --------------------------------------------------
# Create final comparison figure
# --------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))


# --------------------------------------------------
# 1. Classification accuracy
# --------------------------------------------------
axes[0].bar(
    models,
    df["Test Accuracy"] * 100,
)

axes[0].set_title("Classification Accuracy")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(0, 100)

axes[0].tick_params(axis="x", rotation=25)


# --------------------------------------------------
# 2. Silhouette score
# --------------------------------------------------
axes[1].bar(
    models,
    df["Silhouette Score"],
)

axes[1].set_title("Class Separability")
axes[1].set_ylabel("Silhouette Score")

axes[1].axhline(
    0,
    linestyle="--",
    linewidth=1,
)

axes[1].tick_params(axis="x", rotation=25)


# --------------------------------------------------
# 3. Feature diversity
# --------------------------------------------------
axes[2].bar(
    models,
    df["Mean Feature Std"],
)

axes[2].set_title("Feature Diversity")
axes[2].set_ylabel("Mean Feature Standard Deviation")

axes[2].tick_params(axis="x", rotation=25)


# --------------------------------------------------
# Main title
# --------------------------------------------------
fig.suptitle(
    "Shallow RBM vs Deep Belief Network: "
    "Representation Analysis",
    fontsize=16,
)

fig.tight_layout(rect=[0, 0, 1, 0.93])


# --------------------------------------------------
# Save figure
# --------------------------------------------------
fig.savefig(
    output_file,
    dpi=250,
    bbox_inches="tight",
)

plt.close(fig)

print("Final comparison figure created successfully.")
print(f"Saved to: {output_file}")