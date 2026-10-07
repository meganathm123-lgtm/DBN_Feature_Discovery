from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

metrics_file = RESULTS_DIR / "representation_metrics.csv"
df = pd.read_csv(metrics_file)


# ---------------------------------------------
# Accuracy comparison
# ---------------------------------------------
plt.figure(figsize=(9, 6))

plt.bar(
    df["Representation"],
    df["Test Accuracy"] * 100,
)

plt.ylabel("Test Accuracy (%)")
plt.xlabel("Representation")
plt.title("Classification Performance of Learned Representations")
plt.ylim(0, 100)

plt.xticks(rotation=15)

plt.tight_layout()

accuracy_file = RESULTS_DIR / "accuracy_comparison.png"
plt.savefig(accuracy_file, dpi=200, bbox_inches="tight")
plt.close()


# ---------------------------------------------
# Silhouette comparison
# ---------------------------------------------
plt.figure(figsize=(9, 6))

plt.bar(
    df["Representation"],
    df["Silhouette Score"],
)

plt.ylabel("Silhouette Score")
plt.xlabel("Representation")
plt.title("Class Separability of Learned Representations")

plt.xticks(rotation=15)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=1,
)

plt.tight_layout()

silhouette_file = RESULTS_DIR / "silhouette_comparison.png"
plt.savefig(silhouette_file, dpi=200, bbox_inches="tight")
plt.close()


print("Metric plots created successfully.")
print(f"Saved: {accuracy_file}")
print(f"Saved: {silhouette_file}")