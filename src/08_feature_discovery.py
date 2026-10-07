from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# --------------------------------------------------
# Load representations
# --------------------------------------------------
data = np.load(RESULTS_DIR / "dbn_representations.npz")

H1 = data["H1_train"]
H2 = data["H2_train"]
H3 = data["H3_train"]
y = data["y_train"]


# --------------------------------------------------
# Calculate average activation for each digit
# --------------------------------------------------
def class_mean_activation(features, labels):
    result = np.zeros((10, features.shape[1]))

    for digit in range(10):
        result[digit] = features[labels == digit].mean(axis=0)

    return result


H1_means = class_mean_activation(H1, y)
H2_means = class_mean_activation(H2, y)
H3_means = class_mean_activation(H3, y)


# --------------------------------------------------
# Plot feature-discovery heatmaps
# --------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 12))


plots = [
    (H1_means, "DBN Layer 1 - Class-wise Feature Activations"),
    (H2_means, "DBN Layer 2 - Class-wise Feature Activations"),
    (H3_means, "DBN Layer 3 - Class-wise Feature Activations"),
]


for ax, (matrix, title) in zip(axes, plots):

    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="viridis",
    )

    ax.set_title(title)
    ax.set_xlabel("Hidden Feature")
    ax.set_ylabel("Digit Class")

    ax.set_yticks(range(10))
    ax.set_yticklabels(range(10))

    ax.set_xticks(range(matrix.shape[1]))

    fig.colorbar(image, ax=ax, label="Mean Activation")


fig.suptitle(
    "Layer-wise Feature Discovery in the Deep Belief Network",
    fontsize=15,
)

plt.tight_layout(rect=[0, 0, 1, 0.97])


output_file = RESULTS_DIR / "layerwise_feature_discovery.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Print summary
# --------------------------------------------------
print("Feature discovery visualization completed.")

print("Layer 1 features:", H1.shape[1])
print("Layer 2 features:", H2.shape[1])
print("Layer 3 features:", H3.shape[1])

print("\nSaved to:")
print(output_file)