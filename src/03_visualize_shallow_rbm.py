from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load original dataset
# --------------------------------------------------
digits = load_digits()

X = digits.data.astype(np.float32) / 16.0
y = digits.target


# --------------------------------------------------
# Load the trained shallow RBM
# --------------------------------------------------
representation_file = RESULTS_DIR / "shallow_rbm_representation.npz"

data = np.load(representation_file)

H = data["H_train"]
y_train = data["y_train"]


# --------------------------------------------------
# 1. Visualize learned RBM components
# --------------------------------------------------
# Components shape: (32, 64)
# Convert each component back to 8x8 image form.
#
# We reload the RBM by training it again because this
# experiment file needs its learned components.
# --------------------------------------------------
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split


X_train, _, _, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


rbm = BernoulliRBM(
    n_components=32,
    learning_rate=0.05,
    batch_size=32,
    n_iter=30,
    random_state=42,
)

rbm.fit(X_train)


fig, axes = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axes.ravel()):
    ax.imshow(
        rbm.components_[i].reshape(8, 8),
        cmap="gray"
    )
    ax.set_title(f"Feature {i + 1}")
    ax.axis("off")

plt.suptitle("Shallow RBM Learned Features")
plt.tight_layout()

feature_path = RESULTS_DIR / "shallow_rbm_features.png"
plt.savefig(feature_path, dpi=200)
plt.close()

print(f"Saved: {feature_path}")


# --------------------------------------------------
# 2. Reduce hidden representation to 2D using PCA
# --------------------------------------------------
pca = PCA(n_components=2, random_state=42)

H_2d = pca.fit_transform(H)


plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    H_2d[:, 0],
    H_2d[:, 1],
    c=y_train,
    cmap="tab10",
    s=12,
    alpha=0.7,
)

plt.title("Shallow RBM Hidden Representation - PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Digit")

plt.tight_layout()

pca_path = RESULTS_DIR / "shallow_rbm_pca.png"
plt.savefig(pca_path, dpi=200)
plt.close()

print(f"Saved: {pca_path}")

print("\nShallow representation visualization completed.")