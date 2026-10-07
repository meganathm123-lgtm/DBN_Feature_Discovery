from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------
digits = load_digits()

X = digits.data.astype(np.float32) / 16.0
y = digits.target

X_train, _, y_train, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# Recreate the shallow RBM
# --------------------------------------------------
shallow_rbm = BernoulliRBM(
    n_components=32,
    learning_rate=0.05,
    batch_size=32,
    n_iter=30,
    random_state=42,
)

shallow_rbm.fit(X_train)

shallow = shallow_rbm.transform(X_train)


# --------------------------------------------------
# Load DBN representations
# --------------------------------------------------
dbn_file = RESULTS_DIR / "dbn_representations.npz"
dbn = np.load(dbn_file)

H1 = dbn["H1_train"]
H2 = dbn["H2_train"]
H3 = dbn["H3_train"]


# --------------------------------------------------
# PCA function
# --------------------------------------------------
def reduce_to_2d(features):
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(features)
    variance = pca.explained_variance_ratio_.sum()
    return reduced, variance


shallow_2d, shallow_var = reduce_to_2d(shallow)
H1_2d, H1_var = reduce_to_2d(H1)
H2_2d, H2_var = reduce_to_2d(H2)
H3_2d, H3_var = reduce_to_2d(H3)


# --------------------------------------------------
# Create comparison figure
# --------------------------------------------------
representations = [
    ("Shallow RBM", shallow_2d, shallow_var),
    ("DBN Layer 1", H1_2d, H1_var),
    ("DBN Layer 2", H2_2d, H2_var),
    ("DBN Layer 3", H3_2d, H3_var),
]


fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (title, representation, variance) in zip(
    axes.ravel(), representations
):
    scatter = ax.scatter(
        representation[:, 0],
        representation[:, 1],
        c=y_train,
        cmap="tab10",
        s=12,
        alpha=0.7,
    )

    ax.set_title(
        f"{title}\n"
        f"2D PCA variance: {variance:.2%}"
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

fig.colorbar(
    scatter,
    ax=axes.ravel().tolist(),
    label="Digit class",
)

fig.suptitle(
    "Comparison of Hidden Representations: "
    "Shallow RBM vs Deep Belief Network",
    fontsize=14,
)

plt.tight_layout()

output_file = RESULTS_DIR / "representation_comparison.png"

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Print representation sizes
# --------------------------------------------------
print("\nRepresentation comparison completed.")

print("Shallow RBM :", shallow.shape)
print("DBN Layer 1 :", H1.shape)
print("DBN Layer 2 :", H2.shape)
print("DBN Layer 3 :", H3.shape)

print("\nPCA variance:")
print(f"Shallow RBM : {shallow_var:.4f}")
print(f"DBN Layer 1 : {H1_var:.4f}")
print(f"DBN Layer 2 : {H2_var:.4f}")
print(f"DBN Layer 3 : {H3_var:.4f}")

print("\nSaved to:")
print(output_file)