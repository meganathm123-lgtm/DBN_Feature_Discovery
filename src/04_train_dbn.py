from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load the existing handwritten digits dataset
# --------------------------------------------------
digits = load_digits()

X = digits.data.astype(np.float32) / 16.0
y = digits.target


# --------------------------------------------------
# 3. Split the dataset
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 4. Define the DBN architecture
# --------------------------------------------------
# Input: 64
# Layer 1: 32
# Layer 2: 16
# Layer 3: 8
# --------------------------------------------------

rbm1 = BernoulliRBM(
    n_components=32,
    learning_rate=0.05,
    batch_size=32,
    n_iter=30,
    random_state=42,
    verbose=True,
)

rbm2 = BernoulliRBM(
    n_components=16,
    learning_rate=0.05,
    batch_size=32,
    n_iter=30,
    random_state=43,
    verbose=True,
)

rbm3 = BernoulliRBM(
    n_components=8,
    learning_rate=0.05,
    batch_size=32,
    n_iter=30,
    random_state=44,
    verbose=True,
)


# --------------------------------------------------
# 5. Train RBM Layer 1
# --------------------------------------------------
print("\n==============================")
print("Training DBN - Layer 1")
print("==============================")

rbm1.fit(X_train)

H1_train = rbm1.transform(X_train)
H1_test = rbm1.transform(X_test)

print("Layer 1 representation:", H1_train.shape)


# --------------------------------------------------
# 6. Train RBM Layer 2
# --------------------------------------------------
print("\n==============================")
print("Training DBN - Layer 2")
print("==============================")

rbm2.fit(H1_train)

H2_train = rbm2.transform(H1_train)
H2_test = rbm2.transform(H1_test)

print("Layer 2 representation:", H2_train.shape)


# --------------------------------------------------
# 7. Train RBM Layer 3
# --------------------------------------------------
print("\n==============================")
print("Training DBN - Layer 3")
print("==============================")

rbm3.fit(H2_train)

H3_train = rbm3.transform(H2_train)
H3_test = rbm3.transform(H2_test)

print("Layer 3 representation:", H3_train.shape)


# --------------------------------------------------
# 8. Save all hidden representations
# --------------------------------------------------
output_file = RESULTS_DIR / "dbn_representations.npz"

np.savez(
    output_file,
    H1_train=H1_train,
    H1_test=H1_test,
    H2_train=H2_train,
    H2_test=H2_test,
    H3_train=H3_train,
    H3_test=H3_test,
    y_train=y_train,
    y_test=y_test,
)


# --------------------------------------------------
# 9. Display final summary
# --------------------------------------------------
print("\n================================")
print("DBN training completed")
print("================================")

print("Input representation :", X_train.shape)
print("Layer 1 representation:", H1_train.shape)
print("Layer 2 representation:", H2_train.shape)
print("Layer 3 representation:", H3_train.shape)

print("\nSaved representations to:")
print(output_file)