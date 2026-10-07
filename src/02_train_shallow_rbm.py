from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM


# --------------------------------------------------
# 1. Load the existing dataset
# --------------------------------------------------
digits = load_digits()

X = digits.data.astype(np.float32)
y = digits.target

# Convert pixel values from 0-16 to 0-1
X = X / 16.0


# --------------------------------------------------
# 2. Split the data
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 3. Create the shallow RBM
# --------------------------------------------------
rbm = BernoulliRBM(
    n_components=32,
    learning_rate=0.05,
    batch_size=32,
    n_iter=30,
    random_state=42,
    verbose=True,
)


# --------------------------------------------------
# 4. Train the RBM
# --------------------------------------------------
print("\nTraining shallow Bernoulli RBM...")
rbm.fit(X_train)


# --------------------------------------------------
# 5. Extract hidden representations
# --------------------------------------------------
H_train = rbm.transform(X_train)
H_test = rbm.transform(X_test)


# --------------------------------------------------
# 6. Save the learned representation
# --------------------------------------------------
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

np.savez(
    results_dir / "shallow_rbm_representation.npz",
    H_train=H_train,
    H_test=H_test,
    y_train=y_train,
    y_test=y_test,
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------
print("\nShallow RBM training completed.")
print("Input dimension           :", X_train.shape[1])
print("Hidden representation size:", H_train.shape[1])
print("Training representation   :", H_train.shape)
print("Testing representation    :", H_test.shape)

print("\nFirst sample hidden representation:")
print(H_train[0])

print("\nSaved to:")
print(results_dir / "shallow_rbm_representation.npz")