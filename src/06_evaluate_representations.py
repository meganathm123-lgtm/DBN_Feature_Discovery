from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

shallow_file = RESULTS_DIR / "shallow_rbm_representation.npz"
dbn_file = RESULTS_DIR / "dbn_representations.npz"


# --------------------------------------------------
# Load representations
# --------------------------------------------------
shallow_data = np.load(shallow_file)
dbn_data = np.load(dbn_file)

representations = {
    "Shallow RBM": (
        shallow_data["H_train"],
        shallow_data["H_test"],
        shallow_data["y_train"],
        shallow_data["y_test"],
    ),
    "DBN Layer 1": (
        dbn_data["H1_train"],
        dbn_data["H1_test"],
        dbn_data["y_train"],
        dbn_data["y_test"],
    ),
    "DBN Layer 2": (
        dbn_data["H2_train"],
        dbn_data["H2_test"],
        dbn_data["y_train"],
        dbn_data["y_test"],
    ),
    "DBN Layer 3": (
        dbn_data["H3_train"],
        dbn_data["H3_test"],
        dbn_data["y_train"],
        dbn_data["y_test"],
    ),
}


# --------------------------------------------------
# Evaluate each representation
# --------------------------------------------------
results = []

for name, (X_train, X_test, y_train, y_test) in representations.items():

    print(f"\nEvaluating: {name}")

    # Classification model
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
    )

    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Silhouette score on training representation
    silhouette = silhouette_score(
        X_train,
        y_train,
        metric="euclidean",
    )

    num_features = X_train.shape[1]

    results.append(
        {
            "Representation": name,
            "Features": num_features,
            "Test Accuracy": accuracy,
            "Silhouette Score": silhouette,
        }
    )

    print(f"Features       : {num_features}")
    print(f"Test Accuracy  : {accuracy:.4f}")
    print(f"Silhouette Score: {silhouette:.4f}")


# --------------------------------------------------
# Create comparison table
# --------------------------------------------------
results_df = pd.DataFrame(results)

print("\n==========================================")
print("REPRESENTATION COMPARISON")
print("==========================================")

print(results_df.to_string(index=False))


# --------------------------------------------------
# Save results
# --------------------------------------------------
output_file = RESULTS_DIR / "representation_metrics.csv"

results_df.to_csv(
    output_file,
    index=False,
)

print("\nResults saved to:")
print(output_file)