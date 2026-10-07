from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# Load representations
shallow_data = np.load(
    RESULTS_DIR / "shallow_rbm_representation.npz"
)

dbn_data = np.load(
    RESULTS_DIR / "dbn_representations.npz"
)


representations = {
    "Shallow RBM": shallow_data["H_train"],
    "DBN Layer 1": dbn_data["H1_train"],
    "DBN Layer 2": dbn_data["H2_train"],
    "DBN Layer 3": dbn_data["H3_train"],
}


results = []


for name, features in representations.items():

    # Variation of each hidden feature across samples
    feature_std = np.std(features, axis=0)

    mean_std = np.mean(feature_std)
    max_std = np.max(feature_std)

    # Count features showing meaningful variation
    informative_features = np.sum(feature_std > 0.05)

    # Overall representation variance
    overall_variance = np.var(features)

    results.append({
        "Representation": name,
        "Number of Features": features.shape[1],
        "Mean Feature Std": mean_std,
        "Maximum Feature Std": max_std,
        "Informative Features": informative_features,
        "Overall Variance": overall_variance,
    })


df = pd.DataFrame(results)


print("\n==============================================")
print("FEATURE DIVERSITY ANALYSIS")
print("==============================================")

print(df.to_string(index=False))


output_file = RESULTS_DIR / "feature_diversity_metrics.csv"

df.to_csv(
    output_file,
    index=False,
)

print("\nSaved to:")
print(output_file)