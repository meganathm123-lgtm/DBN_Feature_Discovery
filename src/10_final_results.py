from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


# --------------------------------------------------
# Load classification results
# --------------------------------------------------
classification = pd.read_csv(
    RESULTS_DIR / "representation_metrics.csv"
)


# --------------------------------------------------
# Load feature-diversity results
# --------------------------------------------------
diversity = pd.read_csv(
    RESULTS_DIR / "feature_diversity_metrics.csv"
)


# --------------------------------------------------
# Make column names consistent
# --------------------------------------------------
diversity = diversity.rename(
    columns={
        "Number of Features": "Features"
    }
)


# --------------------------------------------------
# Combine both result tables
# --------------------------------------------------
final_results = classification.merge(
    diversity,
    on=["Representation", "Features"],
    how="inner",
)


# --------------------------------------------------
# Select final columns
# --------------------------------------------------
final_results = final_results[
    [
        "Representation",
        "Features",
        "Test Accuracy",
        "Silhouette Score",
        "Mean Feature Std",
        "Maximum Feature Std",
        "Informative Features",
        "Overall Variance",
    ]
]


# --------------------------------------------------
# Display final experimental results
# --------------------------------------------------
print("\n==============================================")
print("FINAL EXPERIMENTAL RESULTS")
print("==============================================")

print(
    final_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# Save final results
# --------------------------------------------------
output_file = RESULTS_DIR / "final_experimental_results.csv"

final_results.to_csv(
    output_file,
    index=False,
)


print("\nSaved to:")
print(output_file)