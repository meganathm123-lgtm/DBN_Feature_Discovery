from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


# Load the existing handwritten digits dataset
digits = load_digits()

X = digits.data
y = digits.target
images = digits.images


print("Dataset loaded successfully")
print("Number of samples :", X.shape[0])
print("Number of features:", X.shape[1])
print("Image shape       :", images.shape[1:])
print("Number of classes :", len(set(y)))
print("Classes           :", sorted(set(y)))


# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.ravel()):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(f"Digit: {y[i]}")
    ax.axis("off")

plt.tight_layout()

output_path = "results/dataset_samples.png"
plt.savefig(output_path, dpi=200)
plt.show()

print(f"Sample image grid saved to: {output_path}")