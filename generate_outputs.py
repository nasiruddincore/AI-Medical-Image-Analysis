
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/png")
OUT.mkdir(parents=True, exist_ok=True)


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT / filename}")


# 1. Project pipeline diagram
fig, ax = plt.subplots(figsize=(10, 2.8))
ax.axis("off")
steps = [
    "Medical Image", "Preprocessing", "AI Model",
    "Prediction", "Evaluation"
]
for i, step in enumerate(steps):
    x = 0.1 + i * 0.2
    ax.text(
        x, 0.55, step, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="black"),
        transform=ax.transAxes
    )
    if i < len(steps) - 1:
        ax.annotate(
            "", xy=(x + 0.1, 0.55), xytext=(x + 0.065, 0.55),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->")
        )
ax.set_title("AI Medical Image Analysis Pipeline")
save_figure("01_project_pipeline.png")


# 2. Sample medical image placeholder
# Replace this with a real public-dataset image for your report.
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(np.zeros((224, 224)), cmap="gray", vmin=0, vmax=1)
ax.set_title("Placeholder — replace with a dataset image")
ax.axis("off")
save_figure("02_sample_image_placeholder.png")


# 3. Image preprocessing illustration
sample = np.tile(np.linspace(0, 1, 224), (224, 1))
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample, cmap="gray")
axes[0].set_title("Example input")
axes[1].imshow(sample.astype(np.float32) / 255.0, cmap="gray")
axes[1].set_title("Illustrative normalization")
for ax in axes:
    ax.axis("off")
save_figure("03_preprocessing_demo.png")


# 4. Training accuracy template — NOT actual model results
epochs = np.arange(1, 6)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, np.zeros_like(epochs), marker="o")
ax.set_title("Training Accuracy — Add Actual Training History")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.grid(True)
save_figure("04_training_accuracy_template.png")


# 5. Training loss template — NOT actual model results
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(epochs, np.zeros_like(epochs), marker="o")
ax.set_title("Training Loss — Add Actual Training History")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(True)
save_figure("05_training_loss_template.png")


# 6. Confusion matrix template — NOT actual evaluation
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(np.zeros((2, 2)), cmap="Blues")
ax.set_xticks([0, 1], labels=["Normal", "Pneumonia"])
ax.set_yticks([0, 1], labels=["Normal", "Pneumonia"])
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title("Confusion Matrix — Replace with Evaluation Results")
for i in range(2):
    for j in range(2):
        ax.text(j, i, "—", ha="center", va="center")
save_figure("06_confusion_matrix_template.png")

print("\nPNG generation completed.")
print("Note: templates are illustrative, not trained model results.")
