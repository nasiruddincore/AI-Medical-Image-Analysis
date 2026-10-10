"""
visualize.py
Grid visualization of predictions vs actual labels.
Shows model confidence on multiple test images.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def plot_sample_predictions(model, test_generator, num_images=12,
                             save_path='outputs/sample_predictions.png'):
    """
    Show a grid of test images with:
    - Predicted label (NORMAL / PNEUMONIA)
    - True label
    - Confidence score
    - Green border = correct, Red border = wrong
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    class_names = {v: k for k, v in test_generator.class_indices.items()}

    test_generator.reset()
    images, labels = next(test_generator)
    images = images[:num_images]
    labels = labels[:num_images]

    probs   = model.predict(images, verbose=0).flatten()
    preds   = (probs >= 0.5).astype(int)

    cols = 4
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            ax.imshow(images[i])

            true_label = class_names[int(labels[i])]
            pred_label = class_names[int(preds[i])]
            correct    = true_label == pred_label

            border_color = 'green' if correct else 'red'
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(4)

            title_color = 'green' if correct else 'red'
            ax.set_title(
                f"Pred: {pred_label}\nTrue: {true_label}\n{probs[i]*100:.1f}%",
                color=title_color, fontsize=9, fontweight='bold'
            )
        ax.axis('off')

    plt.suptitle('Sample Predictions — Green=Correct, Red=Wrong', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Sample predictions saved: {save_path}")