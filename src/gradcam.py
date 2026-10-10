"""
gradcam.py
Gradient-weighted Class Activation Mapping (Grad-CAM).

This is the MOST impressive output of the project.
It produces a heatmap overlaid on the X-ray showing EXACTLY
which regions the AI looked at to make its diagnosis.

In real hospitals, this is critical for doctor trust —
they need to know WHY the AI said pneumonia.
"""

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os


def make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1'):
    """
    Generate Grad-CAM heatmap for a single image.
    
    How it works:
    1. Pass image through model
    2. Get gradients of prediction w.r.t. last conv layer
    3. Pool gradients to get feature map weights
    4. Weighted sum of feature maps = heatmap
    5. Resize and overlay on original image
    """

    # Create a model that outputs both conv features and predictions
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # For binary classification, use the prediction directly
        loss = predictions[:, 0]

    # Gradients of prediction w.r.t. conv layer output
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients across spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature maps by pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_gradcam(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay heatmap on original image.
    Red/yellow = AI focused here, Blue = less important.
    """
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend: original + heatmap
    superimposed = cv2.addWeighted(img, 1 - alpha,
                                   heatmap_colored, alpha, 0)
    return img, heatmap_colored, superimposed


def visualize_gradcam(model, img_path, true_label, pred_label, pred_prob,
                       save_path='outputs/gradcam_output.png'):
    """
    Full Grad-CAM visualization for one image.
    Shows: Original | Heatmap | Overlay
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Preprocess image for model
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    # Generate heatmap
    heatmap = make_gradcam_heatmap(img_array, model)

    # Overlay
    original, heatmap_img, overlay = overlay_gradcam(img_path, heatmap)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original, cmap='gray' if original.ndim == 2 else None)
    axes[0].set_title('Original X-Ray', fontsize=13)
    axes[0].axis('off')

    axes[1].imshow(heatmap_img)
    axes[1].set_title('Grad-CAM Heatmap\n(Red = AI focused here)', fontsize=13)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    color = 'red' if pred_label == 'PNEUMONIA' else 'green'
    axes[2].set_title(
        f'Prediction: {pred_label} ({pred_prob*100:.1f}%)\nTrue: {true_label}',
        fontsize=13, color=color, fontweight='bold'
    )
    axes[2].axis('off')

    plt.suptitle('Grad-CAM: Where did the AI look?', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Grad-CAM saved: {save_path}")