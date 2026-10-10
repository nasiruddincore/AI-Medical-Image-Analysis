"""
predict.py
Load saved model and predict on a single new X-ray image.
Simulates real clinical deployment.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os


def load_model(model_path='models/mobilenetv2_chest.h5'):
    """Load the saved trained model."""
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from: {model_path}")
    return model


def predict_single_image(model, img_path, threshold=0.5):
    """
    Predict whether a single chest X-ray shows pneumonia.
    
    Returns:
        label:       'NORMAL' or 'PNEUMONIA'
        probability: confidence score (0.0 to 1.0)
        diagnosis:   formatted clinical report string
    """
    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prob = model.predict(img_array, verbose=0)[0][0]
    label = 'PNEUMONIA' if prob >= threshold else 'NORMAL'

    # Format output
    diagnosis = f"""
╔══════════════════════════════════════════════╗
║      AI CHEST X-RAY ANALYSIS REPORT         ║
╠══════════════════════════════════════════════╣
║ Image:       {os.path.basename(img_path):<30} ║
║ Prediction:  {label:<30} ║
║ Confidence:  {prob*100:.1f}%{' '*(28-len(f'{prob*100:.1f}%'))} ║
║ Status:      {'⚠️  REFER TO SPECIALIST' if label=='PNEUMONIA' else '✅ APPEARS NORMAL':<30} ║
╚══════════════════════════════════════════════╝
    """

    print(diagnosis)
    return label, prob, diagnosis


def predict_and_display(model, img_path, true_label=None,
                         save_path='outputs/single_prediction.png'):
    """Predict and display the image with result."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    label, prob, _ = predict_single_image(model, img_path)

    img = image.load_img(img_path, target_size=(224, 224))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    color = 'red' if label == 'PNEUMONIA' else 'green'
    title = f"Prediction: {label}\nConfidence: {prob*100:.1f}%"
    if true_label:
        title += f"\nTrue Label: {true_label}"
    ax.set_title(title, color=color, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

    return label, prob