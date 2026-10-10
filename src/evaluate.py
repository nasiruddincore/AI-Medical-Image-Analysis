"""
evaluate.py
Full model evaluation:
- Accuracy, Precision, Recall, F1, AUC
- Confusion matrix
- ROC curve
- Classification report
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, f1_score
)
import os


def evaluate_model(model, test_generator, save_dir='outputs'):
    """
    Run model on test set and compute all metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n📊 Evaluating on test set...")

    # Get predictions
    test_generator.reset()
    y_prob = model.predict(test_generator, verbose=1)
    y_pred = (y_prob > 0.5).astype(int).flatten()
    y_true = test_generator.classes

    class_names = list(test_generator.class_indices.keys())

    # Metrics
    acc  = np.mean(y_pred == y_true)
    f1   = f1_score(y_true, y_pred)

    print(f"\n✅ Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"✅ F1 Score:       {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=class_names)}")

    return y_true, y_pred, y_prob.flatten()


def plot_training_history(history, save_path='outputs/training_history.png'):
    """
    Plot accuracy and loss curves over epochs.
    Shows if model is overfitting or learning well.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='steelblue', lw=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='tomato',    lw=2)
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', color='steelblue', lw=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='tomato',    lw=2)
    axes[1].set_title('Model Loss Over Epochs', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('MobileNetV2 Training History — Chest X-Ray', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Training history saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names,
                           save_path='outputs/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix — MobileNetV2 Chest X-Ray', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true, y_prob, save_path='outputs/roc_curve.png'):
    """Plot ROC curve with AUC score."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'navy', lw=1, linestyle='--',
            label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curve — Chest X-Ray Pneumonia Detection', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ ROC curve saved: {save_path}")
    return roc_auc