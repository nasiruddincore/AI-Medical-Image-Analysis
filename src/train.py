"""
train.py
Trains MobileNetV2 model with:
- Early stopping (stops if val_loss stops improving)
- Model checkpoint (saves best model automatically)
- Learning rate reduction (reduces LR on plateau)
- Class weights (handles class imbalance)
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint,
    ReduceLROnPlateau, CSVLogger
)
import numpy as np
import os


def compute_class_weights(train_generator):
    """
    Compute class weights to handle imbalance.
    Pneumonia images >> Normal images in this dataset.
    Class weights tell the model to pay more attention to minority class.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_generator.classes
    classes = np.unique(labels)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )

    class_weight_dict = dict(zip(classes, weights))
    print(f"✅ Class weights: {class_weight_dict}")
    return class_weight_dict


def get_callbacks(model_save_path='models/mobilenetv2_chest.h5'):
    """
    Define training callbacks:
    
    EarlyStopping:    stop if val_loss doesn't improve for 5 epochs
    ModelCheckpoint:  save the best model automatically
    ReduceLROnPlateau: halve learning rate if val_loss stagnates
    CSVLogger:        log all metrics to CSV for analysis
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger('outputs/training_log.csv', append=False)
    ]

    print("✅ Callbacks configured: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger")
    return callbacks


def train_model(model, train_gen, val_gen, class_weights, epochs=20):
    """
    Phase 1 training: frozen base model, train classification head only.
    """
    print(f"\n🚀 Starting Phase 1 training (frozen base, {epochs} epochs max)...")

    callbacks = get_callbacks()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print(f"\n✅ Phase 1 training complete!")
    best_val_acc = max(history.history['val_accuracy'])
    print(f"   Best validation accuracy: {best_val_acc:.4f}")

    return history


def fine_tune_model(model, train_gen, val_gen, class_weights, epochs=10):
    """
    Phase 2 fine-tuning: unfreeze top layers, train with tiny LR.
    Run this AFTER Phase 1 converges.
    """
    print(f"\n🔥 Starting Phase 2 fine-tuning ({epochs} epochs max)...")

    callbacks = get_callbacks('models/mobilenetv2_chest_finetuned.h5')

    history_ft = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    print(f"\n✅ Fine-tuning complete!")
    best_val_acc = max(history_ft.history['val_accuracy'])
    print(f"   Best validation accuracy after fine-tuning: {best_val_acc:.4f}")

    return history_ft