"""
data_loader.py
Loads chest X-ray images using Keras ImageDataGenerator.
Applies augmentation on training set to prevent overfitting.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

# Constants
IMG_SIZE   = (224, 224)   # MobileNetV2 input size
BATCH_SIZE = 32


def get_data_generators(data_dir):
    """
    Create train, validation, and test data generators.
    
    Augmentation on TRAIN only:
    - Horizontal flip (X-rays can be mirrored)
    - Small rotations (slight angle variations)
    - Zoom (different distances from X-ray machine)
    - Width/height shift
    
    Val and Test: only rescale (no augmentation — real data)
    """
    
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,            # normalize pixel values to [0,1]
        horizontal_flip=True,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

    # Validation and test — only normalize, no augmentation
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',       # binary: NORMAL=0, PNEUMONIA=1
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_dir, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    print(f"✅ Train samples:      {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples:       {test_generator.samples}")
    print(f"✅ Classes:            {train_generator.class_indices}")

    return train_generator, val_generator, test_generator


def show_sample_images(generator, save_path='outputs/sample_images.png'):
    """
    Display a grid of sample chest X-ray images from the dataset.
    This is your first GitHub proof screenshot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    images, labels = next(generator)
    class_names = {v: k for k, v in generator.class_indices.items()}

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    fig.suptitle('Sample Chest X-Ray Images from Dataset', fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(images[i])
            label = class_names[int(labels[i])]
            color = 'red' if label == 'PNEUMONIA' else 'green'
            ax.set_title(label, color=color, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Sample images saved: {save_path}")


def show_class_distribution(data_dir, save_path='outputs/class_distribution.png'):
    """
    Bar chart showing how many images per class.
    Reveals class imbalance (pneumonia >> normal in this dataset).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']

    counts = {split: {} for split in splits}
    for split in splits:
        for cls in classes:
            path = os.path.join(data_dir, split, cls)
            if os.path.exists(path):
                counts[split][cls] = len(os.listdir(path))
            else:
                counts[split][cls] = 0

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ['steelblue', 'tomato']

    for ax, split in zip(axes, splits):
        vals = [counts[split].get(cls, 0) for cls in classes]
        bars = ax.bar(classes, vals, color=colors, edgecolor='white', linewidth=1.2)
        ax.set_title(f'{split.upper()} set', fontsize=13)
        ax.set_ylabel('Number of images')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(val), ha='center', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Class Distribution — Chest X-Ray Dataset', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"✅ Class distribution saved: {save_path}")