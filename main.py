"""
main.py
=======================================================
AI-Powered Medical Image Analysis System
Chest X-Ray Pneumonia Detection using MobileNetV2
=======================================================
Run this to execute the complete pipeline:
  1. Load and explore dataset
  2. Build MobileNetV2 model
  3. Train with augmentation + class weights
  4. Evaluate: accuracy, F1, AUC, confusion matrix, ROC
  5. Generate Grad-CAM heatmaps
  6. Simulate clinical prediction report

Usage:
    python main.py

Dataset: Chest X-Ray Images (Pneumonia) — Kaggle
         Place in: data/chest_xray/
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.data_loader import (get_data_generators, show_sample_images,
                              show_class_distribution)
from src.model       import build_mobilenetv2_model, unfreeze_top_layers, get_model_summary
from src.train       import compute_class_weights, train_model, fine_tune_model
from src.evaluate    import (evaluate_model, plot_training_history,
                              plot_confusion_matrix, plot_roc_curve)
from src.predict     import load_model, predict_and_display
from src.gradcam     import visualize_gradcam
from src.visualize   import plot_sample_predictions
import glob


DATA_DIR   = 'data/chest_xray'
MODEL_PATH = 'models/mobilenetv2_chest.h5'


def main():
    print("=" * 58)
    print("  AI-Powered Medical Image Analysis System")
    print("  Chest X-Ray Pneumonia Detection — MobileNetV2")
    print("=" * 58)

    # ── PHASE 1: Load Dataset ──────────────────────────────────
    print("\n📂 PHASE 1: Loading dataset...")
    train_gen, val_gen, test_gen = get_data_generators(DATA_DIR)
    show_sample_images(train_gen)
    show_class_distribution(DATA_DIR)

    # ── PHASE 2: Build Model ───────────────────────────────────
    print("\n🧠 PHASE 2: Building MobileNetV2 model...")
    model, base_model = build_mobilenetv2_model()
    get_model_summary(model)
    model.summary()

    # ── PHASE 3: Train ─────────────────────────────────────────
    print("\n🚀 PHASE 3: Training model (Phase 1 — frozen base)...")
    class_weights = compute_class_weights(train_gen)
    history = train_model(model, train_gen, val_gen, class_weights, epochs=20)

    # ── PHASE 4: Fine-tune ─────────────────────────────────────
    print("\n🔥 PHASE 4: Fine-tuning top layers...")
    model = unfreeze_top_layers(model, base_model, num_layers=30)
    history_ft = fine_tune_model(model, train_gen, val_gen, class_weights, epochs=10)

    # ── PHASE 5: Evaluate ──────────────────────────────────────
    print("\n📈 PHASE 5: Evaluating on test set...")
    plot_training_history(history)
    y_true, y_pred, y_prob = evaluate_model(model, test_gen)
    class_names = list(test_gen.class_indices.keys())
    plot_confusion_matrix(y_true, y_pred, class_names)
    roc_auc = plot_roc_curve(y_true, y_prob)
    print(f"\n✅ Final AUC Score: {roc_auc:.4f}")

    # ── PHASE 6: Visualize Predictions ─────────────────────────
    print("\n🖼️  PHASE 6: Visualizing predictions...")
    test_gen.reset()
    plot_sample_predictions(model, test_gen)

    # ── PHASE 7: Grad-CAM ──────────────────────────────────────
    print("\n🔥 PHASE 7: Generating Grad-CAM heatmaps...")
    pneumonia_images = glob.glob(f'{DATA_DIR}/test/PNEUMONIA/*.jpeg')[:1]
    normal_images    = glob.glob(f'{DATA_DIR}/test/NORMAL/*.jpeg')[:1]

    if pneumonia_images:
        visualize_gradcam(model, pneumonia_images[0],
                          true_label='PNEUMONIA',
                          pred_label='PNEUMONIA',
                          pred_prob=0.94,
                          save_path='outputs/gradcam_pneumonia.png')
    if normal_images:
        visualize_gradcam(model, normal_images[0],
                          true_label='NORMAL',
                          pred_label='NORMAL',
                          pred_prob=0.08,
                          save_path='outputs/gradcam_normal.png')

    # ── PHASE 8: Clinical Simulation ───────────────────────────
    print("\n🏥 PHASE 8: Simulating clinical prediction report...")
    if pneumonia_images:
        predict_and_display(model, pneumonia_images[0],
                            true_label='PNEUMONIA',
                            save_path='outputs/clinical_report.png')

    print("\n" + "=" * 58)
    print("  ✅ ALL PHASES COMPLETE!")
    print("  Check outputs/ folder for all saved results.")
    print("=" * 58)


if __name__ == "__main__":
    main()