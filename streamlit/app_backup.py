
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# --------------------------------------------------
# Project configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs" / "png"
DATA_DIR = ROOT / "data"

IMAGE_SIZE = (224, 224)

# Change these labels if your model was trained differently.
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

st.set_page_config(
    page_title="AI Medical Image Analysis",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Dashboard styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.4rem;
        border-radius: 12px;
        background: linear-gradient(120deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; }
    .note {
        padding: 0.9rem;
        border-left: 4px solid #d97706;
        background: rgba(217, 119, 6, 0.08);
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>AI-Powered Medical Image Analysis System</h1>
        <p>Medical imaging research dashboard • Deep Learning • Computer Vision</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Educational research prototype using public datasets. "
    "Not intended for clinical diagnosis."
)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def find_model():
    """Find an existing saved Keras model."""
    if not MODEL_DIR.exists():
        return None

    candidates = [
        MODEL_DIR / "medical_image_model.keras",
        MODEL_DIR / "medical_image_model.h5",
        MODEL_DIR / "model.keras",
        MODEL_DIR / "model.h5",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    for pattern in ("*.keras", "*.h5"):
        matches = sorted(MODEL_DIR.glob(pattern))
        if matches:
            return matches[0]

    return None


def count_images(folder):
    """Count common image formats in a directory tree."""
    if not folder.exists():
        return 0

    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def prepare_image(image):
    """Prepare an RGB image for a 224x224 Keras classifier."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return image, resized, np.expand_dims(array, axis=0)


def load_keras_model(model_path):
    """Import TensorFlow only when model inference is requested."""
    import tensorflow as tf
    return tf.keras.models.load_model(model_path, compile=False)


def interpret_prediction(model, batch):
    """
    Supports a binary sigmoid output or a multiclass softmax output.
    The model must have been trained using the same preprocessing.
    """
    prediction = np.asarray(model.predict(batch, verbose=0))

    if prediction.size == 1:
        score = float(prediction.reshape(-1)[0])

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "The model returned a value outside [0, 1]. "
                "Check the model's output activation."
            )

        # Binary convention: index 0 = NORMAL, index 1 = PNEUMONIA.
        probabilities = np.array([1.0 - score, score])
    else:
        probabilities = prediction.reshape(-1).astype(float)

        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError(
                f"The model returned {len(probabilities)} scores, but "
                f"{len(CLASS_NAMES)} class labels are configured. "
                "Update CLASS_NAMES in streamlit/app.py."
            )

        # If model outputs logits rather than probabilities, convert them.
        if (
            np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1.0, atol=0.02)
        ):
            shifted = probabilities - np.max(probabilities)
            exp_scores = np.exp(shifted)
            probabilities = exp_scores / exp_scores.sum()

    predicted_index = int(np.argmax(probabilities))
    return predicted_index, probabilities


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Project navigation")
    page = st.radio(
        "Choose a section",
        [
            "Dashboard",
            "Image Analysis",
            "Dataset Overview",
            "Evaluation Outputs",
            "About Project",
        ],
    )

    st.divider()
    st.subheader("System status")

    model_path = find_model()

    if model_path:
        st.success(f"Model found: {model_path.name}")
    else:
        st.warning("No saved Keras model found.")

    st.write(f"Python project: `{ROOT.name}`")
    st.write(f"Dataset images: {count_images(DATA_DIR)}")

# --------------------------------------------------
# Dashboard page
# --------------------------------------------------
if page == "Dashboard":
    st.subheader("Project overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Dataset images", f"{count_images(DATA_DIR):,}")

    with col2:
        st.metric(
            "Saved model",
            "Available" if model_path else "Not found",
        )

    with col3:
        st.metric(
            "Evaluation PNGs",
            len(list(OUTPUT_DIR.glob("*.png")))
            if OUTPUT_DIR.exists()
            else 0,
        )

    st.divider()
    st.subheader("Analysis workflow")

    steps = [
        "1. Upload a public-dataset image",
        "2. Preview image resizing and normalization",
        "3. Run the saved model",
        "4. Review predicted class and model scores",
        "5. Inspect saved evaluation visualizations",
    ]

    for step in steps:
        st.write(step)

    st.info(
        "A model file and compatible class labels are required for "
        "real predictions. Dashboard counters alone do not indicate "
        "that training or evaluation has succeeded."
    )

# --------------------------------------------------
# Image analysis page
# --------------------------------------------------
elif page == "Image Analysis":
    st.subheader("Upload and analyze an image")

    st.write(
        "Upload a JPG, JPEG or PNG image from your permitted "
        "research dataset."
    )

    uploaded_file = st.file_uploader(
        "Choose a medical image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        try:
            original = Image.open(uploaded_file)
            original, resized, batch = prepare_image(original)

            left, right = st.columns(2)

            with left:
                st.markdown("**Original image**")
                st.image(original, use_container_width=True)

            with right:
                st.markdown("**Resized model input (224 × 224)**")
                st.image(resized, use_container_width=True)

            with st.expander("Preprocessing details"):
                st.write("Color format: RGB")
                st.write(f"Input dimensions: {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}")
                st.write("Array shape:", batch.shape)
                st.write("Data type:", str(batch.dtype))
                st.write(
                    "Pixel range:",
                    f"{batch.min():.3f} to {batch.max():.3f}",
                )
                st.write("Normalization: pixel values divided by 255.")

            st.warning(
                "This model output is experimental and must not be used "
                "to diagnose, exclude or treat a medical condition."
            )

            if st.button("Run AI prediction", type="primary"):
                model_path = find_model()

                if model_path is None:
                    st.error(
                        "No trained model was found in the models folder. "
                        "Train and save your model before predicting."
                    )
                else:
                    try:
                        with st.spinner("Loading model and analyzing image..."):
                            model = load_keras_model(model_path)
                            index, probabilities = interpret_prediction(
                                model, batch
                            )

                        predicted_class = CLASS_NAMES[index]
                        confidence = float(probabilities[index]) * 100

                        st.divider()
                        st.subheader("Experimental model output")

                        result_col, score_col = st.columns(2)

                        with result_col:
                            st.metric("Predicted class", predicted_class)

                        with score_col:
                            st.metric("Model score", f"{confidence:.2f}%")

                        st.caption(
                            "The displayed score is a model output, not a "
                            "validated clinical probability."
                        )

                        score_chart = {
                            CLASS_NAMES[i]: float(probabilities[i]) * 100
                            for i in range(len(probabilities))
                        }

                        st.write("Scores by class")
                        st.bar_chart(score_chart)

                    except Exception as exc:
                        st.error(
                            "Prediction failed. Check TensorFlow installation, "
                            "model compatibility and preprocessing settings."
                        )
                        st.exception(exc)

        except Exception as exc:
            st.error(f"Could not open this image: {exc}")

# --------------------------------------------------
# Dataset overview page
# --------------------------------------------------
elif page == "Dataset Overview":
    st.subheader("Dataset inventory")

    if not DATA_DIR.exists():
        st.warning("The data folder does not exist yet.")
    else:
        class_folders = [
            folder for folder in DATA_DIR.iterdir() if folder.is_dir()
        ]

        if not class_folders:
            st.info(
                "No class folders found directly under data/. "
                "Organize your dataset into class folders to view counts."
            )
        else:
            counts = {
                folder.name: count_images(folder)
                for folder in class_folders
            }

            st.bar_chart(counts)
            st.write("Image counts by folder")
            st.dataframe(
                [
                    {"Class / folder": name, "Image count": count}
                    for name, count in counts.items()
                ],
                use_container_width=True,
            )

    st.caption(
        "Counts are based on image files on disk, not on verified labels "
        "or successful model evaluation."
    )

# --------------------------------------------------
# Evaluation outputs page
# --------------------------------------------------
elif page == "Evaluation Outputs":
    st.subheader("Saved evaluation visualizations")

    if not OUTPUT_DIR.exists():
        st.warning(
            "No outputs/png folder exists. Generate or save your plots first."
        )
    else:
        png_files = sorted(OUTPUT_DIR.glob("*.png"))

        if not png_files:
            st.info("No PNG files were found in outputs/png.")
        else:
            st.write(f"Found {len(png_files)} PNG file(s).")

            for png_file in png_files:
                with st.expander(png_file.name, expanded=True):
                    try:
                        st.image(
                            str(png_file),
                            caption=png_file.name,
                            use_container_width=True,
                        )
                    except Exception as exc:
                        st.error(f"Unable to display image: {exc}")

            st.warning(
                "Check each image's provenance. Template plots and "
                "illustrations are not actual model evaluation results."
            )

# --------------------------------------------------
# About page
# --------------------------------------------------
elif page == "About Project":
    st.subheader("About the system")

    st.markdown(
        """
        **Project:** AI-Powered Medical Image Analysis System

        **Domain:** Artificial Intelligence, Deep Learning and Computer Vision

        **Purpose:** Educational exploration of medical-image classification
        using public datasets and saved machine-learning models.

        **Technologies:** Python, Streamlit, TensorFlow/Keras, NumPy and Pillow.

        **Limitations:** Model predictions can be incorrect or biased.
        Performance must be measured on an appropriate held-out dataset.
        This application is not a medical device and does not replace a
        qualified healthcare professional.
        """
    )

st.divider()
st.caption(
    "AI-Powered Medical Image Analysis System | Academic research prototype"
)
