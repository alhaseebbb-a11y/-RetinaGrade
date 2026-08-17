#!/usr/bin/env python
"""DR-Grade — Streamlit inference app.

Upload a retinal fundus image and get an instant diabetic retinopathy
severity grade (No DR to Proliferative DR) with per-class confidence.

Model: EfficientNet-B3 + CORAL ordinal head, hosted on Hugging Face.
"""

import io
import os
import sys
import time
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import streamlit as st
import numpy as np
from PIL import Image

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_DIR)

from ordinal import CORALLoss, OrdinalAccuracy, threshold_probs, thresholds_to_class_probs  # noqa: E402

GRADE_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]
GRADE_DESCRIPTIONS = [
    "No signs of diabetic retinopathy.",
    "Microaneurysms only — earliest visible change.",
    "More than just microaneurysms but less than severe.",
    "Extensive hemorrhages, venous beading, IRMA.",
    "Neovascularization, vitreous/pre-retinal hemorrhage.",
]
GRADE_COLORS = ["#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"]

IMAGE_SIZE = 300
HF_MODEL_URL = (
    "https://huggingface.co/alhaseebbb/dr-grade-model"
    "/resolve/main/best_model.keras"
)
MODEL_LOCAL = os.path.join(REPO_DIR, "outputs", "best_model.keras")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Downloading and loading model...")
def load_model():
    """Download the Keras model from Hugging Face (if not cached locally) and
    return the loaded tf.keras.Model."""
    import tensorflow as tf

    model_path = MODEL_LOCAL
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        st.info("Downloading model from Hugging Face (~127 MB)...")
        t0 = time.time()
        urllib.request.urlretrieve(HF_MODEL_URL, model_path + ".part")
        os.replace(model_path + ".part", model_path)
        st.info(f"Downloaded in {time.time() - t0:.1f}s")

    model = tf.keras.models.load_model(model_path, compile=False)
    return model


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def preprocess_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to a float32 numpy array sized for the model."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def predict(model, arr: np.ndarray, tta: bool):
    """Return (threshold_probs, class_probs, grade_index, latency_ms)."""
    import tensorflow as tf

    t0 = time.time()

    if tta:
        transforms = [
            lambda x: x,
            lambda x: np.flip(x, axis=2),
            lambda x: np.flip(x, axis=1),
            lambda x: np.flip(np.flip(x, axis=1), axis=2),
        ]
        acc = None
        for fn in transforms:
            cur = threshold_probs(model(fn(arr)[None], training=False)).numpy()[0]
            acc = cur if acc is None else acc + cur
        tp = acc / len(transforms)
    else:
        tp = threshold_probs(model(arr[None], training=False)).numpy()[0]

    class_probs = thresholds_to_class_probs(tp)
    grade = int((tp > 0.5).sum())
    latency_ms = (time.time() - t0) * 1000.0
    return tp, class_probs, grade, latency_ms


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RetinaGrade — DR Severity Grading",
    page_icon="👁",
    layout="wide",
)

st.markdown(
    """
    <style>
    .grade-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 1rem;
        padding: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    model = load_model()

    # -- Header --
    st.markdown(
        '<h1 style="text-align:center; margin-bottom:0.2rem">'
        "👁 RetinaGrade</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="text-align:center; color:#64748b; margin-top:0">'
        "Diabetic Retinopathy severity grading from a single fundus image<br>"
        "<small>EfficientNet-B3 · CORAL ordinal regression · QWK 0.8718</small></p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # -- Sidebar --
    with st.sidebar:
        st.header("Settings")
        tta = st.checkbox(
            "Test-time augmentation",
            value=True,
            help="Average 4 flips for more stable predictions.",
        )
        st.divider()
        st.subheader("Model info")
        st.markdown(
            f"- **Architecture**: EfficientNet-B3 + CORAL head\n"
            f"- **Input**: {IMAGE_SIZE}×{IMAGE_SIZE} RGB\n"
            f"- **Classes**: {len(GRADE_NAMES)} DR grades\n"
            f"- **Device**: CPU"
        )
        st.divider()
        st.subheader("About")
        st.markdown(
            "Research/demo only — **not a medical device**.\n\n"
            "Trained on 45,312 fundus images with ordinal (CORAL) loss "
            "that respects DR grade ordering."
        )

    # -- Upload --
    col_upload, col_result = st.columns([1, 1])

    with col_upload:
        st.subheader("Upload image")
        uploaded = st.file_uploader(
            "Drop a fundus photograph",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )

        if uploaded is not None:
            img_display = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            st.image(img_display, caption=uploaded.name, use_column_width=True)

        analyze = st.button(
            "🔬  Analyze image",
            type="primary",
            use_container_width=True,
            disabled=uploaded is None,
        )

    # -- Prediction --
    with col_result:
        st.subheader("Result")

        if uploaded is not None and analyze:
            with st.spinner("Running inference..."):
                arr = preprocess_image(uploaded.getvalue())
                tp, class_probs, grade, latency_ms = predict(model, arr, tta)

            color = GRADE_COLORS[grade]
            st.markdown(
                f'<span class="grade-badge" style="background:{color}">'
                f"{GRADE_NAMES[grade]}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"*{GRADE_DESCRIPTIONS[grade]}*")

            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{class_probs[grade]:.1%}")
            c2.metric("Latency", f"{latency_ms:.0f} ms")

            st.markdown("**Per-class probability**")
            chart_data = {
                GRADE_NAMES[i]: float(class_probs[i])
                for i in range(len(GRADE_NAMES))
            }
            st.bar_chart(chart_data, color=color, height=200)

            with st.expander("Raw threshold probabilities P(grade ≥ k)"):
                for i in range(len(GRADE_NAMES) - 1):
                    st.progress(
                        float(tp[i]),
                        text=f"P(grade ≥ {GRADE_NAMES[i + 1]}): {float(tp[i]):.4f}",
                    )

        elif uploaded is None:
            st.info("Upload a fundus image on the left to get started.")

    # -- Footer --
    st.divider()
    st.caption(
        "RetinaGrade · EfficientNet-B3 CORAL ordinal classifier · "
        "research/demo only, not a medical device · test QWK 0.8718"
    )


if __name__ == "__main__":
    main()
