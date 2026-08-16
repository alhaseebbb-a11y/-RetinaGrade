#!/usr/bin/env python
"""DR-Grade — FastAPI inference service.

Serves the trained EfficientNet-B3 CORAL model (outputs/best_model.keras) over
HTTP so the React frontend can request real DR-grade predictions.

Endpoints:
  GET  /api/health   -> model loaded? + device info
  GET  /api/metrics  -> test-set metrics (outputs/test_metrics.json)
  POST /api/predict  -> multipart image upload -> {grade, confidence, probs...}

Usage:
  source ../setenv.sh            # from repo root
  CUDA_VISIBLE_DEVICES=1 uvicorn app:app --host 0.0.0.0 --port 8000
  # or BACKEND_DEVICE=cpu to fall back to CPU
"""

import io
import os
import sys
import time
import urllib.request

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, REPO_DIR)

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ordinal import (
    OrdinalAccuracy,
    threshold_probs,
    thresholds_to_class_probs,
)

MODEL_PATH = os.path.join(REPO_DIR, "outputs", "best_model.keras")
# Optional: a public/private URL for the .keras file. On Render the 127 MB
# model cannot live in the repo (GitHub 100 MB/file limit), so the backend
# downloads it from MODEL_URL (e.g. Hugging Face) on first startup and caches
# it on disk. Locally the file already exists, so nothing is downloaded.
MODEL_URL = os.environ.get("MODEL_URL", "")
METRICS_PATH = os.path.join(REPO_DIR, "outputs", "test_metrics.json")
IMAGE_SIZE = 300
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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

TTA_TRANSFORMS = [
    ("original", lambda x: x),
    ("hflip", lambda x: np.flip(x, axis=2)),
    ("vflip", lambda x: np.flip(x, axis=1)),
    ("hflip+vflip", lambda x: np.flip(np.flip(x, axis=1), axis=2)),
]

_MODEL = None
_DEVICE = None


def ensure_model():
    """Download the model from MODEL_URL on first startup if it is not already
    present locally (needed on Render, where the 127 MB file is not in the repo)."""
    if os.path.exists(MODEL_PATH):
        return
    if not MODEL_URL:
        raise RuntimeError(
            f"Model not found at {MODEL_PATH} and MODEL_URL is not set. "
            "Set MODEL_URL to the .keras file location (e.g. Hugging Face) and redeploy."
        )
    print(f"Downloading model from {MODEL_URL} ...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    t0 = time.time()
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH + ".part")
    os.replace(MODEL_PATH + ".part", MODEL_PATH)
    print(f"Model downloaded in {time.time() - t0:.1f}s")


def load_model():
    """Load the Keras model once. Requires ordinal.py already imported."""
    global _MODEL
    import tensorflow as tf

    visible = tf.config.list_physical_devices("GPU")
    for gpu in visible:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    # Importing ordinal.py registers CORALLoss / OrdinalAccuracy, so the
    # serialized custom objects can be resolved by name on load.
    from ordinal import CORALLoss  # noqa: F401

    _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _MODEL


def model_meta():
    return {
        "path": MODEL_PATH,
        "size_mb": round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 1),
        "image_size": IMAGE_SIZE,
        "num_classes": len(GRADE_NAMES),
        "device": _DEVICE,
    }


def decode_image(data: bytes) -> np.ndarray:
    """Decode to RGB uint8, resized to (IMAGE_SIZE, IMAGE_SIZE) — identical to
    the training pipeline (image_dataset_from_directory resize)."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def predict_image(arr: np.ndarray, tta: bool):
    """Return threshold probs, class probs and predicted grade."""
    if tta:
        acc = None
        for _, fn in TTA_TRANSFORMS:
            cur = _MODEL(fn(arr)[None], training=False)
            tp = threshold_probs(cur).numpy()[0]
            acc = tp if acc is None else acc + tp
        tp = acc / len(TTA_TRANSFORMS)
    else:
        tp = threshold_probs(_MODEL(arr[None], training=False)).numpy()[0]
    class_probs = thresholds_to_class_probs(tp)
    grade = int((tp > 0.5).sum())
    return tp, class_probs, grade


app = FastAPI(
    title="DR-Grade API",
    description="Diabetic Retinopathy severity grading (EfficientNet-B3, CORAL ordinal head).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    global _DEVICE
    _DEVICE = os.environ.get("BACKEND_DEVICE", "gpu")
    print("Checking model file ...")
    ensure_model()
    print("Loading model ...")
    t0 = time.time()
    load_model()
    print(f"Model loaded in {time.time() - t0:.1f}s")


@app.get("/api/health")
def health():
    return {
        "status": "ok" if _MODEL is not None else "loading",
        "model": model_meta(),
    }


@app.get("/api/metrics")
def metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="test_metrics.json not found — run evaluate.py first")
    import json

    with open(METRICS_PATH) as f:
        return json.load(f)


@app.post("/api/predict")
async def predict(file: UploadFile = File(...), tta: bool = True):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if file.content_type not in ALLOWED_TYPES and not any(
        file.filename and file.filename.lower().endswith(e) for e in ALLOWED_EXTS
    ):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file. Upload a JPEG, PNG, WEBP or BMP image.",
        )
    data = await file.read()
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB).")
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    arr = decode_image(data)

    t0 = time.time()
    tp, class_probs, grade = predict_image(arr, tta)
    latency_ms = (time.time() - t0) * 1000.0

    return {
        "grade": grade,
        "grade_name": GRADE_NAMES[grade],
        "grade_description": GRADE_DESCRIPTIONS[grade],
        "confidence": float(class_probs[grade]),
        "probs": {str(i): float(p) for i, p in enumerate(class_probs)},
        "threshold_probs": {str(i): float(p) for i, p in enumerate(tp)},
        "tta": tta,
        "latency_ms": round(latency_ms, 1),
        "filename": file.filename,
        "model": model_meta(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
