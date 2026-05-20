"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output

CRITICAL PATTERN: Training/Serving Consistency
- Uses fixed BINARY_MAP for deterministic binary encoding
- Applies same one-hot encoding with drop_first=True
- Maintains exact feature column order from training
- Handles missing/new categorical values gracefully

Production Deployment:
- MODEL_DIR points to containerized model artifacts
- Feature schema loaded from training-time artifacts
- Optimized for single-row inference (real-time serving)
"""

import os
import glob
import json
import pandas as pd
import mlflow

# === PATH CONFIGURATION ===
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Docker: /app/model | Local: auto-discovered from mlruns
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")

# Lazy-loaded globals (populated on first predict() call)
model = None
FEATURE_COLS = None

BINARY_MAP = {
    "gender":          {"Female": 0, "Male": 1},
    "Partner":         {"No": 0, "Yes": 1},
    "Dependents":      {"No": 0, "Yes": 1},
    "PhoneService":    {"No": 0, "Yes": 1},
    "PaperlessBilling":{"No": 0, "Yes": 1},
}
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _load_model():
    global model, MODEL_DIR
    if model is not None:
        return

    # === Docker path (/app/model) ===
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "MLmodel")):
        try:
            model = mlflow.pyfunc.load_model(MODEL_DIR)
            print(f"✅ Model loaded from {MODEL_DIR}")
            return
        except Exception as e:
            print(f"❌ Docker model load failed: {e}")

    # === Local development fallback (mlruns) ===
    patterns = [
        os.path.join(PROJECT_ROOT, "mlruns", "*", "*", "artifacts", "model"),
        os.path.join(PROJECT_ROOT, "model_export", "model"),
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))

    if not paths:
        raise FileNotFoundError(
            "No model found. Run:\n"
            "  python scripts/run_pipeline.py --input src/data/rawdata/Churn.csv --target Churn\n"
            "  python scripts/export_model.py"
        )

    latest = max(paths, key=os.path.getmtime)

    # Windows requires file:/// prefix for local paths
    model_uri = f"file:///{latest.replace(os.sep, '/')}"
    mlruns_uri = f"file:///{os.path.join(PROJECT_ROOT, 'mlruns').replace(os.sep, '/')}"
    mlflow.set_tracking_uri(mlruns_uri)

    model = mlflow.pyfunc.load_model(model_uri)
    MODEL_DIR = latest
    print(f"✅ Model loaded from local mlruns: {latest}")


def _load_features():
    global FEATURE_COLS
    if FEATURE_COLS is not None:
        return

    # Search order: model dir json → model dir txt → artifacts dir json
    candidates = [
        (os.path.join(MODEL_DIR, "feature_columns.json"), "json"),
        (os.path.join(MODEL_DIR, "feature_columns.txt"),  "txt"),
        (os.path.join(PROJECT_ROOT, "artifacts", "feature_columns.json"), "json"),
    ]

    for path, fmt in candidates:
        if os.path.exists(path):
            with open(path) as f:
                FEATURE_COLS = json.load(f) if fmt == "json" else [
                    l.strip() for l in f if l.strip()
                ]
            print(f"✅ Loaded {len(FEATURE_COLS)} feature columns from {path}")
            return

    raise FileNotFoundError(
        f"feature_columns not found in any of:\n"
        + "\n".join(f"  {p}" for p, _ in candidates)
        + "\nRe-run the pipeline and export_model.py"
    )


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip()
                .map(mapping).astype("Int64").fillna(0).astype(int)
            )

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    return df


def predict(input_dict: dict) -> str:
    # Lazy load on first call — safe for both Docker and local
    _load_model()
    _load_features()

    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        result = preds[0] if isinstance(preds, (list, tuple)) else preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    return "Likely to churn" if result == 1 else "Not likely to churn"