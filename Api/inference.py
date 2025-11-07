from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
from tensorflow import keras

load_model = keras.models.load_model

HERE = Path(__file__).resolve().parent
BASE_DIR = Path(os.getenv("BASE_DIR", HERE.parent))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "Model"))
INPUT_DIR = Path(os.getenv("INPUT_DIR", BASE_DIR / "Input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "Comets_Output"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")

_model_cache: Dict[str, keras.Model] = {}


def _ensure_directories() -> None:
    for directory in (MODEL_DIR, INPUT_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def list_models() -> list[str]:
    """Return the available `.h5` models sorted alphabetically."""

    if not MODEL_DIR.exists():
        return []
    return sorted(
        f.name
        for f in MODEL_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".h5"
    )


def _default_model_name() -> str:
    if DEFAULT_MODEL:
        return DEFAULT_MODEL

    available = list_models()
    if not available:
        raise FileNotFoundError(
            "No se encontraron modelos `.h5`. Establece DEFAULT_MODEL o monta "
            f"archivos en {MODEL_DIR}."
        )
    return available[0]


def _get_model_path(model_name: str | None) -> tuple[str, Path]:
    name = model_name or _default_model_name()
    path = MODEL_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Modelo no encontrado: {name}")
    return name, path


def load_unet(model_name: str | None = None):
    name, path = _get_model_path(model_name)
    if name not in _model_cache:
        _model_cache[name] = load_model(path, compile=False)
    return _model_cache[name]


def preprocess(img_bgr: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_hw[::-1])  # cv2 usa (ancho, alto)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def postprocess(pred: np.ndarray, thr: float = 0.5) -> np.ndarray:
    data = np.squeeze(pred, axis=0)
    if data.ndim == 3:
        # Si hay varias clases nos quedamos con la última (cola) por defecto
        data = data[..., -1]
    mask = (data > thr).astype(np.uint8) * 255
    return mask


def run_inference(img_bgr: np.ndarray, model_name: str | None = None) -> np.ndarray:
    model = load_unet(model_name)
    _, height, width, _ = model.input_shape
    x = preprocess(img_bgr, (height, width))
    y = model.predict(x)
    return postprocess(y)


def save_mask(mask: np.ndarray, out_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), mask)
    return out_path


_ensure_directories()
