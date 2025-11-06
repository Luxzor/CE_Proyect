import os
import numpy as np
import cv2
from tensorflow import keras
load_model = keras.models.load_model

HERE = os.path.dirname(__file__)
BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(HERE, "..")))
MODEL_DIR  = os.path.join(BASE_DIR, "Program", "Model")
INPUT_DIR  = os.path.join(BASE_DIR, "Input")
OUTPUT_DIR = os.path.join(BASE_DIR, "Comets_Output")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Model_Unet1.h5")

_model_cache = {}

def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".h5")]

def _get_model_path(model_name: str | None):
    name = model_name or DEFAULT_MODEL
    path = os.path.join(MODEL_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Modelo no encontrado: {name}")
    return name, path

def load_unet(model_name: str | None = None):
    name, path = _get_model_path(model_name)
    if name not in _model_cache:
        _model_cache[name] = load_model(path, compile=False)
    return _model_cache[name]

def preprocess(img_bgr):
    # Ajusta si tu preprocesamiento real difiere
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess(pred, thr=0.5):
    mask = (pred[0] > thr).astype(np.uint8) * 255
    return mask

def run_inference(img_bgr, model_name: str | None = None):
    model = load_unet(model_name)
    x = preprocess(img_bgr)
    y = model.predict(x)
    mask = postprocess(y)
    return mask

def save_mask(mask, out_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, mask)
    return out_path

def load_unet(model_name: str | None = None):
    name, path = _get_model_path(model_name)
    if name not in _model_cache:
        _model_cache[name] = load_model(path, compile=False)
    return _model_cache[name]

def preprocess(img_bgr, target_hw):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_hw[::-1])  # (w,h) para cv2
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def run_inference(img_bgr, model_name: str | None = None):
    model = load_unet(model_name)
    # obtiene alto y ancho del input (batch, h, w, c)
    _, H, W, _ = model.input_shape
    x = preprocess(img_bgr, (H, W))
    y = model.predict(x)
    mask = postprocess(y)
    return mask

for d in (MODEL_DIR, INPUT_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)
