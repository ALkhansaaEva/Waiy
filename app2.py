import os, io, time, json, base64
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ========= STATIC SETTINGS (edit here) =======================================
TFLITE_MODEL_PATH: str = "model/emotion_multiclass_final.tflite"
LABEL_MAP_PATH:    str = "model/label_map_multiclass.json"

ENABLE_API:  bool = True     # Disable to block /predict* routes (503)
ENABLE_UI:   bool = True     # Disable to hide the UI at "/"
ENABLE_DOCS: bool = True     # Disable to remove /docs, /redoc, /openapi.json

MAX_IMAGE_MB: float = 10.0   # Request size guard in MB

# CORS (restrict in production)
CORS_ALLOW_ORIGINS = ["*"]  # e.g. ["http://localhost:5173", "https://your.app"]
CORS_ALLOW_HEADERS = ["*", "Authorization", "Content-Type"]
CORS_ALLOW_METHODS = ["GET", "POST", "OPTIONS"]
# ============================================================================

# Resolve base dir once (works on Windows/Linux/Mac)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Prefer lightweight tflite-runtime; fallback to TensorFlow TFLite
try:
    from tflite_runtime.interpreter import Interpreter
except Exception:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except Exception as e:
        raise RuntimeError(
            "No TFLite runtime found. Install one of:\n"
            "  pip install tflite-runtime\n"
            "  or: pip install tensorflow==2.14.0"
        ) from e


app = FastAPI(
    title="Happy vs Sad API (TFLite)",
    version="1.2.1",
    docs_url=None if not ENABLE_DOCS else "/docs",
    redoc_url=None if not ENABLE_DOCS else "/redoc",
    openapi_url=None if not ENABLE_DOCS else "/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS if CORS_ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=CORS_ALLOW_METHODS if CORS_ALLOW_METHODS != ["*"] else ["*"],
    allow_headers=CORS_ALLOW_HEADERS if CORS_ALLOW_HEADERS != ["*"] else ["*"],
)

# Mount static only if UI enabled and folder exists
if ENABLE_UI and STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Labels
label_map: Dict[int, str] = {
    0: "Happy",
    1: "Disgust",
    2: "Fear",
    3: "Angry",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}
def load_label_map(path: str) -> Dict[int, str]:
    """Load label map JSON; fallback to {0:'Sad',1:'Happy'}."""
    p = BASE_DIR / path if not Path(path).is_absolute() else Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {int(k): v for k, v in m.items()}
    return label_map
# TFLite core
interpreter = None
input_detail = None
output_detail = None

def load_tflite(model_path: str):
    """Load TFLite model and prepare IO details."""
    global interpreter, input_detail, output_detail, label_map
    p = BASE_DIR / model_path if not Path(model_path).is_absolute() else Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"TFLite model not found: {p}")
    interpreter = Interpreter(model_path=str(p))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    label_map = load_label_map(LABEL_MAP_PATH)
    print("Loaded TFLite model:", p)
    print("Input:", input_detail)
    print("Output:", output_detail)
    print("Labels:", label_map)

@app.on_event("startup")
def _startup():
    load_tflite(TFLITE_MODEL_PATH)

# Preprocessing
IMG_W = IMG_H = 48
ACCEPTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

def preprocess_pil_to_gray48(im: Image.Image) -> np.ndarray:
    """PIL image -> [1,48,48,1] float32 in [0,1]."""
    im = im.convert("L").resize((IMG_W, IMG_H))
    arr = np.array(im, dtype=np.uint8)       # [48,48]
    x = arr.astype("float32") / 255.0        # [48,48]
    x = x.reshape(1, IMG_H, IMG_W, 1)        # [1,48,48,1]
    return x

def quantize_if_needed(x_float: np.ndarray) -> np.ndarray:
    """Quantize input if model expects quantized dtype."""
    if input_detail["dtype"] == np.float32:
        return x_float.astype(np.float32)
    scale, zero = input_detail.get("quantization", (0.0, 0))
    if not scale:
        return (x_float * 255.0).astype(input_detail["dtype"])
    return (x_float / scale + zero).round().astype(input_detail["dtype"])

def dequantize_if_needed(y: np.ndarray) -> np.ndarray:
    """Dequantize output if model is quantized."""
    if output_detail["dtype"] == np.float32:
        return y.astype(np.float32)
    scale, zero = output_detail.get("quantization", (0.0, 0))
    if not scale:
        return y.astype(np.float32)
    return scale * (y.astype(np.float32) - zero)

# Inference
def infer_array(x_float: np.ndarray) -> Dict[str, Any]:
    """Run inference and return JSON-friendly dict."""
    x_in = quantize_if_needed(x_float)
    t0 = time.time()
    interpreter.set_tensor(input_detail["index"], x_in)
    interpreter.invoke()
    y = interpreter.get_tensor(output_detail["index"])[0]
    dt = (time.time() - t0) * 1000.0
    probs = dequantize_if_needed(y)

    # Normalize for display
    s = float(np.sum(probs))
    if s and abs(s - 1.0) > 1e-3:
        probs = probs / s

    pred_idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    probs_dict = {label_map[i]: float(p) for i, p in enumerate(probs)}
    return {
        "label": label_map[pred_idx],
        "confidence": conf,
        "probs": probs_dict,
        "inference_ms": round(dt, 2),
    }

def _ensure_api_enabled():
    if not ENABLE_API:
        raise HTTPException(status_code=503, detail="API disabled by configuration.")

def _enforce_size_limit(num_bytes: int):
    max_bytes = int(MAX_IMAGE_MB * 1024 * 1024)
    if num_bytes > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image too large (> {MAX_IMAGE_MB} MB).")

# Routes
@app.get("/health")
def health():
    return {"ok": True, "model_loaded": interpreter is not None}

@app.get("/labels")
def labels():
    return label_map

@app.get("/", response_class=HTMLResponse)
def root():
    if not ENABLE_UI:
        raise HTTPException(status_code=404, detail="UI disabled.")
    # Prefer ./index.html, else ./static/index.html
    candidates = [
        BASE_DIR / "index.html",
        STATIC_DIR / "index.html",
    ]
    for p in candidates:
        if p.exists():
            return FileResponse(p)
    raise HTTPException(
        status_code=500,
        detail="index.html not found. Place it next to app.py or at static/index.html."
    )

@app.post("/predict")
async def predict(file: Optional[UploadFile] = File(default=None)):
    """multipart/form-data: field 'file' with an image."""
    _ensure_api_enabled()
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded. Use 'file' field.")
    try:
        data = await file.read()
        _enforce_size_limit(len(data))
        im = Image.open(io.BytesIO(data))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    x = preprocess_pil_to_gray48(im)
    return JSONResponse(infer_array(x))

@app.post("/predict_base64")
async def predict_base64(payload: Dict[str, Any]):
    """application/json: { "image_base64": "data:image/png;base64,..." }"""
    _ensure_api_enabled()
    b64 = payload.get("image_base64")
    if not b64:
        raise HTTPException(status_code=400, detail="Missing 'image_base64'.")
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        _enforce_size_limit(len(data))
        im = Image.open(io.BytesIO(data))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image.")
    x = preprocess_pil_to_gray48(im)
    return JSONResponse(infer_array(x))

# Dev:
#   pip install -U fastapi uvicorn pillow numpy tflite-runtime
#   # or: pip install tensorflow==2.14.0
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
