# app.py
# -----------------------------------------------------------------------------
# FastAPI Emotion Recognition API (PyTorch + timm EfficientNet)
# - Same endpoints/config names as legacy TFLite API for drop-in compatibility.
# - High-quality preprocessing before inference:
#   denoise -> face crop (with margin) -> CLAHE -> Unsharp -> TTA (FiveCrop+Flip).
# - Pydantic v2-safe query params (no regex on int; use Literal[2, 10]).
# - Forced labels mapping as requested (Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise).
# -----------------------------------------------------------------------------

import os, io, time, json, base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Literal

import numpy as np
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# =================== STATIC SETTINGS (names kept the same) ===================
# NOTE: TFLITE_MODEL_PATH now points to a PyTorch .pt checkpoint (not .tflite).
TFLITE_MODEL_PATH: str = "model/enet_b2_7.pt"          # <- your PyTorch model
LABEL_MAP_PATH:    str = "model/label_map_multiclass.json"

ENABLE_API:  bool = True
ENABLE_UI:   bool = True
ENABLE_DOCS: bool = True

MAX_IMAGE_MB: int = 10        # request size limit in MB
DEFAULT_IMG_SIZE: int = 320   # try 288/320/352
DEFAULT_TTA_CROPS: int = 10   # 10 = five-crop+flips, 2 = center+flip
FORCE_DEFAULT_LABELS: bool = True  # ignore JSON label map and use the mapping below
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# ----------------------------- PyTorch / timm --------------------------------
import torch
import timm
import cv2
import torchvision.transforms.functional as TF
from torchvision import transforms

# Unpickle timm EfficientNet if checkpoint needs it
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[torch.nn.Module] = None

# Labels (forced)
_default_label_map: Dict[int, str] = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}
label_map: Dict[int, str] = dict(_default_label_map)

# ImageNet normalization (EfficientNet/timm)
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_to_tensor_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])

# OpenCV Haar face detector (built-in path)
_HAAR = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

# --------------------------------- FastAPI -----------------------------------
app = FastAPI(
    title="Emotion API (PyTorch/timm)",
    version="2.2.0",
    docs_url=None if not ENABLE_DOCS else "/docs",
    redoc_url=None if not ENABLE_DOCS else "/redoc",
    openapi_url=None if not ENABLE_DOCS else "/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

if ENABLE_UI and STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --------------------------------- Labels ------------------------------------
def load_label_map(path: str) -> Dict[int, str]:
    """
    Load label map JSON if present (and FORCE_DEFAULT_LABELS=False),
    otherwise use the forced default mapping.
    """
    if FORCE_DEFAULT_LABELS:
        return dict(_default_label_map)
    p = BASE_DIR / path if not Path(path).is_absolute() else Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {int(k): v for k, v in m.items()}
    return dict(_default_label_map)

# ---------------------------- Model load / startup ----------------------------
def _load_torch_model(model_path: str) -> torch.nn.Module:
    """Load a PyTorch .pt checkpoint; move to device; eval mode."""
    p = BASE_DIR / model_path if not Path(model_path).is_absolute() else Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    model = torch.load(str(p), map_location=_device, weights_only=False)
    if not hasattr(model, "forward"):
        raise RuntimeError("Loaded object is not a torch.nn.Module")
    model.to(_device).eval()
    return model

@app.on_event("startup")
def _startup():
    global _model, label_map
    label_map = load_label_map(LABEL_MAP_PATH)
    _model = _load_torch_model(TFLITE_MODEL_PATH)  # keep var name for backward-compat
    print("Loaded PyTorch model:", TFLITE_MODEL_PATH)
    print("Using device:", _device)
    print("Labels:", label_map)

# --------------------------- Preprocessing (HQ) -------------------------------
def _largest_bbox(gray: np.ndarray):
    """Return largest face bbox (x,y,w,h) or None."""
    faces = _HAAR.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    return faces[int(np.argmax(areas))]

def _crop_face_or_center(img_rgb: np.ndarray) -> np.ndarray:
    """Face crop with margin; fallback to centered square."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bb = _largest_bbox(gray)
    if bb is not None:
        x, y, w, h = bb
        m = int(0.35 * max(w, h))
        x0 = max(0, x - m); y0 = max(0, y - m)
        x1 = min(img_rgb.shape[1], x + w + m); y1 = min(img_rgb.shape[0], y + h + m)
        return img_rgb[y0:y1, x0:x1]
    # fallback
    H, W = img_rgb.shape[:2]
    side = min(H, W)
    sy = (H - side) // 2; sx = (W - side) // 2
    return img_rgb[sy:sy+side, sx:sx+side]

def _clahe_rgb(img_rgb: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE on L channel to stabilize contrast/lighting."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def _unsharp(img_rgb: np.ndarray, k: float = 0.8, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask to emphasize facial edges."""
    blur = cv2.GaussianBlur(img_rgb, (0, 0), sigma)
    sharp = cv2.addWeighted(img_rgb, 1 + k, blur, -k, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _denoise_gentle(img_rgb: np.ndarray) -> np.ndarray:
    """Light bilateral denoising for compression noise without blurring edges."""
    return cv2.bilateralFilter(img_rgb, d=5, sigmaColor=35, sigmaSpace=35)

def _tta_center_flip(pil_img: Image.Image, size: int) -> torch.Tensor:
    """Fast TTA: center + horizontal flip -> [2,3,S,S]."""
    fitted = ImageOps.fit(pil_img, (size, size), method=Image.BILINEAR)
    x1 = _to_tensor_norm(fitted)
    x2 = _to_tensor_norm(ImageOps.mirror(fitted))
    return torch.stack([x1, x2], dim=0)

def _tta_fivecrop_flip(pil_img: Image.Image, size: int) -> torch.Tensor:
    """Full TTA: FiveCrop + flips -> [10,3,S,S]."""
    fitted = ImageOps.fit(pil_img, (size + 32, size + 32), method=Image.BILINEAR)
    tl, tr, bl, br, center = TF.five_crop(fitted, size)
    crops = [tl, tr, bl, br, center]
    crops += [TF.hflip(c) for c in crops]
    return torch.stack([_to_tensor_norm(c) for c in crops], dim=0)

def preprocess_image_to_tta(
    im: Image.Image,
    img_size: int = DEFAULT_IMG_SIZE,
    tta_crops: int = DEFAULT_TTA_CROPS
) -> torch.Tensor:
    """
    Full pipeline:
      - Convert to RGB ndarray (handle GRAY/RGBA)
      - Denoise -> face crop (with margin) -> CLAHE -> Unsharp
      - PIL -> TTA tensor [N,3,S,S]
    """
    # Normalize mode -> numpy RGB
    if im.mode not in ("RGB", "RGBA", "L"):
        im = im.convert("RGBA")
    nd = np.array(im)
    if nd.ndim == 2:
        img_rgb = cv2.cvtColor(nd, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(nd, cv2.COLOR_RGBA2RGB) if nd.shape[2] == 4 else nd

    # Enhance facial features before inference
    img_rgb = _denoise_gentle(img_rgb)
    face = _crop_face_or_center(img_rgb)
    face = _clahe_rgb(face, clip=2.0)
    face = _unsharp(face, k=0.8, sigma=1.0)

    # PIL + TTA
    pil_face = Image.fromarray(face)
    if tta_crops == 2:
        batch = _tta_center_flip(pil_face, img_size)
    else:
        batch = _tta_fivecrop_flip(pil_face, img_size)
    return batch  # [N,3,S,S]

# -------------------------------- Inference ----------------------------------
def infer_batch_tta(x_batch: torch.Tensor) -> Dict[str, Any]:
    """
    Inference on a TTA batch [N,3,S,S]:
      - Aggregate logits by mean
      - Softmax once
    Returns label, confidence, probs dict, inference_ms.
    """
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    t0 = time.time()
    with torch.no_grad():
        logits = _model(x_batch.to(_device, dtype=torch.float32))  # [N,C]
        if logits.ndim != 2:
            raise HTTPException(status_code=500, detail=f"Unexpected logits shape: {tuple(logits.shape)}")
        logits_mean = logits.mean(dim=0, keepdim=True)             # [1,C]
        probs = torch.softmax(logits_mean, dim=1)[0].detach().cpu().numpy()
    dt = (time.time() - t0) * 1000.0

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    probs_dict = {label_map.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    return {
        "label": label_map.get(pred_idx, str(pred_idx)),
        "confidence": conf,
        "probs": probs_dict,
        "inference_ms": round(dt, 2),
    }

# --------------------------------- Helpers -----------------------------------
def _ensure_api_enabled():
    if not ENABLE_API:
        raise HTTPException(status_code=503, detail="API disabled by configuration.")

def _enforce_size_limit(num_bytes: int):
    max_bytes = int(MAX_IMAGE_MB * 1024 * 1024)
    if num_bytes > max_bytes:
        raise HTTPException(status_code=413, detail=f"Image too large (> {MAX_IMAGE_MB} MB).")

# ---------------------------------- Routes -----------------------------------
@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _model is not None, "device": str(_device)}

@app.get("/labels")
def labels():
    return label_map

@app.get("/", response_class=HTMLResponse)
def root():
    if not ENABLE_UI:
        raise HTTPException(status_code=404, detail="UI disabled.")
    candidates = [BASE_DIR / "index.html", STATIC_DIR / "index.html"]
    for p in candidates:
        if p.exists():
            return FileResponse(p)
    raise HTTPException(
        status_code=500,
        detail="index.html not found. Place it next to app.py or at static/index.html."
    )

@app.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(default=None),
    size: int = Query(default=DEFAULT_IMG_SIZE, ge=128, le=512, description="Input resolution"),
    tta: Literal[2, 10] = Query(default=DEFAULT_TTA_CROPS, description="TTA crops: 2 or 10"),
):
    """
    multipart/form-data: field 'file' with an image.
    Query params:
      - size: input resolution (128..512), default 320.
      - tta: 2 (center+flip) or 10 (five-crop+flips), default 10.
    """
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

    x_batch = preprocess_image_to_tta(im, img_size=size, tta_crops=tta)
    return JSONResponse(infer_batch_tta(x_batch))

@app.post("/predict_base64")
async def predict_base64(
    payload: Dict[str, Any],
    size: int = Query(default=DEFAULT_IMG_SIZE, ge=128, le=512, description="Input resolution"),
    tta: Literal[2, 10] = Query(default=DEFAULT_TTA_CROPS, description="TTA crops: 2 or 10"),
):
    """
    application/json: { "image_base64": "data:image/png;base64,..." }
    Query params:
      - size: input resolution (128..512), default 320.
      - tta: 2 (center+flip) or 10 (five-crop+flips), default 10.
    """
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

    x_batch = preprocess_image_to_tta(im, img_size=size, tta_crops=tta)
    return JSONResponse(infer_batch_tta(x_batch))

# ------------------------------- Dev notes -----------------------------------
# Install:
#   pip install -U fastapi uvicorn pillow numpy torch torchvision timm opencv-python
#
# Run:
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# Check:
#   curl -F "file=@images/haby.jpg" "http://127.0.0.1:8000/predict?size=352&tta=10"
#
# Model checkpoint:
# - Put your PyTorch checkpoint at: model/enet_b2_7.pt (or change TFLITE_MODEL_PATH).
# - The checkpoint should output logits [N, 7] in the same class order as label_map.
#
# Labels:
# - To use a custom class order, set FORCE_DEFAULT_LABELS=False and provide:
#   model/label_map_multiclass.json with keys "0".."6".
# -----------------------------------------------------------------------------
