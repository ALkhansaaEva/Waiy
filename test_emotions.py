# test_emotions_hq.py
# -----------------------------------------------------------------------------
# High-quality test script for emotion recognition using a timm EfficientNet
# checkpoint (.pt). Keeps your original public names:
#   - EmotiEffLibRecognizerBase
#   - extract_features
#   - classify_emotions
#   - test_images_in_folder
#
# Upgrades:
#   - Higher input resolution (default 320; CLI-configurable).
#   - Face detection (Haar) + smart margin crop (+ fallback center square).
#   - Lighting/contrast stabilization (CLAHE) + Unsharp Mask sharpening.
#   - Gentle denoising (bilateral filter) for noisy sources.
#   - Robust RGB/RGBA/gray handling.
#   - TTA: FiveCrop + horizontal flips (10 crops) at high resolution.
#   - Optional visualization: annotated full image + hi-res face crop.
#   - Prints Top-3 probabilities.
#
# Labels updated as requested:
#   {0: Anger, 1: Disgust, 2: Fear, 3: Happy, 4: Neutral, 5: Sad, 6: Surprise}
#
# Usage:
#   python test_emotions_hq.py --folder images --model model/enet_b2_7.pt \
#       --name enet_b2_7 --img-size 320 --save-viz --out-dir outputs
#
# Requirements:
#   pip install torch torchvision timm opencv-python pillow numpy
# -----------------------------------------------------------------------------

import os
import argparse
from typing import List, Tuple

import torch
import timm  # ensure timm is installed
import cv2
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF

# Allow unpickling the timm EfficientNet class if your checkpoint needs it.
torch.serialization.add_safe_globals([timm.models.efficientnet.EfficientNet])

# ----------------------- Face utilities (OpenCV Haar) ------------------------
_HAAR_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

def _largest_face_bbox(gray: np.ndarray):
    """Return (x,y,w,h) of the largest detected face, or None if none found."""
    faces = _HAAR_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    areas = [w * h for (x, y, w, h) in faces]
    return faces[int(np.argmax(areas))]

def _crop_with_margin(img: np.ndarray, bbox, margin: float = 0.35) -> np.ndarray:
    """
    Crop face with extra margin; clamps to image bounds.
    margin is a fraction of max(w, h).
    """
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2.0, y + bh / 2.0
    m = margin * max(bw, bh)
    new_w, new_h = int(bw + 2 * m), int(bh + 2 * m)
    nx = int(max(0, cx - new_w / 2))
    ny = int(max(0, cy - new_h / 2))
    nx2 = int(min(w, nx + new_w))
    ny2 = int(min(h, ny + new_h))
    return img[ny:ny2, nx:nx2], (nx, ny, nx2 - nx, ny2 - ny)

def _center_square_crop(img: np.ndarray) -> np.ndarray:
    """Fallback crop: centered square region when no face is detected."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side], (x0, y0, side, side)

def _apply_clahe_rgb(img_rgb: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """Apply CLAHE on L channel (LAB) to stabilize contrast/lighting."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

def _unsharp_mask(img_rgb: np.ndarray, k: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """
    Simple Unsharp Mask to emphasize facial features (eyes, mouth edges).
    k: sharpening amount; sigma: blur radius.
    """
    blur = cv2.GaussianBlur(img_rgb, (0, 0), sigma)
    sharp = cv2.addWeighted(img_rgb, 1 + k, blur, -k, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)

def _denoise_gentle(img_rgb: np.ndarray) -> np.ndarray:
    """Gentle bilateral denoising to reduce compression noise without killing edges."""
    return cv2.bilateralFilter(img_rgb, d=5, sigmaColor=35, sigmaSpace=35)

# ------------------------------ Model wrapper --------------------------------
class EmotiEffLibRecognizerBase:
    def __init__(self, model_path: str, img_size: int = 320) -> None:
        self.model_path = model_path
        model_name = os.path.basename(model_path)

        # Labels as requested
        self.idx_to_emotion_class = {
            0: "Anger",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Neutral",
            5: "Sad",
            6: "Surprise",
        }

        # ImageNet normalization (expected by EfficientNet/timm)
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.img_size = int(img_size)
        self.device = torch.device("cpu")
        self.model = self._load_model(model_path)
        self.model.eval()

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def _load_model(self, model_path: str):
        """Load torch model checkpoint safely on CPU."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        model = torch.load(model_path, map_location=self.device, weights_only=False)
        if not hasattr(model, "forward"):
            raise RuntimeError("Loaded object is not a torch.nn.Module")
        model.eval()
        return model

    # -------------------------- Face-aware preprocess -------------------------
    def _detect_and_crop_face(self, img_rgb: np.ndarray):
        """Detect largest face; crop with margin; fallback to center square."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        bbox = _largest_face_bbox(gray)
        if bbox is not None:
            face, used = _crop_with_margin(img_rgb, bbox, margin=0.35)
        else:
            face, used = _center_square_crop(img_rgb)
        return face, used

    def _to_pil_rgb(self, img: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL RGB; handle grayscale/RGBA robustly."""
        if img.ndim == 2:  # grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:  # RGBA -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return Image.fromarray(img)

    def _tta_crops(self, pil_img: Image.Image) -> List[Image.Image]:
        """
        Create 10 crops for TTA at the desired resolution:
        - FiveCrop (img_size): TL, TR, BL, BR, Center
        - Horizontal flip of each (another 5)
        """
        # Slightly upscale before five-crop to reduce border artifacts
        resized = ImageOps.fit(pil_img, (self.img_size + 32, self.img_size + 32), method=Image.BILINEAR)
        tl, tr, bl, br, center = TF.five_crop(resized, self.img_size)
        crops = [tl, tr, bl, br, center]
        flips = [TF.hflip(c) for c in crops]
        return crops + flips  # 10 crops

    def _preprocess(self, img_rgb: np.ndarray) -> Tuple[torch.Tensor, dict]:
        """
        Face-aware preprocessing + denoise + CLAHE + Unsharp + TTA.
        Returns:
          - torch.Tensor [N, 3, img_size, img_size] (N=10)
          - meta dict with 'bbox' and 'face_rgb' for visualization
        """
        # 0) Gentle denoise (optional but helps with low-quality PNG/JPG)
        img_rgb = _denoise_gentle(img_rgb)

        # 1) Face-aware crop
        face_rgb, used_bbox = self._detect_and_crop_face(img_rgb)

        # 2) Contrast stabilization
        face_rgb = _apply_clahe_rgb(face_rgb, clip=2.0)

        # 3) Sharpen facial features
        face_rgb = _unsharp_mask(face_rgb, k=0.8, sigma=1.0)

        # 4) PIL + TTA crops
        pil_face = self._to_pil_rgb(face_rgb)
        crops = self._tta_crops(pil_face)
        batch = torch.stack([self.to_tensor_norm(c) for c in crops], dim=0)  # [10,3,S,S]

        meta = {"bbox_used": used_bbox, "face_rgb": face_rgb}
        return batch, meta

    def classify_emotions(self, features: torch.Tensor) -> Tuple[str, float, np.ndarray]:
        """
        Run the model. `features` is [N,3,S,S] (TTA batch).
        We aggregate logits across crops, then softmax once.
        Returns:
          - label (str), confidence (float), probs (numpy array [C])
        """
        with torch.no_grad():
            features = features.to(self.device, dtype=torch.float32)
            logits = self.model(features)  # [N, C]
            if logits.ndim != 2:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
            logits_mean = logits.mean(dim=0, keepdim=True)  # [1, C]
            probs = torch.softmax(logits_mean, dim=1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        emotion = self.idx_to_emotion_class.get(idx, str(idx))
        confidence = float(probs[idx])
        return emotion, confidence, probs

    def extract_features(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Prepare TTA batch from an RGB image array.
        (Keeps original name/signature.)
        """
        batch, _ = self._preprocess(face_img)  # [10,3,S,S]
        return batch

# ------------------------------- Test harness --------------------------------
def _draw_annotation(full_bgr: np.ndarray, bbox, label: str, conf: float) -> np.ndarray:
    """Draw face bbox and label on the full image for visualization."""
    vis = full_bgr.copy()
    x, y, w, h = bbox
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
    text = f"{label} ({conf*100:.1f}%)"
    cv2.rectangle(vis, (x, y - 28), (x + max(160, len(text)*9), y), (0, 200, 255), -1)
    cv2.putText(vis, text, (x + 6, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2, cv2.LINE_AA)
    return vis

def test_images_in_folder(folder_path: str, model_path: str, model_name: str,
                          img_size: int = 320, save_viz: bool = False, out_dir: str = "outputs") -> None:
    """
    Batch test emotion recognition on all images in folder.
    Keeps the same name/signature plus optional args for HQ testing.
    """
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist!")
        return

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]
    if not image_files:
        print(f"No images found in folder '{folder_path}'!")
        return

    os.makedirs(out_dir, exist_ok=True)
    recognizer = EmotiEffLibRecognizerBase(model_path=model_path, img_size=img_size)

    print(f"\nTesting with model: {model_name} (img_size={img_size})")
    for image_file in sorted(image_files):
        img_path = os.path.join(folder_path, image_file)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            print(f"Failed to load image {image_file}")
            continue

        # Normalize to RGB ndarray
        if img_bgr.ndim == 2:  # grayscale
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            if img_bgr.shape[2] == 4:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
            else:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Preprocess (returns batch + meta for viz)
        try:
            batch, meta = recognizer._preprocess(img_rgb)         # [10,3,S,S], {'bbox_used','face_rgb'}
            label, conf, probs = recognizer.classify_emotions(batch)

            # Top-3
            top3_idx = np.argsort(probs)[-3:][::-1]
            top3 = [(recognizer.idx_to_emotion_class.get(int(i), str(int(i))), float(probs[i])) for i in top3_idx]

            print(f"  Image: {image_file}")
            print(f"    Predicted Emotion: {label}")
            print(f"    Confidence: {conf * 100:.2f}%")
            print("    Top-3:", ", ".join([f"{k}:{v*100:.1f}%" for k,v in top3]), "\n")

            # Visualization (optional)
            if save_viz:
                full_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                vis = _draw_annotation(full_bgr, meta["bbox_used"], label, conf)
                face_bgr = cv2.cvtColor(meta["face_rgb"], cv2.COLOR_RGB2BGR)

                base = os.path.splitext(image_file)[0]
                cv2.imwrite(os.path.join(out_dir, f"{base}_annotated.jpg"), vis)
                cv2.imwrite(os.path.join(out_dir, f"{base}_face.jpg"), face_bgr)

        except Exception as e:
            print(f"  Image: {image_file}")
            print(f"    ERROR: {e}\n")

# ----------------------------------- CLI -------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(description="High-quality emotion test runner (timm EfficientNet).")
    p.add_argument("--folder", type=str, default="images", help="Path to images folder")
    p.add_argument("--model", type=str, default="model/enet_b2_7.pt", help="Path to .pt checkpoint")
    p.add_argument("--name", type=str, default="enet_b2_7", help="Model name for printing")
    p.add_argument("--img-size", type=int, default=320, help="Input resolution (try 288/320/352)")
    p.add_argument("--save-viz", action="store_true", help="Save annotated images and face crops")
    p.add_argument("--out-dir", type=str, default="outputs", help="Where to save visualizations")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    test_images_in_folder(
        folder_path=args.folder,
        model_path=args.model,
        model_name=args.name,
        img_size=args.img_size,
        save_viz=args.save_viz,
        out_dir=args.out_dir,
    )
