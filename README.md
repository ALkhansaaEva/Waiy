# Waiy - Happy vs Sad Emotion Detection API (TFLite)

> **Developed by:** Al-khansaa Dabwan

---

## Overview

Waiy is a lightweight FastAPI-based REST API for emotion detection that classifies images into **Happy** or **Sad** categories using a TensorFlow Lite (TFLite) model. The API supports image uploads both as file uploads and base64-encoded strings and returns confidence scores for the detected emotions.

Additionally, Waiy offers a user-friendly **Arabic interface** for easy testing without requiring any programming knowledge.

---

## Features

- Fast, lightweight emotion classification using TFLite
- Accepts images as multipart file uploads or base64 strings
- Supports common image formats: PNG, JPEG, BMP, WEBP
- Configurable maximum upload size (default: 10 MB)
- Built-in CORS middleware for flexible frontend integration
- Optional interactive web UI localized in Arabic
- API documentation available with OpenAPI (Swagger)
- Cross-platform: works on Windows, macOS, Linux
- Simple setup and deployment

---

## Requirements

- Python 3.8 or newer
- `fastapi`
- `uvicorn`
- `pillow`
- `numpy`
- `tflite-runtime` *(preferred)* or `tensorflow==2.14.0` as fallback

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ALkhansaaEva/Waiy.git
cd waiy
````

### 2. Create and activate a virtual environment

#### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### macOS / Linux (bash/zsh)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -U pip
pip install fastapi uvicorn pillow numpy
```

#### Install tensorflow (preferred):

```bash
pip install tensorflow
```

### 4. Prepare model and label files

Ensure the following files are present in the `model/` directory:

* `emotion_binary_final.tflite` — The TFLite model file
* `label_map_binary.json` — JSON label map (e.g., `{ "0": "Sad", "1": "Happy" }`)

### 5. Run the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## Usage

### Web UI (Arabic Interface)

Open your browser and visit:
[http://localhost:8000/](http://localhost:8000/)

Upload images directly from your device and get emotion predictions with an Arabic-friendly interface.

### API Endpoints

* **Health check**
  `GET /health`
  Returns server and model status.

* **Get label map**
  `GET /labels`
  Returns available emotion labels.

* **Predict emotion (file upload)**
  `POST /predict`
  Send a multipart/form-data request with field `file` containing the image.

* **Predict emotion (base64 JSON)**
  `POST /predict_base64`
  Send JSON:

  ```json
  {
    "image_base64": "data:image/png;base64,..."
  }
  ```

* **API documentation (Swagger UI)**
  `GET /docs` (if enabled)

---

## Configuration

Modify these constants in `app.py` to customize behavior:

| Variable             | Description                                | Default                             |
| -------------------- | ------------------------------------------ | ----------------------------------- |
| `TFLITE_MODEL_PATH`  | Path to the TFLite model file              | `model/emotion_binary_final.tflite` |
| `LABEL_MAP_PATH`     | Path to the label map JSON file            | `model/label_map_binary.json`       |
| `ENABLE_API`         | Enable or disable API routes (`/predict`)  | `True`                              |
| `ENABLE_UI`          | Enable or disable the Arabic UI at `/`     | `True`                              |
| `ENABLE_DOCS`        | Enable or disable OpenAPI docs (`/docs`)   | `True`                              |
| `MAX_IMAGE_MB`       | Maximum allowed image size in megabytes    | `10.0`                              |
| `CORS_ALLOW_ORIGINS` | Allowed CORS origins (use `["*"]` for all) | `["*"]`                             |

---

## Arabic UI Highlights

* Fully localized UI with Arabic labels and instructions
* Drag and drop or select images from device for quick testing
* Instant emotion prediction with confidence displayed
* Clean, accessible layout optimized for Arabic readers

---

## Supported Platforms

* **Windows 10/11**
* **macOS (Catalina or later)**
* **Linux distributions** (Ubuntu, Debian, Fedora, etc.)

---

## Troubleshooting

* **Model file not found:** Check your `model/` directory and the `TFLITE_MODEL_PATH` setting.
* **TFLite runtime installation issues:**
  Try installing the official TensorFlow package as fallback.
* **Large image upload errors:** Increase `MAX_IMAGE_MB` or resize images before uploading.
* **Server errors:** Run with `--reload` during development to auto-reload and check logs.

---

## License

This project is licensed under the MIT License.

---

*Thank you for using Waiy! 🎉*