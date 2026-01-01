# app.py (REFRACTORED to use utils/io.py)
# Pipeline:
# 1) Upload (PNG/JPG/TIF)
# 2) Load image via utils.io.load_image_any()
# 3) U-Net++ tiling inference -> mask
# 4) Save pred_mask.png + overlay.png + step5_results.json via utils.io
# 5) Return JSON mapped to frontend

from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
from datetime import datetime
import logging
import json

import numpy as np
import cv2
import torch
import segmentation_models_pytorch as smp

from src.utils.io import (
    safe_filename,
    load_image_any,
    create_run_dir,
    build_run_paths,
    save_mask_png,
    save_overlay_png,
    save_json,
)

# -----------------------------
# Basic config
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("mangrove-app")

project_root = Path(__file__).parent.resolve()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = project_root / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = project_root / "models" / "unetpp_best.pth"

# Training tile size
TILE_H, TILE_W = 160, 160

# Tiling params (quality boost)
TILE_OVERLAP = 32      # try 16/32/64
BATCH_TILES = 24       # lower if MPS memory issues

# If PNG/JPG has no geo metadata
DEFAULT_PIXEL_SIZE_M = 0.7

# Carbon density placeholder (set later from literature)
DEFAULT_CARBON_DENSITY_TON_PER_HA = 150.0

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Model config
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None

# Model cache
model = None
model_in_channels = None


# -----------------------------
# Model loading
# -----------------------------
def _strip_module_prefix(state: dict) -> dict:
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        logger.info("Detected 'module.' prefix in state_dict. Stripping it.")
        return {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def _infer_in_channels_from_state_dict(state: dict) -> int:
    for k in ["encoder.conv1.weight", "encoder.model.conv1.weight", "encoder._conv1.weight"]:
        if k in state and hasattr(state[k], "shape"):
            logger.info(f"in_channels detected from key: {k}")
            return int(state[k].shape[1])

    for k, v in state.items():
        if hasattr(v, "shape") and len(v.shape) == 4 and v.shape[0] == 64 and v.shape[2] == 7 and v.shape[3] == 7:
            logger.info(f"in_channels detected from fallback key: {k}")
            return int(v.shape[1])

    return 3


def load_model_once() -> bool:
    global model, model_in_channels

    if model is not None:
        return True

    if not MODEL_PATH.exists():
        logger.error(f"Model not found: {MODEL_PATH}")
        return False

    logger.info(f"Loading model from: {MODEL_PATH}")
    state = torch.load(str(MODEL_PATH), map_location=DEVICE)

    # Handle wrapped checkpoints
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    if not isinstance(state, dict):
        logger.error("Unsupported checkpoint format. Expected state_dict dict.")
        return False

    state = _strip_module_prefix(state)
    model_in_channels = _infer_in_channels_from_state_dict(state)
    logger.info(f"Detected model in_channels: {model_in_channels}")

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=model_in_channels,
        classes=1,
        activation=None
    ).to(DEVICE)

    model.load_state_dict(state, strict=True)
    model.eval()

    logger.info("✅ Model loaded and ready.")
    return True


# -----------------------------
# Tiling inference
# -----------------------------
def _pad_to_tile(img: np.ndarray, tile_h: int, tile_w: int, stride: int):
    H, W = img.shape[:2]

    if H <= tile_h:
        pad_h = tile_h - H
    else:
        pad_h = (tile_h - (H - tile_h) % stride) % stride

    if W <= tile_w:
        pad_w = tile_w - W
    else:
        pad_w = (tile_w - (W - tile_w) % stride) % stride

    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return img_pad, H, W


def predict_mask_tiled(model_img: np.ndarray) -> np.ndarray:
    if model is None:
        raise RuntimeError("Model not loaded")

    stride = TILE_W - TILE_OVERLAP
    if stride <= 0:
        raise ValueError("TILE_OVERLAP must be < TILE_W")

    img_pad, H0, W0 = _pad_to_tile(model_img, TILE_H, TILE_W, stride)
    Hp, Wp = img_pad.shape[:2]

    prob_sum = np.zeros((Hp, Wp), dtype=np.float32)
    prob_cnt = np.zeros((Hp, Wp), dtype=np.float32)

    tiles, coords = [], []

    def run_batch(batch_tiles, batch_coords):
        if not batch_tiles:
            return
        x = np.stack(batch_tiles, axis=0)          # (B,H,W,C)
        x = np.transpose(x, (0, 3, 1, 2))          # (B,C,H,W)
        xt = torch.from_numpy(x).float().to(DEVICE)

        with torch.no_grad():
            logits = model(xt)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()  # (B,H,W)

        for p, (y, x0) in zip(probs, batch_coords):
            prob_sum[y:y + TILE_H, x0:x0 + TILE_W] += p
            prob_cnt[y:y + TILE_H, x0:x0 + TILE_W] += 1.0

    for y in range(0, Hp - TILE_H + 1, stride):
        for x0 in range(0, Wp - TILE_W + 1, stride):
            tiles.append(img_pad[y:y + TILE_H, x0:x0 + TILE_W, :])
            coords.append((y, x0))

            if len(tiles) >= BATCH_TILES:
                run_batch(tiles, coords)
                tiles, coords = [], []

    run_batch(tiles, coords)

    prob_avg = prob_sum / np.maximum(prob_cnt, 1e-6)
    prob_avg = prob_avg[:H0, :W0]

    return (prob_avg >= 0.5).astype(np.uint8)


# -----------------------------
# Step 5
# -----------------------------
def step5_calculate(mask01: np.ndarray, pixel_size_m: float, carbon_density_ton_per_ha: float) -> dict:
    total_pixels = int(mask01.size)
    mangrove_pixels = int(mask01.sum())

    coverage_percent = (mangrove_pixels / total_pixels) * 100.0 if total_pixels else 0.0

    pixel_area_m2 = float(pixel_size_m * pixel_size_m)
    area_m2 = float(mangrove_pixels * pixel_area_m2)
    area_ha = float(area_m2 / 10000.0)

    carbon_tons = float(area_ha * carbon_density_ton_per_ha)
    co2_tons = float(carbon_tons * 3.67)

    return {
        "pixel_size_m": float(pixel_size_m),
        "pixel_area_m2": float(pixel_area_m2),
        "mangrove_pixels": mangrove_pixels,
        "total_pixels": total_pixels,
        "coverage_percent": float(coverage_percent),
        "area_m2": float(area_m2),
        "area_ha": float(area_ha),
        "carbon_density_ton_per_ha": float(carbon_density_ton_per_ha),
        "carbon_tons": float(carbon_tons),
        "co2_tons": float(co2_tons),
    }


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/status")
def status():
    return jsonify({
        "status": "ready" if model is not None else "not_loaded",
        "device": DEVICE,
        "model_path": str(MODEL_PATH),
        "model_in_channels": model_in_channels
    })


@app.route("/upload", methods=["POST"])
def upload():
    if not load_model_once():
        return jsonify({"success": False, "error": "Model failed to load"}), 500

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    # optional inputs
    pixel_size_input = request.form.get("pixel_size", "").strip()
    carbon_density_input = request.form.get("carbon_density", "").strip()

    try:
        user_pixel_size = float(pixel_size_input) if pixel_size_input else None
    except:
        return jsonify({"success": False, "error": "pixel_size must be a number (e.g., 0.7)"}), 400

    try:
        carbon_density = float(carbon_density_input) if carbon_density_input else DEFAULT_CARBON_DENSITY_TON_PER_HA
    except:
        return jsonify({"success": False, "error": "carbon_density must be a number (e.g., 150)"}), 400

    # save upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_name = f"upload_{timestamp}_{safe_filename(file.filename)}"
    upload_path = app.config["UPLOAD_FOLDER"] / upload_name
    file.save(upload_path)

    try:
        # Use utils/io.py
        model_img, rgb_img, pixel_size_from_tif, tif_source = load_image_any(
            upload_path,
            model_in_channels=model_in_channels
        )

        # decide pixel size (tif beats user beats default)
        if pixel_size_from_tif is not None:
            pixel_size_m = pixel_size_from_tif
            pixel_size_note = "from_tif"
        elif user_pixel_size is not None:
            pixel_size_m = user_pixel_size
            pixel_size_note = "user_input"
        else:
            pixel_size_m = DEFAULT_PIXEL_SIZE_M
            pixel_size_note = "default"

        # inference
        mask01 = predict_mask_tiled(model_img)

        # run folder + standard paths via utils/io.py
        run_dir = create_run_dir(RESULTS_DIR, timestamp)
        paths = build_run_paths(run_dir)

        save_mask_png(mask01, paths["mask"])
        save_overlay_png(rgb_img, mask01, paths["overlay"])

        results = step5_calculate(mask01, pixel_size_m=pixel_size_m, carbon_density_ton_per_ha=carbon_density)
        save_json(results, paths["json"])

        response = {
            "success": True,
            "uploaded": f"/uploads/{upload_name}",
            "overlay": f"/results/run_{timestamp}/overlay.png",
            "mask": f"/results/run_{timestamp}/pred_mask.png",
            "json": f"/results/run_{timestamp}/step5_results.json",

            "coveragePercent": round(results["coverage_percent"], 2),
            "areaHectares": round(results["area_ha"], 4),
            "areaM2": round(results["area_m2"], 2),
            "carbonTons": round(results["carbon_tons"], 2),
            "carbonCO2": round(results["co2_tons"], 2),

            "pixel_size_m": results["pixel_size_m"],
            "pixel_size_source": pixel_size_note,
            "model_in_channels": model_in_channels,
            "warning": None
        }

        # warn if 4ch model but user used PNG/JPG (NIR padded zeros)
        if model_in_channels == 4 and upload_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            response["warning"] = (
                "Model expects 4 bands. PNG/JPG has 3 bands; NIR was padded with zeros (accuracy may drop). "
                "Use GeoTIFF for best results."
            )

        return jsonify(response)

    except Exception as e:
        logger.exception("Upload/inference failed")
        return jsonify({"success": False, "error": str(e)}), 500


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Mangrove Carbon Web App")
    logger.info("=" * 60)

    ok = load_model_once()
    if not ok:
        logger.error("❌ Model not loaded. Fix model path or checkpoint.")
    else:
        logger.info("✅ Open: http://localhost:5000")

    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)