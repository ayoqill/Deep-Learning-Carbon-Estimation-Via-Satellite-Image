# app.py (REFRACTORED to use utils/io.py)
# Pipeline:
# 1) Upload (PNG/JPG/TIF)
# 2) Load image via utils.io.load_image_any()
# 3) Model inference (U-Net++ or DeepLabV3+) with tiling
# 4) Save pred_mask.png + overlay.png + step5_results.json via utils.io
# 5) Return JSON mapped to frontend

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session, Response, make_response
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
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
from src.utils.analytics import AnalyticsManager
from src.utils.study_areas import StudyAreaManager

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
app.config["SECRET_KEY"] = "mangrove-carbon-detection-secret-2024"
app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 7  # 7 days
app.config["SESSION_COOKIE_SECURE"] = False  # Set to True in production with HTTPS
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth"

# In-memory user storage (in production, use a database)
users_db = {}


# User class for Flask-Login
class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username


@login_manager.user_loader
def load_user(username):
    if username in users_db:
        return User(username)
    return None


@app.before_request
def before_request():
    """Refresh user session to keep it alive"""
    session.permanent = True
    app.permanent_session_lifetime = app.config["PERMANENT_SESSION_LIFETIME"]


RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Analytics manager
analytics_manager = AnalyticsManager(RESULTS_DIR)

# Study areas manager
study_areas_manager = StudyAreaManager(
    study_areas_data_path=project_root / "TEST IMAGES",
    results_path=RESULTS_DIR,
    models={},  # Will be populated with loaded models
    device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Define multiple models
MODEL_PATHS = {
    "unetpp": project_root / "models" / "unetpp_best.pth",
    "deeplabv3": project_root / "models" / "deeplabv3" / "deeplabv3_best.pth"
}

# Training tile size
TILE_H, TILE_W = 160, 160

# Tiling params (quality boost)
TILE_OVERLAP = 32      # try 16/32/64
BATCH_TILES = 24       # lower if MPS memory issues

# If PNG/JPG has no geo metadata
DEFAULT_PIXEL_SIZE_M = 10.0

# Carbon density placeholder (set later from literature)
DEFAULT_CARBON_DENSITY_TON_PER_HA = 150.0

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Model config
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None

# Model cache (Dictionary to hold multiple models)
loaded_models = {}
model_in_channels = {}


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


def load_model(model_name: str) -> bool:
    global loaded_models, model_in_channels

    if model_name in loaded_models:
        return True

    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return False

    logger.info(f"Loading {model_name} from: {model_path}")
    state = torch.load(str(model_path), map_location=DEVICE)

    # Handle wrapped checkpoints
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    if not isinstance(state, dict):
        logger.error("Unsupported checkpoint format. Expected state_dict dict.")
        return False

    state = _strip_module_prefix(state)
    channels = _infer_in_channels_from_state_dict(state)
    model_in_channels[model_name] = channels
    logger.info(f"Detected {model_name} in_channels: {channels}")

    # Initialize correct architecture
    if model_name == "unetpp":
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=channels,
            classes=1,
            activation=None
        ).to(DEVICE)
    elif model_name == "deeplabv3":
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=channels,
            classes=1,
            activation=None
        ).to(DEVICE)
    else:
        logger.error(f"Unknown model: {model_name}")
        return False

    model.load_state_dict(state, strict=True)
    model.eval()
    
    loaded_models[model_name] = model

    logger.info(f"✅ {model_name} loaded and ready.")
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


def predict_mask_tiled(model_img: np.ndarray, model_name: str) -> np.ndarray:
    model = loaded_models.get(model_name)
    if model is None:
        raise RuntimeError(f"Model {model_name} not loaded")
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

    return prob_avg


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
@app.route("/login", methods=["POST"])
def login():
    """Handle user login"""
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password are required"}), 400

    if username not in users_db:
        return jsonify({"success": False, "error": "User not found"}), 401

    stored_hash = users_db[username]
    if not check_password_hash(stored_hash, password):
        return jsonify({"success": False, "error": "Invalid password"}), 401

    user = User(username)
    session.permanent = True
    login_user(user, remember=True, force=True)
    session.modified = True
    logger.info(f"User '{username}' logged in successfully")
    
    response = make_response(jsonify({"success": True, "username": username}))
    return response


@app.route("/signup", methods=["POST"])
def signup():
    """Handle user signup"""
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    confirm_password = data.get("confirm_password", "").strip()

    if not username or not password or not confirm_password:
        return jsonify({"success": False, "error": "All fields are required"}), 400

    if len(username) < 3:
        return jsonify({"success": False, "error": "Username must be at least 3 characters"}), 400

    if len(password) < 6:
        return jsonify({"success": False, "error": "Password must be at least 6 characters"}), 400

    if password != confirm_password:
        return jsonify({"success": False, "error": "Passwords do not match"}), 400

    if username in users_db:
        return jsonify({"success": False, "error": "Username already exists"}), 409

    # Store hashed password
    password_hash = generate_password_hash(password)
    users_db[username] = password_hash
    
    # Auto-login after signup
    user = User(username)
    session.permanent = True
    login_user(user, remember=True, force=True)
    session.modified = True
    logger.info(f"New user '{username}' signed up and logged in")
    response = make_response(jsonify({"success": True, "username": username}))
    return response


@app.route("/logout", methods=["POST"])
def logout():
    """Handle user logout"""
    if current_user.is_authenticated:
        username = current_user.username
        logout_user()
        logger.info(f"User '{username}' logged out")
    return jsonify({"success": True})


@app.route("/auth_status", methods=["GET"])
def auth_status():
    """Get current authentication status"""
    return jsonify({
        "authenticated": current_user.is_authenticated,
        "username": current_user.username if current_user.is_authenticated else None
    })


@app.route("/auth")
def auth():
    """Authentication page (login/signup)"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template("auth.html")


@app.route("/analytics")
def analytics():
    """Analytics page for result history and study areas"""
    if not current_user.is_authenticated:
        return redirect(url_for('auth'))
    return render_template("analytics.html")


@app.route("/")
def index():
    if not current_user.is_authenticated:
        return redirect(url_for('auth'))
    return render_template("index.html")


@app.route("/insight")
def insight():
    if not current_user.is_authenticated:
        return redirect(url_for('auth'))
    return render_template("insight.html")


@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/status")
def status():
    return jsonify({
        "status": "ready" if loaded_models else "not_loaded",
        "device": DEVICE,
        "loaded_models": list(loaded_models.keys()),
        "model_in_channels": model_in_channels
    })


@app.route("/upload", methods=["POST"])
def upload():
    if not current_user.is_authenticated:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    model_choice = request.form.get("model", "unetpp")  # default to unetpp
    
    if not load_model(model_choice):
        return jsonify({"success": False, "error": f"Model {model_choice} failed to load"}), 500

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
        current_channels = model_in_channels[model_choice]
        # Use utils/io.py
        model_img, rgb_img, pixel_size_from_tif, tif_source = load_image_any(
            upload_path,
            model_in_channels=current_channels
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

        # inference - returns probability map (float32)
        prob_map = predict_mask_tiled(model_img, model_choice)
        
        # Apply threshold to convert to binary mask
        DETECTION_THRESHOLD = 0.01
        mask01 = (prob_map >= DETECTION_THRESHOLD).astype(np.uint8)

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
            "used_model": model_choice,
            "model_in_channels": current_channels,
            "warning": None
        }

        # warn if 4ch model but user used PNG/JPG (NIR padded zeros)
        if current_channels == 4 and upload_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            response["warning"] = (
                "Model expects 4 bands. PNG/JPG has 3 bands; NIR was padded with zeros (accuracy may drop). "
                "Use GeoTIFF for best results."
            )

        # Save to analytics DB
        save_analysis_result(upload_name, timestamp, results, model_choice)

        return jsonify(response)

    except Exception as e:
        logger.exception("Upload/inference failed")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================
# Analytics API Routes
# ============================
@app.route("/api/analyses", methods=["GET"])
def get_analyses():
    """Get all analysis records"""
    if not current_user.is_authenticated:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        analyses = analytics_manager.get_all_analyses()
        return jsonify({"success": True, "data": analyses})
    except Exception as e:
        logger.exception("Failed to get analyses")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyses/<analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    """Get a specific analysis"""
    if not current_user.is_authenticated:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        analysis = analytics_manager.get_analysis(analysis_id)
        if not analysis:
            return jsonify({"success": False, "error": "Analysis not found"}), 404
        return jsonify({"success": True, "data": analysis})
    except Exception as e:
        logger.exception("Failed to get analysis")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/analyses/type/<analysis_type>", methods=["GET"])
def get_analyses_by_type(analysis_type):
    """Get analyses by type (uploaded/precomputed)"""
    if not current_user.is_authenticated:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    if analysis_type not in ["uploaded", "precomputed"]:
        return jsonify({"success": False, "error": "Invalid analysis type"}), 400
    
    try:
        analyses = analytics_manager.get_by_type(analysis_type)
        return jsonify({"success": True, "data": analyses})
    except Exception as e:
        logger.exception("Failed to get analyses by type")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/study-areas", methods=["GET"])
def get_study_areas():
    """Get all study areas"""
    if not current_user.is_authenticated:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        areas = study_areas_manager.get_study_areas()
        return jsonify({"success": True, "data": areas})
    except Exception as e:
        logger.exception("Failed to get study areas")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/study-areas/<study_area_id>", methods=["GET"])
def get_study_area(study_area_id):
    """Get a specific study area"""
    if not current_user.is_authenticated:
        return jsonify({"success": False, "error": "Not authenticated"}), 401
    
    try:
        area = study_areas_manager.get_study_area(study_area_id)
        if not area:
            return jsonify({"success": False, "error": "Study area not found"}), 404
        return jsonify({"success": True, "data": area})
    except Exception as e:
        logger.exception("Failed to get study area")
        return jsonify({"success": False, "error": str(e)}), 500


# ============================
# Helper: Save analysis to analytics DB after upload
# ============================
def save_analysis_result(upload_name, timestamp, results, model_choice):
    """
    Save analysis result to analytics manager after successful upload.
    """
    try:
        analysis = {
            "type": "uploaded",
            "title": upload_name.replace(f"upload_{timestamp}_", ""),
            "location": "",
            "originalImagePath": f"/uploads/{upload_name}",
            "resultImagePath": f"/results/run_{timestamp}/overlay.png",
            "maskPath": f"/results/run_{timestamp}/pred_mask.png",
            "model": model_choice,
            "mangroveCoverage": round(results["coverage_percent"], 2),
            "totalAreaHectares": round(results["area_ha"], 4),
            "totalAreaM2": round(results["area_m2"], 2),
            "carbonStock": round(results["carbon_tons"], 2),
            "co2Equivalent": round(results["co2_tons"], 2),
            "pixelSizeM": results["pixel_size_m"],
        }
        analytics_manager.save_analysis(analysis)
        logger.info(f"Analysis saved to analytics DB: {analysis['title']}")
    except Exception as e:
        logger.error(f"Failed to save analysis to analytics DB: {e}")


# Update the /upload route to save analysis
# We need to modify the response section-------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Mangrove Carbon Web App")
    logger.info("=" * 60)

# Run
# -----------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Mangrove Carbon Web App")
    logger.info("=" * 60)

    # Preload default model (optional)
    load_model("unetpp")
    
    logger.info("✅ Open: http://localhost:5000")

    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)