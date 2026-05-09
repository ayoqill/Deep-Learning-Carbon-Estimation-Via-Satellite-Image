# app.py
# Pipeline:
# 1) Upload (PNG/JPG/TIF)
# 2) Load image via utils.io.load_image_any()
# 3) Model inference with DeepLabV3+ using tiling
# 4) Save pred_mask.png + overlay.png + step5_results.json
# 5) Return JSON mapped to frontend
#
# Auth:
# - Admin account uses Render Environment Variables:
#   ADMIN_USERNAME
#   ADMIN_PASSWORD
#   SECRET_KEY
#
# - Normal users use SQLite temporary database:
#   app_data.db
#
# Note:
# On Render Free, SQLite may reset after restart/redeploy.

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
from datetime import datetime
import logging
import os
import requests

import numpy as np
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

# Read secret key from Render Environment Variables
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "local-dev-secret-key")

app.config["PERMANENT_SESSION_LIFETIME"] = 86400 * 7  # 7 days
app.config["SESSION_COOKIE_SECURE"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# SQLite temporary database
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{project_root / 'app_data.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# -----------------------------
# Flask-Login setup
# -----------------------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "auth"


# ============================
# Database Models
# ============================
class UserAccount(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(255), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.username}>"


class Analysis(db.Model):
    __tablename__ = "analyses"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    location = db.Column(db.String(255))
    original_image_path = db.Column(db.String(500))
    result_image_path = db.Column(db.String(500))
    mask_path = db.Column(db.String(500))
    model = db.Column(db.String(50))
    mangrove_coverage = db.Column(db.Float)
    total_area_hectares = db.Column(db.Float)
    total_area_m2 = db.Column(db.Float)
    carbon_stock = db.Column(db.Float)
    co2_equivalent = db.Column(db.Float)
    pixel_size_m = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "type": "uploaded",
            "title": self.title,
            "location": self.location or "",
            "originalImagePath": self.original_image_path,
            "resultImagePath": self.result_image_path,
            "maskPath": self.mask_path,
            "model": self.model,
            "mangroveCoverage": self.mangrove_coverage,
            "totalAreaHectares": self.total_area_hectares,
            "totalAreaM2": self.total_area_m2,
            "carbonStock": self.carbon_stock,
            "co2Equivalent": self.co2_equivalent,
            "pixelSizeM": self.pixel_size_m,
            "createdAt": self.created_at.isoformat() if self.created_at else None
        }


with app.app_context():
    db.create_all()
    logger.info("✅ SQLite database initialized")


# -----------------------------
# User class for Flask-Login
# -----------------------------
class User(UserMixin):
    def __init__(self, username):
        self.id = username
        self.username = username


@login_manager.user_loader
def load_user(username):
    """
    Allows:
    1. Admin from Render Environment Variables
    2. Normal users from SQLite
    """
    admin_username = os.getenv("ADMIN_USERNAME")

    if admin_username and username == admin_username:
        return User(username)

    user_account = UserAccount.query.filter_by(username=username).first()
    if user_account:
        return User(username)

    return None


@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = app.config["PERMANENT_SESSION_LIFETIME"]


# -----------------------------
# Folders
# -----------------------------
RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

analytics_manager = AnalyticsManager(RESULTS_DIR)

study_areas_manager = StudyAreaManager(
    study_areas_data_path=project_root / "TEST IMAGES",
    results_path=RESULTS_DIR,
    models={},
    device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
)


# ============================
# Model Configuration & Download
# ============================
MODEL_PATH = os.path.join(
    os.getcwd(),
    "models",
    "deeplabv3",
    "deeplabv3_best.pth"
)

MODEL_URL = "https://huggingface.co/aqllaimaa/deeplabv3-mangrove/resolve/main/deeplabv3_best.pth"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    logger.info("Downloading model from HuggingFace...")

    try:
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        logger.info("✅ Model downloaded successfully.")

    except Exception as e:
        logger.error(f"❌ Failed to download model from HuggingFace: {e}")
        raise


# Training tile size
TILE_H, TILE_W = 160, 160

# Tiling params
TILE_OVERLAP = 32
BATCH_TILES = 24

# If PNG/JPG has no geo metadata
DEFAULT_PIXEL_SIZE_M = 10.0

# Carbon density placeholder
DEFAULT_CARBON_DENSITY_TON_PER_HA = 150.0

# Device
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Model config
ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = None

# Model cache
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


def load_model(model_name: str = "deeplabv3") -> bool:
    global loaded_models, model_in_channels

    if model_name in loaded_models:
        return True

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return False

    logger.info(f"Loading DeepLabV3+ from: {MODEL_PATH}")

    state = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    if not isinstance(state, dict):
        logger.error("Unsupported checkpoint format. Expected state_dict dict.")
        return False

    state = _strip_module_prefix(state)
    channels = _infer_in_channels_from_state_dict(state)

    model_in_channels["deeplabv3"] = channels
    logger.info(f"Detected DeepLabV3+ in_channels: {channels}")

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=channels,
        classes=1,
        activation=None
    ).to(DEVICE)

    model.load_state_dict(state, strict=True)
    model.eval()

    loaded_models["deeplabv3"] = model

    logger.info("✅ DeepLabV3+ loaded and ready.")
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

    tiles = []
    coords = []

    def run_batch(batch_tiles, batch_coords):
        if not batch_tiles:
            return

        x = np.stack(batch_tiles, axis=0)
        x = np.transpose(x, (0, 3, 1, 2))
        xt = torch.from_numpy(x).float().to(DEVICE)

        with torch.no_grad():
            logits = model(xt)
            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()

        for p, (y, x0) in zip(probs, batch_coords):
            prob_sum[y:y + TILE_H, x0:x0 + TILE_W] += p
            prob_cnt[y:y + TILE_H, x0:x0 + TILE_W] += 1.0

    for y in range(0, Hp - TILE_H + 1, stride):
        for x0 in range(0, Wp - TILE_W + 1, stride):
            tiles.append(img_pad[y:y + TILE_H, x0:x0 + TILE_W, :])
            coords.append((y, x0))

            if len(tiles) >= BATCH_TILES:
                run_batch(tiles, coords)
                tiles = []
                coords = []

    run_batch(tiles, coords)

    prob_avg = prob_sum / np.maximum(prob_cnt, 1e-6)
    prob_avg = prob_avg[:H0, :W0]

    return prob_avg


# -----------------------------
# Step 5 calculation
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
# Auth Routes
# -----------------------------
@app.route("/login", methods=["POST"])
def login():
    """
    Login flow:
    1. Check admin account from Render Environment Variables
    2. If not admin, check normal SQLite users
    """
    data = request.get_json() or {}

    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({
            "success": False,
            "error": "Username and password are required"
        }), 400

    admin_username = os.getenv("ADMIN_USERNAME")
    admin_password = os.getenv("ADMIN_PASSWORD")

    # 1. Admin login from Render Environment Variables
    if admin_username and admin_password:
        if username == admin_username and password == admin_password:
            user = User(username)
            session.permanent = True
            login_user(user, remember=True, force=True)
            session.modified = True

            logger.info(f"Admin user '{username}' logged in successfully")

            return jsonify({
                "success": True,
                "username": username
            })

    # 2. Normal user login from SQLite
    user_account = UserAccount.query.filter_by(username=username).first()

    if not user_account:
        return jsonify({
            "success": False,
            "error": "User not found"
        }), 401

    if not check_password_hash(user_account.password_hash, password):
        return jsonify({
            "success": False,
            "error": "Invalid password"
        }), 401

    user = User(username)
    session.permanent = True
    login_user(user, remember=True, force=True)
    session.modified = True

    logger.info(f"User '{username}' logged in successfully")

    return jsonify({
        "success": True,
        "username": username
    })


@app.route("/signup", methods=["POST"])
def signup():
    """
    Signup is only for normal temporary users.
    Admin account is not saved in SQLite.
    """
    data = request.get_json() or {}

    username = data.get("username", "").strip()
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    confirm_password = data.get("confirm_password", "").strip()

    admin_username = os.getenv("ADMIN_USERNAME")

    if not username or not password or not confirm_password:
        return jsonify({
            "success": False,
            "error": "All required fields are required"
        }), 400

    if len(username) < 3:
        return jsonify({
            "success": False,
            "error": "Username must be at least 3 characters"
        }), 400

    if len(password) < 6:
        return jsonify({
            "success": False,
            "error": "Password must be at least 6 characters"
        }), 400

    if password != confirm_password:
        return jsonify({
            "success": False,
            "error": "Passwords do not match"
        }), 400

    if admin_username and username == admin_username:
        return jsonify({
            "success": False,
            "error": "This username is reserved for admin"
        }), 409

    if UserAccount.query.filter_by(username=username).first():
        return jsonify({
            "success": False,
            "error": "Username already exists"
        }), 409

    password_hash = generate_password_hash(password)

    new_user = UserAccount(
        username=username,
        email=email if email else None,
        password_hash=password_hash
    )

    db.session.add(new_user)
    db.session.commit()

    user = User(username)
    session.permanent = True
    login_user(user, remember=True, force=True)
    session.modified = True

    logger.info(f"New normal user '{username}' signed up and logged in")

    return jsonify({
        "success": True,
        "username": username
    })


@app.route("/logout", methods=["GET", "POST"])
def logout():
    if current_user.is_authenticated:
        username = current_user.username
        logout_user()
        logger.info(f"User '{username}' logged out")

    if request.method == "GET":
        return redirect(url_for("auth"))

    return jsonify({"success": True})


@app.route("/auth_status", methods=["GET"])
def auth_status():
    return jsonify({
        "authenticated": current_user.is_authenticated,
        "username": current_user.username if current_user.is_authenticated else None
    })


@app.route("/auth")
def auth():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    return render_template("auth.html")


# -----------------------------
# Page Routes
# -----------------------------
@app.route("/")
def index():
    if not current_user.is_authenticated:
        return redirect(url_for("auth"))

    return render_template("index.html")


@app.route("/analytics")
def analytics():
    if not current_user.is_authenticated:
        return redirect(url_for("auth"))

    return render_template("analytics.html")


@app.route("/insight")
def insight():
    if not current_user.is_authenticated:
        return redirect(url_for("auth"))

    return render_template("insight.html")


# -----------------------------
# Static file serving
# -----------------------------
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


# -----------------------------
# Upload Route
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    if not current_user.is_authenticated:
        return jsonify({
            "success": False,
            "error": "Not authenticated"
        }), 401

    model_choice = "deeplabv3"

    if not load_model(model_choice):
        return jsonify({
            "success": False,
            "error": f"Model {model_choice} failed to load"
        }), 500

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image provided"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400

    pixel_size_input = request.form.get("pixel_size", "").strip()
    carbon_density_input = request.form.get("carbon_density", "").strip()

    try:
        user_pixel_size = float(pixel_size_input) if pixel_size_input else None
    except Exception:
        return jsonify({
            "success": False,
            "error": "pixel_size must be a number, for example 10"
        }), 400

    try:
        carbon_density = float(carbon_density_input) if carbon_density_input else DEFAULT_CARBON_DENSITY_TON_PER_HA
    except Exception:
        return jsonify({
            "success": False,
            "error": "carbon_density must be a number, for example 150"
        }), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_name = f"upload_{timestamp}_{safe_filename(file.filename)}"
    upload_path = app.config["UPLOAD_FOLDER"] / upload_name

    file.save(upload_path)

    try:
        current_channels = model_in_channels[model_choice]

        model_img, rgb_img, pixel_size_from_tif, tif_source = load_image_any(
            upload_path,
            model_in_channels=current_channels
        )

        if pixel_size_from_tif is not None:
            pixel_size_m = pixel_size_from_tif
            pixel_size_note = "from_tif"
        elif user_pixel_size is not None:
            pixel_size_m = user_pixel_size
            pixel_size_note = "user_input"
        else:
            pixel_size_m = DEFAULT_PIXEL_SIZE_M
            pixel_size_note = "default"

        prob_map = predict_mask_tiled(model_img, model_choice)

        DETECTION_THRESHOLD = 0.001
        mask01 = (prob_map >= DETECTION_THRESHOLD).astype(np.uint8)

        run_dir = create_run_dir(RESULTS_DIR, timestamp)
        paths = build_run_paths(run_dir)

        save_mask_png(mask01, paths["mask"])
        save_overlay_png(rgb_img, mask01, paths["overlay"])

        results = step5_calculate(
            mask01,
            pixel_size_m=pixel_size_m,
            carbon_density_ton_per_ha=carbon_density
        )

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

        if current_channels == 4 and upload_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            response["warning"] = (
                "Model expects 4 bands. PNG/JPG has 3 bands; NIR was padded with zeros. "
                "Use GeoTIFF for best results."
            )

        save_analysis_result(
            upload_name=upload_name,
            timestamp=timestamp,
            results=results,
            model_choice=model_choice,
            username=current_user.username
        )

        return jsonify(response)

    except Exception as e:
        logger.exception("Upload/inference failed")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================
# Analytics API Routes
# ============================
@app.route("/api/analyses", methods=["GET"])
def get_analyses():
    if not current_user.is_authenticated:
        return jsonify({
            "success": False,
            "error": "Not authenticated"
        }), 401

    try:
        analyses = Analysis.query.filter_by(
            username=current_user.username
        ).order_by(
            Analysis.created_at.desc()
        ).all()

        data = [analysis.to_dict() for analysis in analyses]

        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.exception("Failed to get analyses")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/analyses/<int:analysis_id>", methods=["GET"])
def get_analysis(analysis_id):
    if not current_user.is_authenticated:
        return jsonify({
            "success": False,
            "error": "Not authenticated"
        }), 401

    try:
        analysis = Analysis.query.filter_by(
            id=analysis_id,
            username=current_user.username
        ).first()

        if not analysis:
            return jsonify({
                "success": False,
                "error": "Analysis not found"
            }), 404

        return jsonify({
            "success": True,
            "data": analysis.to_dict()
        })

    except Exception as e:
        logger.exception("Failed to get analysis")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/analyses/<int:analysis_id>", methods=["DELETE"])
def delete_analysis_api(analysis_id):
    if not current_user.is_authenticated:
        return jsonify({
            "success": False,
            "error": "Not authenticated"
        }), 401

    try:
        analysis = Analysis.query.filter_by(
            id=analysis_id,
            username=current_user.username
        ).first()

        if not analysis:
            return jsonify({
                "success": False,
                "error": "Analysis not found"
            }), 404

        db.session.delete(analysis)
        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Analysis deleted successfully"
        })

    except Exception as e:
        logger.exception("Failed to delete analysis")
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/analyses/type/<analysis_type>", methods=["GET"])
def get_analyses_by_type(analysis_type):
    if not current_user.is_authenticated:
        return jsonify({
            "success": False,
            "error": "Not authenticated"
        }), 401

    if analysis_type not in ["uploaded", "precomputed"]:
        return jsonify({
            "success": False,
            "error": "Invalid analysis type"
        }), 400

    try:
        if analysis_type == "uploaded":
            analyses = Analysis.query.filter_by(
                username=current_user.username
            ).order_by(
                Analysis.created_at.desc()
            ).all()

            data = [analysis.to_dict() for analysis in analyses]
        else:
            data = []

        return jsonify({
            "success": True,
            "data": data
        })

    except Exception as e:
        logger.exception("Failed to get analyses by type")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/study-areas", methods=["GET"])
def get_study_areas():
    if not current_user.is_authenticated:
        return jsonify({
            "success": False,
            "error": "Not authenticated"
        }), 401

    try:
        areas = study_areas_manager.get_study_areas()

        return jsonify({
            "success": True,
            "data": areas
        })

    except Exception as e:
        logger.exception("Failed to get study areas")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/study-areas/<study_area_id>", methods=["GET"])
def get_study_area(study_area_id):
    if not current_user.is_authenticated:
        return jsonify({
            "success": False,
            "error": "Not authenticated"
        }), 401

    try:
        area = study_areas_manager.get_study_area(study_area_id)

        if not area:
            return jsonify({
                "success": False,
                "error": "Study area not found"
            }), 404

        return jsonify({
            "success": True,
            "data": area
        })

    except Exception as e:
        logger.exception("Failed to get study area")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================
# Helper: Save analysis to SQLite
# ============================
def save_analysis_result(upload_name, timestamp, results, model_choice, username=None):
    try:
        title = upload_name.replace(f"upload_{timestamp}_", "")

        analysis = Analysis(
            username=username,
            title=title,
            location="",
            original_image_path=f"/uploads/{upload_name}",
            result_image_path=f"/results/run_{timestamp}/overlay.png",
            mask_path=f"/results/run_{timestamp}/pred_mask.png",
            model=model_choice,
            mangrove_coverage=round(results["coverage_percent"], 2),
            total_area_hectares=round(results["area_ha"], 4),
            total_area_m2=round(results["area_m2"], 2),
            carbon_stock=round(results["carbon_tons"], 2),
            co2_equivalent=round(results["co2_tons"], 2),
            pixel_size_m=results["pixel_size_m"],
        )

        db.session.add(analysis)
        db.session.commit()

        logger.info(f"Analysis saved for user '{username}': {title}")

    except Exception as e:
        logger.error(f"Failed to save analysis: {e}")
        db.session.rollback()


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting Mangrove Carbon Web App")
    logger.info("=" * 60)

    load_model("deeplabv3")

    logger.info("✅ Open: http://localhost:5000")

    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)