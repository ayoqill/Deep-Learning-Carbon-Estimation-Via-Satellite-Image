import json
from pathlib import Path

import numpy as np
import cv2
import rasterio
import torch
import segmentation_models_pytorch as smp


#  KIV DULUUUUUUUUU

# -------------------------
# SETTINGS
# -------------------------
TILE = 157           # your training tile size
BATCH = 64           # lower if memory issue (16/32)
THRESH = 0.5         # mask threshold
ALPHA = 0.70         # red overlay strength (0.6–0.8)

MODEL_PATH = Path("models/unetpp_best.pth")
INPUT_TIF  = Path("data/region/TYPESAMPLEHEREEEEEEEE.tif")
OUT_DIR    = Path("data/outputs")

OUT_OVERLAY = OUT_DIR / "stitched_overlay.png"
OUT_STATS   = OUT_DIR / "stats.json"

# Mac M-series device
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# -------------------------
# Helpers
# -------------------------
def build_model_4band():
    # MUST match your training model settings:
    # - encoder_name must match what you used in training
    # - in_channels=4 because RGB+NIR
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",   # change if your training used another encoder
        encoder_weights=None,
        in_channels=4,
        classes=1
    )
    return model


def load_model():
    model = build_model_4band().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def normalize_tile(tile_chw):
    """
    Keep it consistent with training.
    If you used a specific normalization during training, copy it here.
    This generic min-max per tile works OK for many SR/TOA tiles.
    """
    x = tile_chw.astype(np.float32)
    mn = np.min(x)
    mx = np.max(x)
    return (x - mn) / (mx - mn + 1e-8)


def to_display_rgb(img_chw_4):
    """
    Display only RGB for overlay (ignore NIR).
    Use percentile stretch so it looks nice.
    Assumption: bands are [B, G, R, NIR] OR [R,G,B,NIR] depending on product.
    Your previous visualization suggested band2=Red, band3=NIR.
    Most common multispectral order is B,G,R,NIR.
    We'll treat it as B,G,R,NIR for display.
    """
    b = img_chw_4[0].astype(np.float32)
    g = img_chw_4[1].astype(np.float32)
    r = img_chw_4[2].astype(np.float32)

    rgb = np.stack([r, g, b], axis=-1)  # HWC RGB

    out = np.zeros_like(rgb, dtype=np.uint8)
    for i in range(3):
        ch = rgb[:, :, i]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi - lo < 1e-6:
            out[:, :, i] = 0
        else:
            out[:, :, i] = (np.clip((ch - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return out


def draw_red_fill(rgb_uint8, mask_uint8, alpha=0.7):
    overlay = rgb_uint8.copy()
    red = np.zeros_like(overlay, dtype=np.uint8)
    red[:, :, 0] = 255  # R channel
    m = mask_uint8 > 0
    overlay[m] = (overlay[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return overlay


def predict_stitched_mask(model, img_chw_4):
    """
    Tile -> predict -> stitch (returns full-size mask uint8 0/255).
    """
    _, H, W = img_chw_4.shape

    # pad to multiple of TILE
    pad_h = (TILE - (H % TILE)) % TILE
    pad_w = (TILE - (W % TILE)) % TILE
    img_pad = np.pad(img_chw_4, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    Hp, Wp = img_pad.shape[1], img_pad.shape[2]

    stitched = np.zeros((Hp, Wp), dtype=np.uint8)

    tiles = []
    coords = []

    for y in range(0, Hp, TILE):
        for x in range(0, Wp, TILE):
            tile = img_pad[:, y:y+TILE, x:x+TILE]  # 4,H,W
            tile = normalize_tile(tile)
            tiles.append(tile)
            coords.append((y, x))

    tiles = np.stack(tiles, axis=0)  # N,4,157,157
    N = tiles.shape[0]
    print("Total tiles:", N)

    with torch.no_grad():
        for i in range(0, N, BATCH):
            batch = torch.from_numpy(tiles[i:i+BATCH]).to(DEVICE)
            probs = torch.sigmoid(model(batch)).squeeze(1).cpu().numpy()  # N,H,W
            preds = (probs >= THRESH).astype(np.uint8) * 255

            for j in range(preds.shape[0]):
                y, x = coords[i + j]
                stitched[y:y+TILE, x:x+TILE] = preds[j]

    # crop back to original size
    stitched = stitched[:H, :W]
    return stitched


def get_pixel_area_m2(transform):
    """
    pixel area in m² if transform is meters-based.
    If CRS is degrees, this won't be accurate (but most TOA products are projected).
    """
    try:
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        if px_w > 0 and px_h > 0:
            return float(px_w * px_h)
    except:
        pass
    return None


# -------------------------
# MAIN
# -------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", DEVICE)
    print("Model :", MODEL_PATH)
    print("Input :", INPUT_TIF)

    model = load_model()

    with rasterio.open(INPUT_TIF) as src:
        img = src.read()  # (bands,H,W)
        transform = src.transform
        crs = src.crs
        H, W = src.height, src.width

    if img.shape[0] < 4:
        raise ValueError(f"Your ROI tif has only {img.shape[0]} bands. Need 4 (RGB+NIR).")

    img4 = img[:4]  # take first 4 bands

    # 1) stitched mask (internal)
    stitched_mask = predict_stitched_mask(model, img4)

    # 2) stitched overlay (final feature)
    rgb_disp = to_display_rgb(img4)
    overlay = draw_red_fill(rgb_disp, stitched_mask, alpha=ALPHA)
    cv2.imwrite(str(OUT_OVERLAY), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("✅ Saved overlay:", OUT_OVERLAY)

    # 3) total area (final feature)
    mangrove_px = int((stitched_mask > 0).sum())
    pixel_area_m2 = get_pixel_area_m2(transform)

    stats = {
        "input_file": str(INPUT_TIF),
        "width_px": W,
        "height_px": H,
        "bands_used": 4,
        "tile_size": TILE,
        "threshold": THRESH,
        "mangrove_pixels": mangrove_px,
        "pixel_area_m2": pixel_area_m2,
        "mangrove_area_m2": (mangrove_px * pixel_area_m2) if pixel_area_m2 else None,
        "mangrove_area_ha": ((mangrove_px * pixel_area_m2) / 10000.0) if pixel_area_m2 else None,
        "crs": str(crs) if crs else None
    }

    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print("✅ Saved stats:", OUT_STATS)

    if stats["mangrove_area_ha"] is not None:
        print(f"\nTOTAL MANGROVE AREA: {stats['mangrove_area_ha']:.2f} ha")
    else:
        print("\nArea not computed (CRS may be degrees). Still saved pixel count.")


if __name__ == "__main__":
    main()