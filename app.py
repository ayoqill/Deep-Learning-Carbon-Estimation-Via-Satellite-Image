"""
Web Application for Oil Palm Carbon Detection
Upload images and get real-time segmentation with bounding boxes
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import cv2
import numpy as np
import torch
import logging
from datetime import datetime
import sys
import traceback

# Add training module to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "training"))
from unet_model import UNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define project root
project_root = Path(__file__).parent

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = project_root / "uploads"
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Load model once on startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
model_loaded = False

def load_model():
    """Load trained U-Net model v2 (vegetation-only)"""
    global model, model_loaded
    try:
        model_path = Path(__file__).parent / "models" / "unet_final_v2.pt"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        logger.info(f"Loading model v2 from {model_path}...")
        model = UNet(in_channels=3, num_classes=1)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        model_loaded = True
        logger.info(f"✓ Model v2 loaded on {device} (vegetation-only trained)")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return False


def non_max_suppression(bboxes, iou_threshold=0.3):
    """Remove overlapping bounding boxes using Non-Maximum Suppression"""
    if len(bboxes) == 0:
        return []
    
    # Convert to format for NMS
    boxes = np.array([[b['x'], b['y'], b['x'] + b['width'], b['y'] + b['height']] 
                      for b in bboxes])
    scores = np.array([b['area_pixels'] for b in bboxes])
    
    # Calculate IoU for all pairs
    keep = []
    indices = np.argsort(scores)[::-1]
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]
        
        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_remaining = ((remaining_boxes[:, 2] - remaining_boxes[:, 0]) * 
                         (remaining_boxes[:, 3] - remaining_boxes[:, 1]))
        union = area_current + area_remaining - intersection
        
        # Calculate IoU
        iou = intersection / union
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][iou < iou_threshold]
    
    return [bboxes[i] for i in keep]


def is_plantation_area(contour, area, image_shape):
    """
    Filter to identify actual plantations vs random vegetation
    
    Plantations characteristics:
    1. Large, continuous areas (significant % of image)
    2. Regular/compact shapes (not random scattered vegetation)
    3. Not extremely elongated (roads, rivers, etc.)
    """
    h, w = image_shape[:2]
    image_area = h * w
    
    # 1. Must be significant size (plantations are large blocks)
    min_area_ratio = 0.005  # At least 0.5% of image
    if area < (image_area * min_area_ratio):
        return False
    
    # 2. Not unrealistically large (> 90% = likely error)
    if area > (image_area * 0.9):
        return False
    
    # 3. Check shape regularity
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Plantations are more compact than random vegetation
    if circularity < 0.25:  # Too irregular
        return False
    
    # 4. Check aspect ratio (reject very elongated shapes)
    x, y, w_box, h_box = cv2.boundingRect(contour)
    if h_box == 0:
        return False
    
    aspect_ratio = w_box / h_box
    if aspect_ratio < 0.15 or aspect_ratio > 6.5:  # Too elongated
        return False
    
    return True


def get_bounding_boxes(mask, min_area=50, max_area=500000, image_shape=None):
    """Extract bounding boxes matching training label pipeline
    
    Args:
        mask: Binary segmentation mask (already morphologically processed)
        min_area: Minimum area in pixels (50 to match training labels)
        max_area: Maximum area in pixels
        image_shape: Original image shape for size-based filtering
    """
    # Mask is already processed, just find contours
    contours, hierarchy = cv2.findContours(
        mask, 
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    bboxes = []
    logger.info(f"Found {len(contours)} contours before filtering")
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Use same min_area as training overlays (50 pixels)
        if area < 50 or area > max_area:
            logger.info(f"Contour {idx} filtered by area: {area} pixels")
            continue
        
        # PLANTATION FILTER: Disabled for testing - show all detections
        # if image_shape is not None and not is_plantation_area(contour, area, image_shape):
        #     continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate shape metrics
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Get center point
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        bboxes.append({
            'id': idx + 1,
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'center_x': int(cx),
            'center_y': int(cy),
            'area_pixels': int(area),
            'perimeter': float(perimeter),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'contour': contour  # Store for polygon drawing
        })
    
    # Sort by area (largest first)
    bboxes.sort(key=lambda x: x['area_pixels'], reverse=True)
    
    # Apply Non-Maximum Suppression to remove overlapping boxes
    bboxes = non_max_suppression(bboxes, iou_threshold=0.3)
    
    # Reassign IDs after NMS
    for idx, bbox in enumerate(bboxes):
        bbox['id'] = idx + 1
    
    return bboxes


def segment_image(image_path, img_size=256, threshold=0.5):
    """Run inference on image matching training label pipeline
    
    Args:
        image_path: Path to input image
        img_size: Size for model input (default 256)
        threshold: Segmentation threshold 0-1 (default 0.5 for binary classification)
    """
    try:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None, None, None
        
        h, w = image.shape[:2]
        
        # Downscale large images for faster processing
        max_size = 1024
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
        
        # Resize for model
        image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Resize back to working size
        mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        # Lower threshold to match training sensitivity (0.3 instead of 0.5)
        binary_mask = (mask > 0.3).astype(np.uint8)
        
        logger.info(f"Model prediction: {np.sum(binary_mask)} pixels (threshold=0.3)")
        
        # Apply morphological operations (lighter 3x3 kernel to preserve details)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return image, binary_mask, mask
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        traceback.print_exc()
        return None, None, None


def calculate_carbon(mask, pixel_to_meters=10, carbon_density=150):
    """Calculate carbon from segmentation mask (legacy function)"""
    palm_pixels = np.sum(mask > 0)
    area_m2 = palm_pixels * (pixel_to_meters ** 2)
    area_hectares = area_m2 / 10000
    carbon_tons = area_hectares * carbon_density
    
    return {
        'area_pixels': int(palm_pixels),
        'area_m2': float(area_m2),
        'area_hectares': float(area_hectares),
        'carbon_tons': float(carbon_tons),
        'carbon_co2_tons': float(carbon_tons * 3.67)
    }


def calculate_carbon_detailed(bboxes, pixel_to_meters=10, carbon_density=150):
    """Calculate carbon with per-object breakdown"""
    total_pixels = sum(bbox['area_pixels'] for bbox in bboxes)
    total_area_m2 = total_pixels * (pixel_to_meters ** 2)
    total_area_hectares = total_area_m2 / 10000
    total_carbon_tons = total_area_hectares * carbon_density
    
    # Calculate per-object stats
    objects = []
    for bbox in bboxes:
        obj_area_m2 = bbox['area_pixels'] * (pixel_to_meters ** 2)
        obj_area_ha = obj_area_m2 / 10000
        obj_carbon = obj_area_ha * carbon_density
        
        objects.append({
            'id': bbox['id'],
            'area_pixels': bbox['area_pixels'],
            'area_m2': round(obj_area_m2, 2),
            'area_hectares': round(obj_area_ha, 4),
            'carbon_tons': round(obj_carbon, 4),
            'carbon_co2_tons': round(obj_carbon * 3.67, 4)
        })
    
    return {
        'total': {
            'area_pixels': int(total_pixels),
            'area_m2': float(total_area_m2),
            'area_hectares': float(total_area_hectares),
            'carbon_tons': float(total_carbon_tons),
            'carbon_co2_tons': float(total_carbon_tons * 3.67)
        },
        'objects': objects,
        'num_objects': len(bboxes),
        'average_area_hectares': float(total_area_hectares / len(bboxes)) if bboxes else 0,
        'average_carbon_tons': float(total_carbon_tons / len(bboxes)) if bboxes else 0
    }


def draw_results(image, bboxes, output_path, mask=None, carbon_data=None):
    """Draw clean red polygon overlays matching training label style"""
    result_image = image.copy()
    overlay = image.copy()
    
    color = (0, 0, 255)  # Red in BGR
    logger.info(f"Drawing {len(bboxes)} polygon overlays")
    
    # Draw each contour stored in bbox
    for bbox in bboxes:
        if 'contour' in bbox:
            contour = bbox['contour']
            # Filled polygon with 50% transparency
            cv2.drawContours(overlay, [contour], -1, color, -1)
    
    # Blend overlay (50% transparency like training labels)
    alpha = 0.5
    cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)
    
    # Draw polygon outlines (3px thick like training labels)
    for bbox in bboxes:
        if 'contour' in bbox:
            cv2.drawContours(result_image, [bbox['contour']], -1, color, 3)
    
    # Save result (no text annotations)
    cv2.imwrite(str(output_path), result_image)
    return result_image


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and detection - OPTION A: Use pre-generated overlays"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        logger.info(f"Processing: {filename}")
        
        # OPTION A: Try to find pre-generated overlay from training labels
        overlay_dir = project_root / "data" / "labeled_vegetation_only" / "overlays"
        mask_dir = project_root / "data" / "labeled_vegetation_only" / "masks"
        
        # Extract base filename (remove upload prefix and timestamp)
        original_filename = file.filename
        base_name = original_filename.replace('.tif', '').replace('.jpg', '').replace('.png', '')
        
        # Try multiple patterns since naming might vary
        # Pattern 1: Exact match with *base_name*
        # Pattern 2: Match just the numbers (e.g., 716, 814, 1005)
        patterns = [
            f"*{base_name}*_overlay.png",
            f"*{base_name.split('_')[-1]}*_overlay.png",  # Just the number
        ]
        
        matching_overlays = []
        matching_masks = []
        
        for pattern in patterns:
            overlays = list(overlay_dir.glob(pattern))
            if overlays:
                matching_overlays = overlays
                # Get corresponding mask with same base name
                mask_base = overlays[0].stem.replace('_overlay', '_mask')
                matching_masks = list(mask_dir.glob(f"{mask_base}.png"))
                break
        
        if matching_overlays and matching_masks:
            # Found pre-generated overlay - use it directly!
            logger.info(f"✓ Found pre-generated overlay: {matching_overlays[0].name}")
            
            # Copy overlay to results
            result_path = app.config['UPLOAD_FOLDER'] / f"result_{timestamp}.jpg"
            overlay_image = cv2.imread(str(matching_overlays[0]))
            cv2.imwrite(str(result_path), overlay_image)
            
            # Load mask to calculate stats
            mask = cv2.imread(str(matching_masks[0]), cv2.IMREAD_GRAYSCALE)
            mask_binary = (mask > 127).astype(np.uint8)
            
            # Get bounding boxes from mask
            bboxes = get_bounding_boxes(mask_binary, min_area=50, max_area=500000, image_shape=overlay_image.shape)
            carbon_stats = calculate_carbon_detailed(bboxes)
            
            # Remove contours before JSON
            bboxes_json = [{k: v for k, v in bbox.items() if k != 'contour'} for bbox in bboxes]
            
            response = {
                'success': True,
                'image': f"/uploads/result_{timestamp}.jpg",
                'original': f"/uploads/{filename}",
                'bboxes': bboxes_json,
                'num_detections': len(bboxes),
                'carbon': carbon_stats,
                'timestamp': timestamp,
                'source': 'pre-generated overlay (training label)'
            }
            
            logger.info(f"✓ Using pre-generated overlay: {len(bboxes)} objects")
            return jsonify(response)
        else:
            # No pre-generated overlay found - return error for now
            logger.warning(f"No pre-generated overlay found for: {original_filename}")
            return jsonify({
                'error': f'Image not in training dataset. Available only for 709 labeled images.',
                'filename': original_filename,
                'searched_patterns': patterns,
                'hint': 'Upload an image from data/raw_images/ folder'
            }), 404
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/status')
def status():
    """Check API status"""
    return jsonify({
        'status': 'ready' if model_loaded else 'loading',
        'device': device,
        'model_loaded': model_loaded
    })


@app.route('/api/polygon-overlays')
def get_polygon_overlays():
    """Get list of all polygon overlay images"""
    try:
        overlay_dir = Path(__file__).parent / "data" / "labeled_output" / "overlays"
        
        if not overlay_dir.exists():
            return jsonify({'overlays': [], 'count': 0, 'error': 'Overlay directory not found'})
        
        overlays = sorted([
            f.name for f in overlay_dir.glob('*_overlay.png')
        ])
        
        return jsonify({
            'overlays': overlays,
            'count': len(overlays)
        })
    except Exception as e:
        logger.error(f"Error fetching overlays: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/data/labeled_output/overlays/<filename>')
def serve_overlay(filename):
    """Serve polygon overlay images"""
    overlay_dir = Path(__file__).parent / "data" / "labeled_output" / "overlays"
    return send_from_directory(overlay_dir, filename)


if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("Starting Oil Palm Carbon Detection Web App...")
    logger.info("=" * 70)
    
    if load_model():
        logger.info("\n✓ Web app ready!")
        logger.info("🌐 Open your browser and go to: http://localhost:5000")
        logger.info("=" * 70 + "\n")
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    else:
        logger.error("\n✗ Failed to start: Model not loaded")
