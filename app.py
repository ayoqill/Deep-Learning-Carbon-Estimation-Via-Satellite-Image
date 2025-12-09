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

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / "uploads"
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Load model once on startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
model_loaded = False

def load_model():
    """Load trained U-Net model"""
    global model, model_loaded
    try:
        model_path = Path(__file__).parent / "models" / "unet_final.pt"
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        logger.info(f"Loading model from {model_path}...")
        model = UNet(in_channels=3, num_classes=1)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        model_loaded = True
        logger.info(f"✓ Model loaded on {device}")
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


def get_bounding_boxes(mask, min_area=100, max_area=50000):
    """Extract bounding boxes with improved filtering and shape validation"""
    # Apply stronger morphological operations for better separation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Erode-dilate to separate touching objects
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_cleaned = cv2.erode(mask_cleaned, kernel_erode, iterations=1)
    mask_cleaned = cv2.dilate(mask_cleaned, kernel_erode, iterations=1)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(
        mask_cleaned, 
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    bboxes = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by area range (remove noise and false detections)
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate shape metrics
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        
        # Filter by shape - oil palms are roughly circular
        if circularity < 0.3:  # Too elongated/irregular
            continue
        
        # Filter extreme aspect ratios
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            continue
        
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
            'aspect_ratio': float(aspect_ratio)
        })
    
    # Sort by area (largest first)
    bboxes.sort(key=lambda x: x['area_pixels'], reverse=True)
    
    # Apply Non-Maximum Suppression to remove overlapping boxes
    bboxes = non_max_suppression(bboxes, iou_threshold=0.3)
    
    # Reassign IDs after NMS
    for idx, bbox in enumerate(bboxes):
        bbox['id'] = idx + 1
    
    return bboxes


def segment_image(image_path, img_size=256, threshold=0.6):
    """Run inference on image with adjustable threshold
    
    Args:
        image_path: Path to input image
        img_size: Size for model input (default 256)
        threshold: Segmentation threshold 0-1 (default 0.6 for higher precision)
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
        binary_mask = (mask > threshold).astype(np.uint8)  # Use adjustable threshold
        
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


def draw_results(image, bboxes, output_path, carbon_data=None):
    """Draw bounding boxes with enhanced visualization"""
    result_image = image.copy()
    
    # Color palette for different objects
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
        (255, 255, 0),  # Cyan
        (128, 0, 128),  # Purple
        (0, 128, 255),  # Orange-Red
    ]
    
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        color = colors[i % len(colors)]
        
        # Draw border only (no transparency overlay for speed)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 3)
        
        # Draw center point
        cx, cy = bbox['center_x'], bbox['center_y']
        cv2.circle(result_image, (cx, cy), 5, color, -1)
        
        # Create detailed label
        if carbon_data and i < len(carbon_data['objects']):
            area_ha = carbon_data['objects'][i]['area_hectares']
            label = f"#{i+1} | {area_ha:.3f} ha"
        else:
            area_ha = bbox['area_pixels'] * (10 ** 2) / 10000
            label = f"#{i+1} | {area_ha:.3f} ha"
        
        # Draw label background
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            result_image, 
            (x, y - label_h - baseline - 5), 
            (x + label_w + 5, y), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            result_image, 
            label, 
            (x + 2, y - baseline - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        # Draw object ID at center
        cv2.putText(
            result_image,
            str(i + 1),
            (cx - 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
    
    # Add summary text
    summary = f"Detected: {len(bboxes)} objects"
    cv2.putText(
        result_image,
        summary,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        3
    )
    
    # Save result
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
    """Handle image upload and detection"""
    try:
        if not model_loaded:
            return jsonify({'error': 'Model not loaded'}), 500
        
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
        
        # Segment image with higher threshold for better precision
        image, binary_mask, prob_mask = segment_image(filepath, threshold=0.6)
        if image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Get bounding boxes with improved filtering
        bboxes = get_bounding_boxes(binary_mask, min_area=100, max_area=50000)
        
        # Calculate carbon with detailed breakdown
        carbon_stats = calculate_carbon_detailed(bboxes)
        
        # Draw results (enhanced visualization)
        result_path = app.config['UPLOAD_FOLDER'] / f"result_{timestamp}.jpg"
        draw_results(image, bboxes, result_path, carbon_stats)
        
        # Prepare response
        response = {
            'success': True,
            'image': f"/uploads/result_{timestamp}.jpg",
            'original': f"/uploads/{filename}",
            'bboxes': bboxes,
            'num_detections': len(bboxes),
            'carbon': carbon_stats,
            'timestamp': timestamp
        }
        
        logger.info(f"✓ Detection complete: {len(bboxes)} objects found")
        
        return jsonify(response)
        
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
