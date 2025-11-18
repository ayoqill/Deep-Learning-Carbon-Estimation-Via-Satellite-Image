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


def get_bounding_boxes(mask):
    """Extract bounding boxes from segmentation mask"""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Filter small noise
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h),
            'area_pixels': int(area)
        })
    
    return bboxes


def segment_image(image_path, img_size=256):
    """Run inference on image"""
    try:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            return None, None, None
        
        h, w = image.shape[:2]
        
        # Resize for model
        image_resized = cv2.resize(image, (img_size, img_size))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = model(image_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        
        # Resize back to original
        mask = cv2.resize(pred, (w, h))
        binary_mask = (mask > 0.5).astype(np.uint8)
        
        return image, binary_mask, mask
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        traceback.print_exc()
        return None, None, None


def calculate_carbon(mask, pixel_to_meters=10, carbon_density=150):
    """Calculate carbon from segmentation mask"""
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


def draw_results(image, bboxes, output_path):
    """Draw bounding boxes on image"""
    result_image = image.copy()
    
    # Draw bounding boxes
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Draw rectangle
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label
        label = f"Palm {i+1}"
        cv2.putText(result_image, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
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
        
        # Segment image
        image, binary_mask, prob_mask = segment_image(filepath)
        if image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Get bounding boxes
        bboxes = get_bounding_boxes(binary_mask)
        
        # Calculate carbon
        carbon_stats = calculate_carbon(binary_mask)
        
        # Draw results
        result_path = app.config['UPLOAD_FOLDER'] / f"result_{timestamp}.jpg"
        draw_results(image, bboxes, result_path)
        
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
