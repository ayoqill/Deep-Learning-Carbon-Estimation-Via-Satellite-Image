"""
Batch run SAM-2 on all cleaned tiles to generate draft binary masks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from sam2_annotator import SAM2Annotator, batch_annotate

# Input/output directories
INPUT_DIR = '/Users/amxr666/Desktop/mangrove-carbon-pipeline/data/tiles_clean'
OUTPUT_DIR = '/Users/amxr666/Desktop/mangrove-carbon-pipeline/data/masks_raw'

# Option: use a box covering the whole image for each tile (default prompt)
def get_full_image_boxes(image_dir):
    boxes = {}
    for tif_path in Path(image_dir).glob('*.tif'):
        import rasterio
        with rasterio.open(tif_path) as src:
            h, w = src.height, src.width
        # [x1, y1, x2, y2] (full image)
        boxes[tif_path.name] = [0, 0, w-1, h-1]
    return boxes

def main():
    annotator = SAM2Annotator()
    boxes = get_full_image_boxes(INPUT_DIR)
    batch_annotate(
        image_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        annotator=annotator,
        prompt_type='box',
        boxes_per_image=boxes
    )

if __name__ == '__main__':
    main()
