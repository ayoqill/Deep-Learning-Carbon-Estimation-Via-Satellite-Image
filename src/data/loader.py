def load_geotiff(file_path):
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(file_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height),
                int(src.width)
            ),
            resampling=Resampling.bilinear
        )
        metadata = src.meta

    return data, metadata

def load_metadata(metadata_path):
    import json

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata

def load_satellite_data(image_path, metadata_path):
    image_data, image_metadata = load_geotiff(image_path)
    metadata = load_metadata(metadata_path)

    return image_data, image_metadata, metadata