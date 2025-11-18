def extract_bands(image, bands):
    """
    Extract specified bands from a satellite image.

    Parameters:
    - image: The input satellite image.
    - bands: A list of band indices to extract.

    Returns:
    - A new image containing only the specified bands.
    """
    return image[bands]

def export_geotiff(image, filename, metadata):
    """
    Export a processed image to a GeoTIFF file.

    Parameters:
    - image: The image to export.
    - filename: The name of the output GeoTIFF file.
    - metadata: Metadata to include in the GeoTIFF.
    """
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(filename, 'w', driver='GTiff',
                       height=image.shape[0], width=image.shape[1],
                       count=image.shape[2], dtype=image.dtype,
                       crs=metadata['crs'], transform=from_origin(metadata['transform'][0], metadata['transform'][1], metadata['transform'][2], metadata['transform'][3])) as dst:
        dst.write(image)

def process_image(image_path, bands, output_path, metadata):
    """
    Process a satellite image by extracting bands and exporting it.

    Parameters:
    - image_path: Path to the input satellite image.
    - bands: A list of band indices to extract.
    - output_path: Path to save the processed GeoTIFF image.
    - metadata: Metadata for the output image.
    """
    import rasterio

    with rasterio.open(image_path) as src:
        image = src.read()
        processed_image = extract_bands(image, bands)
        export_geotiff(processed_image, output_path, metadata)