from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from preprocessing.normalisation import geotiff_to_rgb, normalize_array

WEB_SAFE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def generate_display_image(source_path: str, output_dir: str) -> str | None:
    """
    Return a path to a browser-renderable image for `source_path`.

    - Already web-safe formats (png/jpg/...) are returned unchanged.
    - GeoTIFF/TIFF files are converted to an 8-bit RGB PNG saved
      into `output_dir`.
    - If conversion fails for any reason, returns None so the
      caller can decide to omit the preview rather than link to
      a broken file.
    """

    source_path = Path(source_path)
    extension = source_path.suffix.lower()

    if extension in WEB_SAFE_EXTENSIONS:
        return str(source_path)

    if extension not in {".tif", ".tiff"}:
        # Unknown format - nothing sensible we can do, let the
        # caller fall back to the original path.
        return str(source_path)

    output_path = Path(output_dir) / f"{source_path.stem}_preview.png"

    # ----------------------------------------------------------
    # FAST PATH:
    # Many "photo-style" TIFFs (like the sample test.tiff files
    # in this project) are already plain 8-bit RGB and PIL can
    # open + convert them directly, no band math needed.
    # ----------------------------------------------------------

    try:
        with Image.open(source_path) as image:
            image.convert("RGB").save(output_path, "PNG")
            return str(output_path)

    except Exception:
        pass

    # ----------------------------------------------------------
    # FALLBACK:
    # Real multi/hyperspectral GeoTIFFs - read via rasterio and
    # build an RGB preview from the first 3 bands (or replicate
    # a single band 3x for SAR/panchromatic imagery).
    # ----------------------------------------------------------

    try:
        with rasterio.open(source_path) as src:
            data = src.read()

        if data.shape[0] >= 3:
            rgb = geotiff_to_rgb(data)
        else:
            band = normalize_array(data[0])
            rgb = np.stack([band, band, band], axis=-1)

        Image.fromarray(rgb, mode="RGB").save(output_path, "PNG")
        return str(output_path)

    except Exception:
        return None