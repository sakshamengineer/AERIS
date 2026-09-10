import numpy as np
from PIL import Image

# ============================================================
# GENERIC NORMALIZATION
# ============================================================

def normalize_array(image: np.ndarray,min_value: float | None = None,max_value: float | None = None,) -> np.ndarray:

    image = image.astype(np.float32)

    if min_value is None:
        min_value = np.nanpercentile(image, 2)

    if max_value is None:
        max_value = np.nanpercentile(image, 98)

    if max_value <= min_value:
        return np.zeros_like(image,dtype=np.uint8,)

    image = np.clip(image,min_value,max_value,)

    image = ((image - min_value)/(max_value - min_value)* 255.0)

    return image.astype(np.uint8)

# ============================================================
# GEO-TIFF → RGB
# ============================================================

def geotiff_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a GeoTIFF array into an RGB HWC image.

    Expected input:
        (bands, height, width)

    Output:
        (height, width, 3)

    For the prototype, the first three available
    bands are treated as RGB.

    NOTE:
        Actual satellite products may use different
        band ordering. This will be made sensor-aware
        later.
    """

    if image.ndim != 3:
        raise ValueError("Expected GeoTIFF image with shape (bands, height, width).")

    bands, height, width = image.shape

    if bands < 3:
        raise ValueError(
            "At least 3 bands are required "
            "to create an RGB representation."
        )
    
    rgb = image[:3]

    normalized_bands = []

    for band in rgb:
        normalise_band = normalize_array(band)
        normalized_bands.append(normalise_band)

    # --------------------------------------------------------
    # Convert CHW → HWC
    # --------------------------------------------------------

    rgb = np.stack(normalized_bands,axis=0,)
    rgb = np.transpose(rgb,(1, 2, 0),)
    return rgb

# ============================================================
# STANDARD IMAGE → RGB
# ============================================================

def standardize_rgb(image: np.ndarray,) -> np.ndarray:
    """
    Ensure a standard image is represented as
    an RGB uint8 NumPy array.

    Expected input:
        (H, W, 3)

    Output:
        (H, W, 3) uint8
    """

    if image.ndim != 3:
        raise ValueError("Expected image with shape (H, W, C).")

    if image.shape[2] != 3:
        raise ValueError("Expected an RGB image with 3 channels.")

    if image.dtype != np.uint8:
        image = normalize_array(image)

    return image

def to_pil_image(image: np.ndarray,) -> Image.Image:
    """
    Convert an RGB NumPy array into a PIL Image.
    """

    if image.dtype != np.uint8:
        image = normalize_array(image)

    return Image.fromarray(image,mode="RGB")