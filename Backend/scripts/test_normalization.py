import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessing.normalisation import (
    normalize_array,
    geotiff_to_rgb,
    to_pil_image,
)
# ------------------------------------------------------------
# Test generic normalization
# ------------------------------------------------------------

image = np.random.randint(
    0,
    10000,
    size=(256, 256),
    dtype=np.uint16,
)

normalized = normalize_array(image)

print("Original:")
print(image.dtype, image.min(), image.max())

print("\nNormalized:")
print(
    normalized.dtype,
    normalized.min(),
    normalized.max(),
)


# ------------------------------------------------------------
# Test GeoTIFF → RGB
# ------------------------------------------------------------

geotiff = np.random.randint(
    0,
    10000,
    size=(4, 256, 256),
    dtype=np.uint16,
)

rgb = geotiff_to_rgb(geotiff)

print("\nGeoTIFF shape:")
print(geotiff.shape)

print("\nRGB shape:")
print(rgb.shape)


# ------------------------------------------------------------
# Test PIL conversion
# ------------------------------------------------------------

pil_image = to_pil_image(rgb)

print("\nPIL image:")
print(pil_image.mode)
print(pil_image.size)