from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from config import SUPPORTED_GEOSPATIAL_FORMATS,SUPPORTED_IMAGE_FORMATS

# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(file_path: str) -> tuple[np.ndarray, dict]:

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # --------------------------------------------------------
    # GeoTIFF / TIFF
    # --------------------------------------------------------

    extension = path.suffix.lower()

    if extension in SUPPORTED_GEOSPATIAL_FORMATS:
        with rasterio.open(path) as src:

            image = src.read()

            metadata = {
                "file_name": path.name,
                "file_type": "GeoTIFF",
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs)
                if src.crs
                else None,
                "transform": list(src.transform),
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "resolutions": src.res,
                "georeferenced" : (src.crs is not None and not src.transform.is_identity)
            }

        return image, metadata

    # --------------------------------------------------------
    # PNG / JPEG
    # --------------------------------------------------------

    if extension in SUPPORTED_IMAGE_FORMATS:
        pil_image = Image.open(path).convert("RGB")
        image = np.array(pil_image)

        metadata = {
            "file_name": path.name,
            "file_type": "Standard Image",
            "width": image.shape[1],
            "height": image.shape[0],
            "bands": image.shape[2],
            "dtype": str(image.dtype),
            "crs": None,
            "transform": None,
            "bounds": None,
            "resolutions": None,
        }
        return image, metadata

    # --------------------------------------------------------
    # Unsupported format
    # --------------------------------------------------------

    raise ValueError(
        f"Unsupported image format: {extension}"
    )

# ============================================================
# IMAGE INFORMATION
# ============================================================

def get_image_info(metadata: dict) -> str:
    """
    Convert image metadata into a human-readable summary.
    """
    info = []

    info.append(f"File: {metadata['file_name']}")
    info.append(f"Type: {metadata['file_type']}")
    info.append(f"Dimensions: "f"{metadata['width']} x {metadata['height']}")
    info.append(f"Bands: {metadata['bands']}")
    info.append(f"Data type: {metadata['dtype']}")
    if metadata["crs"]:
        info.append(f"CRS: {metadata['crs']}")
    if metadata["resolutions"]:
        info.append(f"Resolution: "f"{metadata['resolutions']}")

    return "\n".join(info)