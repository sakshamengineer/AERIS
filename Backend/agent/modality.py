import os
from typing import Any
import rasterio
from PIL import Image
import numpy as np

OPTICAL = "optical"
SAR = "sar"
UNKNOWN = "unknown"


def detect_modalities(images: list[str],declared_modalities: list[str] | None = None,) -> list[dict[str, Any]]:
    modalities = []

    for i, image_path in enumerate(images):
        declared_modality = declared_modalities[i] if declared_modalities else None

        modality = detect_modality(image_path, declared_modality)
        modalities.append(modality)

    return modalities


def detect_modality(
    image_path: str,
    declared_modality: str | None = None,
) -> dict[str, Any]:
    if declared_modality is not None and declared_modality.strip():
        declared = declared_modality.strip().lower()

        if declared in {OPTICAL, SAR}:

            return {
                "modality": declared,
                "confidence": 1.0,
                "reason": (
                    "Modality explicitly provided by "
                    "the user."
                ),
                "source": "user_declared",
            }

        return {
            "modality": UNKNOWN,
            "confidence": 0.0,
            "reason": (
                f"Unsupported declared modality: "
                f"{declared_modality}"
            ),
            "source": "user_declared",
        }

    """
    Estimate the modality of an input remote-sensing image.

    IMPORTANT:
    File extension alone is never used to declare SAR or Optical.
    When the available metadata is insufficient, the result is UNKNOWN.
    """

    if not image_path:
        return {
            "modality": UNKNOWN,
            "confidence": 0.0,
            "reason": "No image path provided.",
        }

    if not os.path.exists(image_path):
        return {
            "modality": UNKNOWN,
            "confidence": 0.0,
            "reason": "Image file does not exist.",
        }

    extension = os.path.splitext(image_path)[1].lower()

    # --------------------------------------------------------
    # GeoTIFF / TIFF
    # --------------------------------------------------------

    if extension in {".tif", ".tiff"}:

        try:
            with rasterio.open(image_path) as src:

                band_count = src.count
                dtypes = src.dtypes

                descriptions = [
                    description.lower()
                    for description in src.descriptions
                    if description
                ]

                tags = {
                    str(key).lower(): str(value).lower()
                    for key, value in src.tags().items()
                }

                all_metadata_text = " ".join(
                    descriptions
                    + list(tags.keys())
                    + list(tags.values())
                )

                # --------------------------------------------
                # Strong SAR indicators
                # --------------------------------------------

                sar_keywords = (
                    "sar",
                    "sentinel-1",
                    "sentinel1",
                    "radar",
                    "sigma0",
                    "sigma_0",
                    "backscatter",
                    "vv",
                    "vh",
                    "hh",
                    "hv",
                    "rtc",
                )

                if any(
                    keyword in all_metadata_text
                    for keyword in sar_keywords
                ):
                    return {
                        "modality": SAR,
                        "confidence": 0.95,
                        "reason": (
                            "SAR/radar metadata indicators "
                            "were detected."
                        ),
                        "bands": band_count,
                        "dtype": list(dtypes),
                    }

                # --------------------------------------------
                # Strong optical indicators
                # --------------------------------------------

                optical_keywords = (
                    "optical",
                    "sentinel-2",
                    "sentinel2",
                    "landsat",
                    "multispectral",
                    "rgb",
                    "nir",
                    "red",
                    "green",
                    "blue",
                    "rededge",
                )

                if any(
                    keyword in all_metadata_text
                    for keyword in optical_keywords
                ):
                    return {
                        "modality": OPTICAL,
                        "confidence": 0.95,
                        "reason": (
                            "Optical/multispectral metadata "
                            "indicators were detected."
                        ),
                        "bands": band_count,
                        "dtype": list(dtypes),
                    }

                # --------------------------------------------
                # Conservative band-based inference
                # --------------------------------------------

                # 3+ bands strongly suggest multispectral/RGB
                # but should not be treated as absolute proof.
                if band_count >= 3:
                    return {
                        "modality": OPTICAL,
                        "confidence": 0.65,
                        "reason": (
                            f"Image contains {band_count} bands; "
                            "this is consistent with optical/"
                            "multispectral imagery."
                        ),
                        "bands": band_count,
                        "dtype": list(dtypes),
                    }

                # A single-band TIFF is ambiguous:
                # it could be SAR, panchromatic optical,
                # DEM, or another raster product.
                return {
                    "modality": UNKNOWN,
                    "confidence": 0.0,
                    "reason": (
                        "Single-band raster without reliable "
                        "modality metadata is ambiguous."
                    ),
                    "bands": band_count,
                    "dtype": list(dtypes),
                }

        except Exception as error:

            return {
                "modality": UNKNOWN,
                "confidence": 0.0,
                "reason": f"Could not inspect TIFF: {error}",
            }

    # --------------------------------------------------------
    # JPEG / PNG
    # --------------------------------------------------------

    if extension in {".jpg", ".jpeg", ".png"}:

        try:

            with Image.open(image_path) as image:

                arr = np.array(image)

                # RGB/RGBA images are generally visual/optical
                # imagery, but we still use conservative wording.
                dict1 =  {
                        "modality": SAR,
                        "confidence": 0.80,
                        "reason": (
                            "RGB/RGBA image format is "
                            "consistent with optical imagery."
                        ),
                        "bands": (
                            3
                            if image.mode == "RGB"
                            else 4
                        ),
                    }
                
                if image.mode == "L" :
                    dict1["modality"] = SAR
                    return dict1
                else:
                    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

                    if np.allclose(r, g, atol=5) and np.allclose(g, b, atol=5):
                        dict1["modality"] = SAR
                        dict1["reason"] = (
                            "Near-grayscale RGB image is "
                            "consistent with SAR imagery."
                        )
                        return dict1

                    else:
                        dict1["modality"] = OPTICAL
                        return dict1

        except Exception as error:

            return {
                "modality": UNKNOWN,
                "confidence": 0.0,
                "reason": f"Could not inspect image: {error}",
            }

    return {
        "modality": UNKNOWN,
        "confidence": 0.0,
        "reason": (
            f"Unsupported or unrecognized extension: "
            f"{extension}"
        ),
    }


def check_optical_sar_pair(
    images: list[str],
    declared_modalities: list[str] | None = None,
) -> dict[str, Any]:
    """
    Check whether two images can confidently be treated
    as an Optical + SAR pair.

    The function is intentionally conservative.
    Unknown modality means the system should not claim
    that the pair is definitely Optical + SAR.
    """

    if len(images) != 2:

        return {
            "valid": False,
            "is_optical_sar": False,
            "error": (
                "Optical-SAR analysis requires "
                "exactly two images."
            ),
            "modalities": [],
        }

    results = detect_modalities(
        images,
        declared_modalities=declared_modalities,
    )
    modality_values = [
        result["modality"]
        for result in results
    ]

    has_optical = OPTICAL in modality_values
    has_sar = SAR in modality_values

    if has_optical and has_sar:

        return {
            "valid": True,
            "is_optical_sar": True,
            "message": (
                "One optical image and one SAR image "
                "were identified."
            ),
            "modalities": results,
        }

    if UNKNOWN in modality_values:

        return {
            "valid": True,
            "is_optical_sar": False,
            "requires_confirmation": True,
            "message": (
                "At least one image has unknown modality. "
                "Optical-SAR analysis should require "
                "user confirmation or reliable metadata."
            ),
            "modalities": results,
        }

    return {
        "valid": True,
        "is_optical_sar": False,
        "requires_confirmation": True,
        "message": (
            "The supplied pair could not be confidently "
            "identified as one optical image and one SAR image."
        ),
        "modalities": results,
    }

def check_optical_sar_temporal_pair(
    images: list[str],
    declared_modalities: list[str] | None = None,
) -> dict[str, Any]:
    """Validate the four inputs required by OpticalSARChangeNet.

    Expected order is optical T1, optical T2, SAR T1, SAR T2. Explicit
    modality declarations are preferred because SAR may be stored as an
    RGB/grayscale preview image rather than a native multi-band raster.
    """
    if len(images) != 4:
        return {
            "valid": False,
            "is_optical_sar": False,
            "message": (
                "Optical-SAR bi-temporal analysis requires exactly 4 images: "
                "optical T1, optical T2, SAR T1, and SAR T2."
            ),
            "modalities": [],
        }

    results = detect_modalities(
        images,
        declared_modalities=declared_modalities,
    )
    values = [item.get("modality") for item in results]

    if values.count(OPTICAL) >= 2 and values.count(SAR) >= 2:
        return {
            "valid": True,
            "is_optical_sar": True,
            "message": "Verified two optical images and two SAR images.",
            "modalities": results,
        }

    return {
        "valid": True,
        "is_optical_sar": False,
        "requires_confirmation": True,
        "message": (
            "The four inputs must contain two optical images and two SAR images. "
            "Use the modality tags to correct ambiguous inputs."
        ),
        "modalities": results,
    }
