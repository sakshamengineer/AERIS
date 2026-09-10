from typing import Any
from  preprocessing.loader import load_image

def validate_inputs(images: list[str],query: str,task: str | None = None,) -> dict[str, Any]:
    """
    Validate image files and basic compatibility.

    This is a structural compatibility check.
    It does NOT claim to identify optical vs SAR
    from file extension alone.
    """

    if not images:
        return {
            "valid": False,
            "error": "No images provided.",
        }

    if not query or not query.strip():
        return {
            "valid": False,
            "error": "No query provided.",
        }

    # --------------------------------------------------------
    # Inspect every image
    # --------------------------------------------------------

    metadata = []

    try:
        for image_path in images:
            image,metadata1 = load_image(image_path)
            metadata.append(metadata1)

    except (ValueError, FileNotFoundError) as error:

        return {
            "valid": False,
            "error": str(error),
        }

    # --------------------------------------------------------
    # Image count rules
    # --------------------------------------------------------

    if task in {
        "change_detection",
        "change_vqa",
    }:

        if len(images) != 2:
            return {
                "valid": False,
                "error": (
                    f"{task} requires exactly 2 images, "
                    f"but {len(images)} were provided."
                ),
                "metadata": metadata,
            }

    elif task == "optical_sar":

        if len(images) != 4:
            return {
                "valid": False,
                "error": (
                    "Optical-SAR bi-temporal analysis requires "
                    "exactly 4 images: optical T1, optical T2, "
                    "SAR T1, and SAR T2."
                ),
                "metadata": metadata,
            }

    elif task in {"vqa", "captioning"}:

        if len(images) != 1:
            return {
                "valid": False,
                "error": (
                    f"{task} requires exactly 1 image, "
                    f"but {len(images)} were provided."
                ),
                "metadata": metadata,
            }

    # --------------------------------------------------------
    # Pair dimension compatibility
    # --------------------------------------------------------

    if len(metadata) == 2:

        first = metadata[0]
        second = metadata[1]

        same_dimensions = (
            first["width"] == second["width"]
            and first["height"] == second["height"]
        )

        dimension_status = (
            "compatible"
            if same_dimensions
            else "different_but_alignable"
        )

    elif len(metadata) == 4:

        optical_pair = metadata[:2]
        sar_pair = metadata[2:4]
        all_dims = [(m["width"], m["height"]) for m in metadata]
        same_dimensions = len(set(all_dims)) == 1
        dimension_status = "compatible" if same_dimensions else "different_but_alignable"

    else:

        dimension_status = "single_image"

    # --------------------------------------------------------
    # CRS compatibility
    # --------------------------------------------------------

    if len(metadata) == 2:

        first_crs = metadata[0]["crs"]
        second_crs = metadata[1]["crs"]

        if first_crs is None or second_crs is None:
            crs_status = "unknown"
        elif first_crs == second_crs:
            crs_status = "matching"
        else:
            crs_status = "different"

    elif len(metadata) == 4:

        crs_values = [m["crs"] for m in metadata]
        if any(value is None for value in crs_values):
            crs_status = "unknown"
        elif len(set(crs_values)) == 1:
            crs_status = "matching"
        else:
            crs_status = "different"

    else:

        crs_status = "single_image"

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "valid": True,
        "metadata": metadata,
        "dimension_status": dimension_status,
        "crs_status": crs_status,
        "message": (
            "Input format, file accessibility, "
            "image count and basic compatibility "
            "checks passed."
        ),
    }