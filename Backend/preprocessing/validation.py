from pathlib import Path
from config import (SUPPORTED_GEOSPATIAL_FORMATS,SUPPORTED_IMAGE_FORMATS,MAX_UPLOAD_SIZE_MB,)


def validate_file(file_path: str) -> dict:

    path = Path(file_path)

    result = {
        "valid": True,
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "file_size_mb": 0.0,
        "file_type": None,
        "errors": [],
    }

    # --------------------------------------------------------
    # 1. Check whether file exists
    # --------------------------------------------------------

    if not path.exists():
        result["valid"] = False
        result["errors"].append("File does not exist.")
        return result

    # --------------------------------------------------------
    # 2. Check file size
    # --------------------------------------------------------

    file_size_mb = path.stat().st_size / (1024 * 1024)

    result["file_size_mb"] = round(file_size_mb, 2)

    if file_size_mb > MAX_UPLOAD_SIZE_MB:
        result["valid"] = False
        result["errors"].append(
            f"File size exceeds the {MAX_UPLOAD_SIZE_MB} MB limit."
        )

    if path.stat().st_size == 0:
        result["valid"] = False
        result["errors"].append("File is empty.")

    # --------------------------------------------------------
    # 3. Check file extension
    # --------------------------------------------------------

    extension = path.suffix.lower()

    if extension in SUPPORTED_GEOSPATIAL_FORMATS:
        result["file_type"] = "geospatial"

    elif extension in SUPPORTED_IMAGE_FORMATS:
        result["file_type"] = "image"

    else:
        result["valid"] = False
        result["errors"].append(
            f"Unsupported file format: {extension or 'unknown'}"
        )

    return result


def validate_multiple_files(file_paths: list[str]) -> dict:
    """
    Validate multiple files together.

    This performs basic checks on:
        - Number of images
        - Individual file validity

    Pair-specific checks such as modality, CRS,
    dimensions and co-registration will be added later.

    Returns:
        Dictionary containing overall validation status
        and individual file results.
    """

    result = {
        "valid": True,
        "number_of_files": len(file_paths),
        "files": [],
        "errors": [],
    }

    # --------------------------------------------------------
    # Check number of files
    # --------------------------------------------------------

    if len(file_paths) == 0:
        result["valid"] = False
        result["errors"].append(
            "At least one image is required."
        )
        return result

    if len(file_paths) > 2:
        result["valid"] = False
        result["errors"].append(
            "A maximum of two images is supported."
        )

    # --------------------------------------------------------
    # Validate individual files
    # --------------------------------------------------------

    for file_path in file_paths:

        file_result = validate_file(file_path)

        result["files"].append(file_result)

        if not file_result["valid"]:
            result["valid"] = False
            result["errors"].extend(
                [
                    f"{file_result['file_name']}: {error}"
                    for error in file_result["errors"]
                ]
            )

    return result