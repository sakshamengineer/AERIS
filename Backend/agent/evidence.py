import os
from typing import Any


def create_evidence(evidence_type: str,path: str | None,description: str,) -> dict[str, Any] | None:
    """
    Create a standardized evidence object.

    Evidence is only considered available when the
    referenced file actually exists.
    """

    if not path:
        return None

    exists = os.path.exists(path)

    return {
        "type": evidence_type,
        "path": path,
        "description": description,
        "exists": exists,
    }


def create_input_evidence(image_path: str,description: str = "Input image analyzed by the model.",) -> dict[str, Any] | None:
    """
    Create evidence referencing an input image.
    """

    return create_evidence(
        evidence_type="input_image",
        path=image_path,
        description=description,
    )


def create_change_map_evidence(path: str) -> dict[str, Any] | None:
    """
    Create evidence for a bi-temporal change map.
    """

    return create_evidence(
        evidence_type="change_map",
        path=path,
        description=(
            "Visual map highlighting pixels identified "
            "as changed between the two input images."
        ),
    )


def create_optical_sar_evidence(path: str,) -> dict[str, Any] | None:
    """
    Create evidence for Optical-SAR fusion analysis.
    """

    return create_evidence(
        evidence_type="optical_sar_fusion",
        path=path,
        description=(
            "Visual output generated from Optical-SAR "
            "cross-modal fusion analysis."
        ),
    )


def create_multi_image_evidence(images: list[str]) -> list[dict[str, Any]]:
    """
    Create evidence entries for multiple input images.
    """

    evidence = []

    for index, image_path in enumerate(images):

        item = create_input_evidence(
            image_path=image_path,
            description=(
                f"Input image {index + 1} used "
                "for analysis."
            ),
        )

        if item:
            evidence.append(item)

    return evidence