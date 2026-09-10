from typing import Literal


TaskType = Literal[
    "vqa",
    "captioning",
    "change_detection",
    "change_vqa",
    "optical_sar",
    "unknown",
]


# ============================================================
# KEYWORD GROUPS
# ============================================================

CHANGE_KEYWORDS = [
    "change",
    "changed",
    "changes",
    "difference",
    "differences",
    "before",
    "after",
    "increase",
    "increased",
    "decrease",
    "decreased",
    "growth",
    "loss",
    "construction",
    "built",
    "building",
    "demolition",
    "demolished",
    "deforestation",
    "forest loss",
    "flood",
    "flooded",
    "water expansion",
    "land cover",
    "development",
    "development",
    "new road",
]


CHANGE_VQA_KEYWORDS = [
    "what changed",
    "what has changed",
    "describe the change",
    "describe changes",
    "explain the change",
    "explain changes",
    "identify the changes",
    "what is different",
    "what are the differences",
    "how has the area changed",
    "how did the area change",
    "compare the two images",
    "compare these images",
]


OPTICAL_SAR_KEYWORDS = [
    "sar",
    "optical",
    "multispectral",
    "radar",
    "both images",
    "combine",
    "together",
    "cross-modal",
    "fusion",
    "fuse"
]


CAPTION_KEYWORDS = [
    "describe",
    "description",
    "caption",
    "scene",
    "summarize",
    "summary",
]


# ============================================================
# TASK DESCRIPTIONS
# ============================================================

TASK_DESCRIPTIONS = {

    "vqa":
        "Visual Question Answering",

    "captioning":
        "Satellite Image Captioning",

    "change_detection":
        "Bi-temporal Change Detection",

    "change_vqa":
        "Bi-temporal Change-VQA",

    "optical_sar":
        "Optical-SAR Cross-Modal Analysis",

    "unknown":
        "Unknown Task",
}


# ============================================================
# KEYWORD SCORING
# ============================================================

def _count_matches(
    query: str,
    keywords: list[str],
) -> int:
    """
    Count how many routing keywords occur in the query.
    """

    return sum(
        1
        for keyword in keywords
        if keyword in query
    )


# ============================================================
# AGENTIC ROUTING
# ============================================================

def route_query(
    query: str,
    number_of_images: int,
    modalities: list[str] | None = None,
) -> dict:
    """
    Agentic task-selection layer.

    Uses:
        - user query
        - number of images
        - known modalities

    Returns:
        selected task
        candidate scores
        routing reason
        confidence
    """

    if not query or not query.strip():

        return {
            "task": "unknown",
            "confidence": 0.0,
            "reason": "No query provided.",
            "candidates": {},
        }

    query = query.lower().strip()

    # --------------------------------------------------------
    # Initialize scores
    # --------------------------------------------------------

    scores = {
        "vqa": 0.0,
        "captioning": 0.0,
        "change_detection": 0.0,
        "change_vqa": 0.0,
        "optical_sar": 0.0,
    }

    # --------------------------------------------------------
    # Query signals
    # --------------------------------------------------------

    change_matches = _count_matches(
        query,
        CHANGE_KEYWORDS,
    )

    change_vqa_matches = _count_matches(
        query,
        CHANGE_VQA_KEYWORDS,
    )

    optical_sar_matches = _count_matches(
        query,
        OPTICAL_SAR_KEYWORDS,
    )

    caption_matches = _count_matches(
        query,
        CAPTION_KEYWORDS,
    )

    # --------------------------------------------------------
    # Image-count signals
    # --------------------------------------------------------

    two_images = number_of_images >= 2
    single_image = number_of_images == 1

    # --------------------------------------------------------
    # VQA baseline
    # --------------------------------------------------------

    if single_image:
        scores["vqa"] += 1.0

    # --------------------------------------------------------
    # Captioning
    # --------------------------------------------------------

    if caption_matches > 0 and single_image:

        scores["captioning"] += (
            2.0 + caption_matches * 0.75
        )

    # --------------------------------------------------------
    # Change detection
    # --------------------------------------------------------

    if two_images:
        # Two co-temporal optical images are the native input contract for
        # SiameseTemporalCD. Give the spatial change detector a baseline
        # score so ordinary bi-temporal requests do not fall through to a
        # single-image task.
        scores["change_detection"] += 1.5

    if two_images and change_matches > 0:

        scores["change_detection"] += (
            2.0 + change_matches * 0.75
        )

    # --------------------------------------------------------
    # Change-VQA
    # --------------------------------------------------------

    if two_images and change_vqa_matches > 0:

        # Bi-temporal change detection is the primary specialist for
        # two-image change analysis. Natural-language wording alone must
        # not bypass the trained spatial change detector.
        scores["change_detection"] += (
            3.0 + change_vqa_matches * 1.0
        )

    # Questions asking "what changed" are stronger
    # Change-VQA signals than generic "change".
    if two_images and (
        "what changed" in query
        or "what has changed" in query
        or "what is different" in query
        or "what are the differences" in query
    ):

        scores["change_detection"] += 2.0

    # --------------------------------------------------------
    # Optical + SAR temporal analysis
    # --------------------------------------------------------

    four_images = number_of_images == 4

    if four_images and optical_sar_matches > 0:
        scores["optical_sar"] += (
            4.0 + optical_sar_matches * 0.75
        )

    # Explicit modality information is a very strong signal.
    if modalities:

        normalized_modalities = [
            str(modality).lower().strip()
            for modality in modalities
        ]

        if (
            normalized_modalities.count("optical") >= 2
            and normalized_modalities.count("sar") >= 2
            and four_images
        ):
            scores["optical_sar"] += 7.0

    # Four explicitly typed inputs are unambiguous for the trained
    # OpticalSARChangeNet and must not be hijacked by generic change wording.
    if modalities and four_images:
        normalized_modalities = [str(m).lower().strip() for m in modalities]
        if normalized_modalities.count("optical") >= 2 and normalized_modalities.count("sar") >= 2:
            scores["optical_sar"] += 10.0
            scores["change_detection"] = 0.0
            scores["change_vqa"] = 0.0

    # --------------------------------------------------------
    # Invalid image-count combinations
    # --------------------------------------------------------

    if not single_image:

        scores["captioning"] = 0.0

    if not two_images:

        scores["change_detection"] = 0.0
        scores["change_vqa"] = 0.0

    if not four_images:
        scores["optical_sar"] = 0.0

    # --------------------------------------------------------
    # Select candidate
    # --------------------------------------------------------

    best_task = max(
        scores,
        key=scores.get,
    )

    best_score = scores[best_task]

    if best_score <= 0:

        return {
            "task": "unknown",
            "confidence": 0.0,
            "reason": (
                "No compatible task signals "
                "were detected."
            ),
            "candidates": scores,
        }

    # --------------------------------------------------------
    # Convert score into routing confidence
    # --------------------------------------------------------

    total_score = sum(scores.values())

    if total_score > 0:
        confidence = best_score / total_score
    else:
        confidence = 0.0

    confidence = round(
        min(max(confidence, 0.0), 1.0),
        4,
    )

    # --------------------------------------------------------
    # Routing reason
    # --------------------------------------------------------

    if best_task == "optical_sar":

        if modalities:
            reason = (
                "Four images were provided with two optical "
                "and two SAR inputs for bi-temporal multimodal analysis."
            )
        else:
            reason = (
                "Four-image Optical/SAR cross-modal analysis "
                "signals were detected in the query."
            )

    elif best_task == "change_vqa":

        reason = (
            "A bi-temporal natural-language change question was detected."
        )

    elif best_task == "change_detection":

        reason = (
            "Two images and change/difference signals "
            "were detected without a specific change question."
        )

    elif best_task == "captioning":

        reason = (
            "A scene description/captioning request "
            "was detected for a single image."
        )

    else:

        reason = (
            "A single-image visual question was detected."
        )

    return {
        "task": best_task,
        "confidence": confidence,
        "reason": reason,
        "candidates": {
            task: round(score, 3)
            for task, score in scores.items()
        },
    }


# ============================================================
# BACKWARD-COMPATIBLE ROUTER
# ============================================================

def identify_task(
    query: str,
    number_of_images: int,
    modalities: list[str] | None = None,
) -> TaskType:
    """
    Existing controller-compatible interface.

    Returns only the selected task.
    """

    decision = route_query(
        query=query,
        number_of_images=number_of_images,
        modalities=modalities,
    )

    return decision["task"]


# ============================================================
# TASK DESCRIPTION
# ============================================================

def get_task_description(
    task: TaskType,
) -> str:

    return TASK_DESCRIPTIONS.get(
        task,
        "Unknown Task",
    )