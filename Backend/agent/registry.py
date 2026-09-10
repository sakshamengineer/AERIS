from models.shared_vlm import (get_shared_vlm)
from models.temporal_change_detection import (get_temporal_change_detection_model)
from models.change_vqa import (ChangeVQA)
from models.optical_sar_inference import (get_optical_sar_model)

# ============================================================
# SATQUERY AI MODEL REGISTRY
# ============================================================

MODEL_REGISTRY = {

    "vqa": {
        "name": "Remote-Sensing VQA",
        "type": "vision_language",
        "description": (
            "Answers natural-language questions "
            "about a remote-sensing image."
        ),
        "input_type": "single_image",
    },

    "captioning": {
        "name": "Remote-Sensing Captioning",
        "type": "vision_language",
        "description": (
            "Generates a natural-language description "
            "of a remote-sensing image."
        ),
        "input_type": "single_image",
    },

    "change_detection": {
        "name": "SiameseTemporalCD",
        "type": "change_detection",
        "description": (
            "Detects spatial changes between "
            "two corresponding satellite images."
        ),
        "input_type": "bi_temporal_pair",
    },

    "change_vqa": {
        "name": "Bi-temporal Change-VQA",
        "type": "vision_language_change_analysis",
        "description": (
            "Answers natural-language questions "
            "about changes between two satellite images."
        ),
        "input_type": "bi_temporal_pair",
    },

    "optical_sar": {
        "name": "OpticalSARChangeNet",
        "type": "multimodal_change_detection",
        "description": (
            "Performs bi-temporal change detection using "
            "optical and SAR imagery with multimodal fusion."
        ),
        "input_type": "optical_sar_bi_temporal",
    },
}


# ============================================================
# MODEL FACTORIES
# ============================================================

_shared_vlm = None
_change_detection_model = None
_optical_sar_model = None
_change_vqa_model = None


def get_model_info(task: str) -> dict:

    if task not in MODEL_REGISTRY:
        raise ValueError(
            f"No model registered for task: {task}"
        )

    return MODEL_REGISTRY[task]


def get_model(task: str):

    global _shared_vlm
    global _change_detection_model
    global _optical_sar_model
    global _change_vqa_model

    # --------------------------------------------------------
    # Shared Qwen VLM
    # --------------------------------------------------------

    if task in ["vqa", "captioning"]:

        if _shared_vlm is None:
            _shared_vlm = get_shared_vlm()
        return _shared_vlm

    # --------------------------------------------------------
    # Change Detection
    # --------------------------------------------------------

    if task == "change_detection":
        if _change_detection_model is None:
            _change_detection_model = get_temporal_change_detection_model()
        return _change_detection_model

    # --------------------------------------------------------
    # Change-VQA
    # --------------------------------------------------------

    if task == "change_vqa":
        if _change_vqa_model is None:
            _change_vqa_model = ChangeVQA()
        return _change_vqa_model

    # --------------------------------------------------------
    # Optical + SAR
    # --------------------------------------------------------

    if task == "optical_sar":
        if _optical_sar_model is None:
            _optical_sar_model = get_optical_sar_model()
        return _optical_sar_model

    raise ValueError(
        f"No model available for task: {task}"
    )


def is_model_loaded(task: str) -> bool:

    global _shared_vlm
    global _change_detection_model
    global _optical_sar_model
    global _change_vqa_model

    if task in ["vqa", "captioning"]:
        return _shared_vlm is not None

    if task == "change_detection":
        return _change_detection_model is not None

    if task == "change_vqa":
        return _change_vqa_model is not None

    if task == "optical_sar":
        return _optical_sar_model is not None

    return False


def get_available_tasks() -> list[str]:
    return list(MODEL_REGISTRY.keys())