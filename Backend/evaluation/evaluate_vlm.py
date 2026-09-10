"""
Evaluation for the shared fine-tuned VLM (models/shared_vlm.py,
weights in models/satquery_vlm/) across every task it powers:
VQA, captioning, and grounding.

WHY THIS ONE REPORTS "pending" IN MOST ENVIRONMENTS:
Unlike change_detection and optical_sar, this model needs its
~3B-parameter base (Qwen/Qwen2.5-VL-3B-Instruct) downloaded from
Hugging Face plus the LoRA adapter loaded on top, and realistically
a GPU to run in reasonable time. That is a normal requirement for
your own dev machine / server, but it isn't something a sandboxed
evaluation environment (or CI) can always satisfy. So this script:

  1. Tries to import torch/transformers/peft and load the model.
  2. If that fails (no GPU, no internet to Hugging Face, out of
     memory, etc.) it returns a clearly-labeled "pending" result
     instead of guessing or fabricating numbers - the frontend
     shows this as "Evaluation pending" rather than a fake score.
  3. If the model DOES load, it runs the real held-out split from
     evaluation/build_eval_split.py and computes real metrics:
       - VQA (binary/mcq):        accuracy, macro-F1
       - Grounding (bounding box): mean IoU vs. ground-truth box
       - Captioning:               BLEU-4, ROUGE-L
     

To actually produce numbers: run this on a machine with a GPU and
internet access to huggingface.co,
`pip install torch transformers peft accelerate bitsandbytes`, then:

    python evaluation/build_eval_split.py
    python evaluation/evaluate_vlm.py
"""

import json
import re
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from evaluation import metrics  # noqa: E402

EVAL_SPLIT_PATH = BACKEND_ROOT / "data" / "bigearthnet" / "eval_split.json"

REQUIRED_TASKS = ["vqa", "grounding", "captioning"]

PENDING_REASON_TEMPLATE = (
    "The fine-tuned VLM (Qwen2.5-VL-3B base + LoRA adapter) could not be "
    "loaded in this environment ({reason}). Evaluation metrics for VQA, "
    "captioning, and grounding depend on running real "
    "inference with this model, so no numbers are reported here rather "
    "than showing placeholders. Run `python evaluation/evaluate_vlm.py` "
    "on a machine with a GPU and Hugging Face access to populate this."
)


def _try_load_model():
    try:
        from models.shared_vlm import get_shared_vlm

        model = get_shared_vlm()

        return model, None
    except Exception as error:  # noqa: BLE001 - deliberately broad
        return None, str(error)


def _load_eval_split():
    if not EVAL_SPLIT_PATH.exists():
        return []

    with open(EVAL_SPLIT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_image_path(record):
    # Paths in vlm_training.json were written on Windows
    # (backslashes); normalize and resolve relative to BACKEND_ROOT.
    relative = record["image"].replace("\\", "/")

    return BACKEND_ROOT / relative


def _parse_box(text):
    numbers = [float(value) for value in re.findall(r"-?\d+\.?\d*", text)]

    if len(numbers) < 4:
        return None

    return numbers[:4]


def _pending_result():
    return {
        "model": "satquery_vlm",
        "display_name": "VQA / Captioning / Grounding (Shared VLM)",
        "model_type": "Fine-tuned Qwen2.5-VL-3B-Instruct (LoRA adapter)",
        "status": "pending",
        "methodology": (
            "Held-out split of data/bigearthnet/vlm_training.json "
            f"({len(_load_eval_split())} examples across VQA, grounding, "
            "and captioning) via evaluation/build_eval_split.py. Scoring: "
            "accuracy + macro-F1 for VQA, mean IoU for grounding, "
            "BLEU-4 + ROUGE-L for captioning."
        ),
        "sample_size": len(_load_eval_split()),
        "metrics": {},
    }


def run():
    model, load_error = _try_load_model()

    if model is None:
        result = _pending_result()
        result["pending_reason"] = PENDING_REASON_TEMPLATE.format(
            reason=load_error or "unknown import/load failure"
        )

        return result

    eval_split = _load_eval_split()

    vqa_true, vqa_pred = [], []
    grounding_ious = []
    caption_bleu, caption_rouge = [], []
    runtimes = []
    skipped = 0

    for record in eval_split:
        image_path = _resolve_image_path(record)

        if not image_path.exists():
            skipped += 1
            continue

        task = record.get("task")
        instruction = record.get("instruction", "")
        answer = record.get("answer", "")

        start = time.time()

        try:
            if task == "vqa":
                prediction = model.predict_vqa(str(image_path), instruction)
                vqa_true.append(answer)
                vqa_pred.append(prediction)

            elif task == "grounding":
                prediction = model.predict(str(image_path), instruction)
                true_box = _parse_box(answer)
                pred_box = _parse_box(prediction)

                if true_box and pred_box:
                    grounding_ious.append(metrics.iou_boxes(true_box, pred_box))

            elif task == "captioning":
                prediction = model.predict_caption(str(image_path))
                caption_bleu.append(metrics.bleu_n(answer, prediction, n=4))
                caption_rouge.append(metrics.rouge_l(answer, prediction))

        except Exception:  # noqa: BLE001
            skipped += 1
            continue

        runtimes.append(time.time() - start)

    vqa_accuracy = metrics.accuracy(vqa_true, vqa_pred) if vqa_true else None
    _, _, vqa_f1 = (
        metrics.precision_recall_f1(vqa_true, vqa_pred) if vqa_true else (0, 0, None)
    )

    import numpy as np

    return {
        "model": "satquery_vlm",
        "display_name": "VQA / Captioning / Grounding (Shared VLM)",
        "model_type": "Fine-tuned Qwen2.5-VL-3B-Instruct (LoRA adapter)",
        "status": "evaluated",
        "methodology": (
            "Held-out split of data/bigearthnet/vlm_training.json "
            f"({len(eval_split)} examples) via evaluation/build_eval_split.py."
        ),
        "sample_size": len(eval_split) - skipped,
        "skipped": skipped,
        "metrics": {
            "vqa_accuracy": round(vqa_accuracy, 4) if vqa_accuracy is not None else None,
            "vqa_macro_f1": round(vqa_f1, 4) if vqa_f1 is not None else None,
            "grounding_mean_iou": (
                round(float(np.mean(grounding_ious)), 4)
                if grounding_ious
                else None
            ),
            "captioning_bleu4": (
                round(float(np.mean(caption_bleu)), 4) if caption_bleu else None
            ),
            "captioning_rouge_l": (
                round(float(np.mean(caption_rouge)), 4) if caption_rouge else None
            ),
            "mean_inference_time_sec": (
                round(float(np.mean(runtimes)), 4) if runtimes else None
            ),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
