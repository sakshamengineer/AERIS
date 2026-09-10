import torch
from pathlib import Path

from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "processed"
    / "S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_37_88_rgb.png"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    DTYPE = torch.float16
else:
    DTYPE = torch.float32


print("=" * 60)
print("DIRECT QWEN VISION TEST")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Model: {MODEL_NAME}")


# ============================================================
# IMAGE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"Image: {image.size}")


# ============================================================
# PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Processor loaded.")


# ============================================================
# MODEL
# ============================================================

print("\nLoading model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
)

model.eval()

print("Model loaded.")


# ============================================================
# MESSAGE
# ============================================================

question = (
    "What objects or land cover types can you see "
    "in this satellite image? Give a short answer."
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": str(IMAGE_PATH),
            },
            {
                "type": "text",
                "text": question,
            },
        ],
    }
]


# ============================================================
# PREPARE VISION INPUT
# ============================================================

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs = process_vision_info(
    messages
)


# ============================================================
# PROCESS
# ============================================================

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


print("\nProcessor output:")

for key, value in inputs.items():

    if isinstance(value, torch.Tensor):
        print(
            f"  {key}: "
            f"{tuple(value.shape)} "
            f"{value.dtype}"
        )
    else:
        print(
            f"  {key}: "
            f"{type(value)}"
        )


# ============================================================
# MOVE TO MODEL DEVICE
# ============================================================

model_device = next(
    p for p in model.parameters()
    if p.device.type != "meta"
).device

print(f"\nModel active device: {model_device}")

for key, value in inputs.items():

    if isinstance(value, torch.Tensor):

        inputs[key] = value.to(
            model_device
        )


# ============================================================
# CHECK VISION INPUT
# ============================================================

print("\nVision tensors:")

print(
    "pixel_values:",
    inputs["pixel_values"].shape
)

print(
    "image_grid_thw:",
    inputs["image_grid_thw"]
)


# ============================================================
# GENERATION
# ============================================================

print("\nQuestion:")
print(question)

print("\nGenerating...")

with torch.inference_mode():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
    )


# ============================================================
# TRIM INPUT TOKENS
# ============================================================

input_token_count = inputs["input_ids"].shape[1]

generated_tokens = generated_ids[
    :, input_token_count:
]


# ============================================================
# DECODE
# ============================================================

response = processor.batch_decode(
    generated_tokens,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

print(response)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)