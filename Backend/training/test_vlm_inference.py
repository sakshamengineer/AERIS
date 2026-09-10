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

if torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32


print("=" * 60)
print("VLM INFERENCE TEST")
print("=" * 60)

print(f"\nDevice: {DEVICE}")
print(f"Model:  {MODEL_NAME}")


# ============================================================
# CHECK IMAGE
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Satellite image not found:\n{IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"Image size: {image.size}")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Processor loaded.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading VLM model...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
    device_map="auto",
)

model.eval()

print("Model loaded.")


# ============================================================
# QUESTION
# ============================================================

question = (
    "Describe the main land cover visible in this "
    "satellite image."
)

print("\nQuestion:")
print(question)


# ============================================================
# MESSAGE
# ============================================================

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
# CHAT TEMPLATE
# ============================================================

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print("\nChat template created.")


# ============================================================
# PROCESS VISION INPUT
# ============================================================

image_inputs, video_inputs = process_vision_info(
    messages
)

print("Vision input processed.")


# ============================================================
# PROCESS MODEL INPUTS
# ============================================================

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)


# ============================================================
# MOVE TENSORS TO DEVICE
# ============================================================

inputs = {
    key: value.to(DEVICE)
    if isinstance(value, torch.Tensor)
    else value
    for key, value in inputs.items()
}


print("Model inputs prepared.")


# ============================================================
# GENERATE
# ============================================================

print("\nGenerating answer...")

with torch.no_grad():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=150,
    )


# ============================================================
# REMOVE INPUT TOKENS
# ============================================================

generated_ids_trimmed = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(
        inputs["input_ids"],
        generated_ids,
    )
]


# ============================================================
# DECODE
# ============================================================

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("VLM RESPONSE")
print("=" * 60)

print(output_text)

print("\n" + "=" * 60)
print("VLM INFERENCE SUCCESSFUL")
print("=" * 60)