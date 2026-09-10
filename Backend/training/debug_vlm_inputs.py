import torch
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
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
# LOAD IMAGE
# ============================================================

print("=" * 60)
print("DEBUGGING QWEN VISION INPUTS")
print("=" * 60)

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

print(f"\nImage path: {IMAGE_PATH}")
print(f"Image size: {image.size}")
print(f"Image mode: {image.mode}")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Processor loaded.")


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
                "text": (
                    "Describe the main land cover "
                    "visible in this satellite image."
                ),
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

print("\nChat template:")
print("-" * 40)
print(text)


# ============================================================
# PROCESS VISION INFORMATION
# ============================================================

print("\nProcessing vision information...")

image_inputs, video_inputs = process_vision_info(
    messages
)

print("Vision processing complete.")

print("\nImage inputs:")
print(type(image_inputs))

if image_inputs is not None:
    print(f"Number of image inputs: {len(image_inputs)}")

    for i, img in enumerate(image_inputs):
        print(
            f"  Image {i}: "
            f"type={type(img)}, "
            f"size={getattr(img, 'size', 'unknown')}"
        )

print(f"\nVideo inputs: {video_inputs}")


# ============================================================
# PROCESSOR
# ============================================================

print("\nRunning processor...")

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

print("Processor completed.")


# ============================================================
# INSPECT OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("PROCESSOR OUTPUT")
print("=" * 60)

print("\nKeys:")
for key in inputs.keys():
    print(f"  {key}")


print("\nTensor information:")

for key, value in inputs.items():

    if isinstance(value, torch.Tensor):

        print(
            f"  {key}: "
            f"shape={tuple(value.shape)}, "
            f"dtype={value.dtype}, "
            f"device={value.device}"
        )

        if value.numel() > 0:

            print(
                f"       min={value.min().item():.6f}, "
                f"max={value.max().item():.6f}"
            )

    else:

        print(
            f"  {key}: "
            f"type={type(value)}"
        )


# ============================================================
# CRITICAL CHECKS
# ============================================================

print("\n" + "=" * 60)
print("CRITICAL VISION CHECKS")
print("=" * 60)

if "pixel_values" in inputs:

    print("✓ pixel_values FOUND")

    print(
        f"  Shape: "
        f"{tuple(inputs['pixel_values'].shape)}"
    )

else:

    print("✗ pixel_values MISSING")


if "image_grid_thw" in inputs:

    print("✓ image_grid_thw FOUND")

    print(
        f"  Value: "
        f"{inputs['image_grid_thw']}"
    )

else:

    print("✗ image_grid_thw MISSING")


if "input_ids" in inputs:

    print("✓ input_ids FOUND")

    print(
        f"  Shape: "
        f"{tuple(inputs['input_ids'].shape)}"
    )

else:

    print("✗ input_ids MISSING")


print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)