import torch
from pathlib import Path
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ADAPTER_PATH = (
    PROJECT_ROOT
    / "models"
    / "satquery_vlm"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "processed"
)


# ============================================================
# FIND TEST IMAGE
# ============================================================

def find_test_image():

    images = list(
        PROCESSED_DIR.glob("*_rgb.png")
    )

    if not images:
        raise FileNotFoundError(
            f"No processed RGB images found:\n{PROCESSED_DIR}"
        )

    image_path = images[0]

    print("\nTest image:")
    print(image_path)

    return image_path


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 60)
    print("LOADING SATQUERY VLM")
    print("=" * 60)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\nLoading base Qwen2.5-VL...")

    base_model = (Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    )

    print("Loading SatQuery LoRA adapter...")

    model = PeftModel.from_pretrained(base_model,str(ADAPTER_PATH),)

    model.eval()

    processor = AutoProcessor.from_pretrained(str(ADAPTER_PATH))

    print("\nModel loaded successfully.")

    return model, processor


# ============================================================
# INFERENCE
# ============================================================

def generate_answer(model,processor,image,instruction,):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": instruction,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,)
    inputs = processor(text=[text],images=[image],padding=True,return_tensors="pt",)
    model_device = next(model.parameters()).device

    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(model_device)

    with torch.no_grad():

        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    # Remove input tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(
            inputs["input_ids"],
            generated_ids,
        )
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("SATQUERY AI VLM TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image_path = find_test_image()

    image = Image.open(image_path).convert("RGB")

    print(f"Image size: {image.size}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model, processor = load_model()

    # --------------------------------------------------------
    # TEST 1: VQA
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 1: VQA")
    print("=" * 60)

    vqa_question = (
        "Does this satellite image contain forest, "
        "agricultural land, or urban areas? "
        "Describe what you observe."
    )

    print(f"\nQuestion:\n{vqa_question}")

    answer = generate_answer(
        model,
        processor,
        image,
        vqa_question,
    )

    print(f"\nAnswer:\n{answer}")

    # --------------------------------------------------------
    # TEST 2: CAPTIONING
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 2: CAPTIONING")
    print("=" * 60)

    caption_prompt = (
        "Provide a detailed description of this satellite "
        "image. Describe the major land-cover types, "
        "visible objects, and spatial arrangement."
    )

    print(f"\nPrompt:\n{caption_prompt}")

    caption = generate_answer(
        model,
        processor,
        image,
        caption_prompt,
    )

    print(f"\nCaption:\n{caption}")

    # --------------------------------------------------------
    # TEST 3: GROUNDING
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST 3: GROUNDING")
    print("=" * 60)

    grounding_prompt = (
        "Identify the most prominent urban or built-up "
        "region in the satellite image. "
        "Return its normalized bounding box in the format "
        "[x1 y1, x2 y2], where all coordinates are between "
        "0 and 1."
    )

    print(f"\nPrompt:\n{grounding_prompt}")

    grounding = generate_answer(
        model,
        processor,
        image,
        grounding_prompt,
    )

    print(f"\nBounding box:\n{grounding}")

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SATQUERY VLM TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()