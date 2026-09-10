import json
from pathlib import Path

import torch
from PIL import Image

from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "vlm_training.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "satquery_vlm"
)

# ============================================================
# LOAD DATASET
# ============================================================


def load_training_data():
    print("=" * 60)
    print("LOADING VLM TRAINING DATA")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nTotal examples: {len(data)}")

    dataset = Dataset.from_list(data)

    return dataset


# ============================================================
# MODEL
# ============================================================


def load_model_and_processor():

    print("\n" + "=" * 60)
    print("LOADING QWEN2.5-VL")
    print("=" * 60)

    compute_dtype = torch.float16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME
    )

    model = prepare_model_for_kbit_training(model)

    return model, processor


# ============================================================
# LORA
# ============================================================


def setup_lora(model):

    print("\n" + "=" * 60)
    print("SETTING UP LORA")
    print("=" * 60)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config
    )

    model.print_trainable_parameters()

    return model


# ============================================================
# DATA COLLATOR
# ============================================================


class VLMDataCollator:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):

        full_texts = []
        prompt_texts = []
        images = []

        for example in examples:

            image_path = PROJECT_ROOT / example["image"]

            image = Image.open(image_path).convert("RGB")

            task = example["task"]
            instruction = example["instruction"]
            answer = str(example["answer"])

            # ------------------------------------------------
            # TASK-SPECIFIC PROMPT
            # ------------------------------------------------

            if task == "vqa":

                prompt = (
                    "You are a remote sensing AI assistant. "
                    "Analyze the satellite image and answer the "
                    "question accurately.\n\n"
                    f"Question: {instruction}"
                )

            elif task == "captioning":

                prompt = (
                    "You are a remote sensing AI assistant. "
                    "Describe the satellite image in detail, "
                    "focusing on land cover, objects, and spatial "
                    "relationships.\n\n"
                    f"Instruction: {instruction}"
                )

            elif task == "grounding":

                prompt = (
                    "You are a remote sensing AI assistant. "
                    "Identify the requested region in the satellite "
                    "image and provide its normalized bounding box "
                    "coordinates.\n\n"
                    f"Instruction: {instruction}"
                )

            else:

                prompt = instruction

            # ------------------------------------------------
            # PROMPT-ONLY CONVERSATION
            # ------------------------------------------------

            prompt_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            # ------------------------------------------------
            # FULL CONVERSATION
            # ------------------------------------------------

            full_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": answer,
                        }
                    ],
                },
            ]

            # ------------------------------------------------
            # APPLY QWEN CHAT TEMPLATE
            # ------------------------------------------------

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            images.append(image)

        # ----------------------------------------------------
        # TOKENIZE FULL CONVERSATION
        # ----------------------------------------------------

        batch = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        # ----------------------------------------------------
        # TOKENIZE PROMPTS
        # ----------------------------------------------------

        prompt_batch = self.processor.tokenizer(
            prompt_texts,
            padding=True,
            return_tensors="pt",
        )

        # ----------------------------------------------------
        # CREATE LABELS
        # ----------------------------------------------------

        labels = batch["input_ids"].clone()

        # Ignore padding
        labels[
            labels == self.processor.tokenizer.pad_token_id
        ] = -100

        # Ignore prompt tokens
        for i in range(len(examples)):

            prompt_length = (
                prompt_batch["attention_mask"][i]
                .sum()
                .item()
            )

            labels[i, :prompt_length] = -100

        batch["labels"] = labels

        return batch


# ============================================================
# TRAINING
# ============================================================


def train():

    dataset = load_training_data()

    model, processor = load_model_and_processor()

    model = setup_lora(model)

    model.gradient_checkpointing_enable()

    model.config.use_cache = False

    collator = VLMDataCollator(processor)

    # ========================================================
    # TRAINING CONFIGURATION
    # ========================================================

    training_args = TrainingArguments(

        output_dir=str(OUTPUT_DIR),

        # ----------------------------------------------------
        # BATCHING
        # ----------------------------------------------------

        per_device_train_batch_size=1,

        gradient_accumulation_steps=8,

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        num_train_epochs=1,

        learning_rate=2e-4,

        weight_decay=0.01,

        # ----------------------------------------------------
        # MEMORY OPTIMIZATION
        # ----------------------------------------------------

        gradient_checkpointing=True,

        fp16=True,

        optim="paged_adamw_8bit",

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        logging_steps=10,

        # ----------------------------------------------------
        # CHECKPOINTS
        # ----------------------------------------------------

        save_strategy="steps",

        save_steps=100,

        save_total_limit=2,

        # ----------------------------------------------------
        # DATALOADER
        # ----------------------------------------------------

        dataloader_num_workers=0,

        remove_unused_columns=False,

        # ----------------------------------------------------
        # REPORTING
        # ----------------------------------------------------

        report_to="none",

    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=dataset,

        data_collator=collator,

    )

    # ========================================================
    # START TRAINING
    # ========================================================

    print("=" * 60)
    print("STARTING SATQUERY AI VLM TRAINING")
    print("=" * 60)

    print(f"\nTraining examples: {len(dataset)}")
    print("Epochs: 1")
    print("Batch size: 1")
    print("Gradient accumulation: 8")
    print("Effective batch size: 8")
    print("Learning rate: 2e-4")
    print("Precision: FP16")
    print("Quantization: 4-bit")
    print("Fine-tuning: LoRA")

    trainer.train()

    # ========================================================
    # SAVE MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("SAVING SATQUERY AI VLM")
    print("=" * 60)

    trainer.save_model(str(OUTPUT_DIR))

    processor.save_pretrained(
        str(OUTPUT_DIR)
    )

    print(f"\nModel saved to:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":

    train()