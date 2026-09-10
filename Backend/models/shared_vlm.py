import torch
from pathlib import Path
from PIL import Image

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from models.land_registry import desc


# ============================================================
# SHARED SATQUERY VLM
# ============================================================


class get_shared_vlm:
    """
    Shared fine-tuned Vision-Language Model for SatQuery AI.

    Supports:
        - Visual Question Answering
        - Satellite Image Captioning

    The same model instance is shared between both tasks.
    """

    MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    ADAPTER_PATH = (
        PROJECT_ROOT
        / "models"
        / "satquery_vlm"
    )

    # Keep visual input reasonably small for 6 GB VRAM.
    MAX_IMAGE_SIZE = 512

    def __init__(self):

        self.adapter_path = Path(
            self.ADAPTER_PATH
        )

        if not self.adapter_path.exists():
            raise FileNotFoundError(
                f"VLM adapter not found:\n"
                f"{self.adapter_path}"
            )

        self.model = None
        self.processor = None

        self._load_model()

    # ========================================================
    # MODEL LOADING
    # ========================================================

    def _load_model(self):

        print("=" * 60)
        print("LOADING SHARED SATQUERY VLM")
        print("=" * 60)

        if torch.cuda.is_available():

            print(
                f"\nUsing GPU: "
                f"{torch.cuda.get_device_name(0)}"
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            base_model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    self.MODEL_NAME,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
            )

        else:

            print("\nCUDA unavailable. Using CPU.")

            base_model = (
                Qwen2_5_VLForConditionalGeneration
                .from_pretrained(
                    self.MODEL_NAME,
                    torch_dtype=torch.float32,
                )
            )

        print("\nLoading SatQuery LoRA adapter...")

        self.model = PeftModel.from_pretrained(
            base_model,
            str(self.adapter_path),
        )

        self.model.eval()

        print("Loading processor...")

        self.processor = AutoProcessor.from_pretrained(
            str(self.adapter_path)
        )

        print("\nShared SatQuery VLM loaded successfully.")

    # ========================================================
    # IMAGE
    # ========================================================

    def _load_image(self, image):

        if isinstance(image, (str, Path)):

            image_path = Path(image)

            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image not found:\n"
                    f"{image_path}"
                )

            image = Image.open(
                image_path
            ).convert("RGB")

        elif isinstance(image, Image.Image):

            image = image.convert("RGB")

        else:

            raise TypeError(
                "Image must be a file path "
                "or PIL Image."
            )

        # ----------------------------------------------------
        # Resize large images
        # ----------------------------------------------------

        width, height = image.size

        if max(width, height) > self.MAX_IMAGE_SIZE:

            scale = (
                self.MAX_IMAGE_SIZE
                / max(width, height)
            )

            new_width = int(width * scale)
            new_height = int(height * scale)

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

        return image

    # ========================================================
    # GENERATION
    # ========================================================

    def _generate(
        self,
        image,
        prompt,
        max_new_tokens=128,
    ):

        image = self._load_image(image)

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
                        "text": prompt,
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------
        #
        # With device_map="auto", the model can be split
        # between GPU and CPU.
        #
        # Move inputs to the model's primary device only.
        # Do NOT move every model layer manually.
        # ----------------------------------------------------

        if torch.cuda.is_available():

            input_device = self.model.device

            inputs = {
                key: value.to(input_device)
                if torch.is_tensor(value)
                else value
                for key, value in inputs.items()
            }

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        # ----------------------------------------------------
        # REMOVE INPUT TOKENS
        # ----------------------------------------------------

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(
                inputs["input_ids"],
                generated_ids,
            )
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        output = output_text.strip().upper()

        if output in desc:
            return f'{output} : {desc[output]}'
        else:
            return output.title()

    # ========================================================
    # VQA
    # ========================================================

    def predict_vqa(
        self,
        image,
        question,
    ):

        if not question or not question.strip():

            raise ValueError(
                "Question cannot be empty."
            )

        # Must match the VQA prompt template used in
        # training/vlm_train.py exactly.
        prompt = (
            "You are a remote sensing AI assistant. "
            "Analyze the satellite image and answer the "
            "question accurately.\n\n"
            f"Question: {question}"
        )

        return self._generate(
            image=image,
            prompt=prompt,
            max_new_tokens=128,
        )

    # ========================================================
    # CAPTIONING
    # ========================================================

    DEFAULT_CAPTION_INSTRUCTION = (
        "Describe the satellite image in detail, including "
        "the location, season, and observed land cover."
    )

    def predict_caption(
        self,
        image,
        instruction=None,
    ):

        instruction = (
            instruction.strip()
            if instruction and instruction.strip()
            else self.DEFAULT_CAPTION_INSTRUCTION
        )

        # This must match the captioning prompt template used in
        # training/vlm_train.py exactly (same system framing +
        # "Instruction: ..." suffix). The LoRA adapter is sensitive
        # to prompt format - an out-of-distribution prompt causes
        # it to drift into other tasks' output styles (e.g. the
        # bounding-box format used for "grounding").
        prompt = (
            "You are a remote sensing AI assistant. "
            "Describe the satellite image in detail, "
            "focusing on land cover, objects, and spatial "
            "relationships.\n\n"
            f"Instruction: {instruction}"
        )

        return self._generate(
            image=image,
            prompt=prompt,
            max_new_tokens=192,
        )

    # ========================================================
    # CAPTION ALIAS
    # ========================================================

    def caption(
        self,
        image,
        instruction=None,
    ):

        return self.predict_caption(
            image=image,
            instruction=instruction,
        )

    # ========================================================
    # GENERIC PREDICT
    # ========================================================

    def predict(
        self,
        image,
        question=None,
        task="vqa",
    ):

        if task == "vqa":

            if question is None:

                raise ValueError(
                    "Question is required for VQA."
                )

            return self.predict_vqa(
                image=image,
                question=question,
            )

        if task == "captioning":

            return self.predict_caption(
                image=image,
                instruction=question,
            )

        raise ValueError(
            f"Unsupported VLM task: {task}"
        )

        # ========================================================
    # CHANGE VQA
    # ========================================================

    def predict_change(
        self,
        image_before,
        image_after,
        question,
    ):
        """
        Compare two satellite images and answer
        a question about the observed change.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        image_before = self._load_image(
            image_before
        )

        image_after = self._load_image(
            image_after
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_before,
                    },
                    {
                        "type": "image",
                        "image": image_after,
                    },
                    {
                        "type": "text",
                        "text": (
                            "You are a remote sensing AI assistant. "
                            "You are given two satellite images. "
                            "The first image represents the BEFORE state "
                            "and the second image represents the AFTER state.\n\n"
                            "Compare both images carefully and answer "
                            "the user's question about the change.\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=[
                image_before,
                image_after,
            ],
            padding=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():

            input_device = self.model.device

            inputs = {
                key: value.to(input_device)
                if torch.is_tensor(value)
                else value
                for key, value in inputs.items()
            }

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=192,
                do_sample=False,
                use_cache=True,
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(
                inputs["input_ids"],
                generated_ids,
            )
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text.strip()