import sys
from pathlib import Path

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

# ------------------------------------------------------------
# IMPORT SHARED VLM
# ------------------------------------------------------------

from models.shared_vlm import get_shared_vlm


# ------------------------------------------------------------
# TEST IMAGE
# ------------------------------------------------------------

IMAGE_PATH = "data/samples/test1.png"


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("SATQUERY AI - VLM TEST")
    print("=" * 60)

    print("\nLoading Qwen2.5-VL...")

    model = get_shared_vlm()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # TEST VQA
    # --------------------------------------------------------

    question = (
        "What type of land cover is visible "
        "in this satellite image?"
    )

    print("\nQuestion:")
    print(question)

    print("\nGenerating answer...")

    answer = model.predict(
        image=IMAGE_PATH,
        question=question,
    )

    print("\nAnswer:")
    print(answer)

    # --------------------------------------------------------
    # TEST CAPTIONING
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("Testing captioning...")

    caption = model.caption(
        image=IMAGE_PATH
    )

    print("\nCaption:")
    print(caption)

    print("\n" + "=" * 60)
    print("VLM TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()