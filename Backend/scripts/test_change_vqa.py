import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.change_vqa import ChangeVQA

IMAGE_BEFORE = "data/samples/test.jpg"
IMAGE_AFTER = "data/samples/test.jpg"


def main():

    print("=" * 60)
    print("SATQUERY AI - CHANGE VQA TEST")
    print("=" * 60)

    print("\nLoading shared Change-VQA model...")

    model = ChangeVQA()

    question = """
What changed between the earlier and later
satellite images? Describe any visible
changes in land cover, buildings, roads,
vegetation, or water.
"""

    print("\nGenerating Change-VQA answer...")

    answer = model.predict(
        image_before=IMAGE_BEFORE,
        image_after=IMAGE_AFTER,
        question=question,
    )

    print("\nChange-VQA Answer:")
    print(answer)

    print("\n" + "=" * 60)
    print("CHANGE-VQA TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()