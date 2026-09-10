import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "vlm_training.json"
)

BATCH_SIZE = 2
NUM_WORKERS = 0


# ============================================================
# VLM DATASET
# ============================================================

class VLMTrainingDataset(Dataset):

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.dataset_path}"
            )

        with open(self.dataset_path,"r",encoding="utf-8") as f:
            self.examples = json.load(f)

        print(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        image_path = (PROJECT_ROOT / example["image"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "instruction": example["instruction"],
            "answer": example["answer"],
            "task": example["task"],
            "patch_id": example["patch_id"],
        }


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("=" * 60)
    print("TESTING PYTORCH VLM DATASET")
    print("=" * 60)

    dataset = VLMTrainingDataset(DATASET_PATH)

    print(f"\nDataset size: {len(dataset)}")

    # --------------------------------------------------------
    # Test single example
    # --------------------------------------------------------

    sample = dataset[0]

    print("\nSingle example:")
    print("-" * 40)

    print(f"Image size:    {sample['image'].size}")
    print(f"Task:          {sample['task']}")
    print(f"Patch ID:      {sample['patch_id']}")
    print(f"Instruction:   {sample['instruction']}")
    print(f"Answer:        {sample['answer']}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=lambda batch: batch,
    )

    batch = next(iter(dataloader))

    print("\nDataLoader test:")
    print("-" * 40)

    print(f"Batch size: {len(batch)}")

    for i, item in enumerate(batch):

        print(f"\nBatch item {i + 1}:")
        print(f"  Image: {item['image'].size}")
        print(f"  Task: {item['task']}")
        print(f"  Question: {item['instruction'][:100]}")
        print(f"  Answer: {item['answer']}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()