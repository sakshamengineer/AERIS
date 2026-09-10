from pathlib import Path
import json
from datasets import load_dataset


class BigEarthNetDataset:
    """
    BigEarthNet.txt dataset interface for SatQuery AI.

    BigEarthNet.txt is exposed by Hugging Face as one dataset
    split called `all_data`.

    The actual train/validation/test designation is stored
    inside the `split` column of each record.

    This class therefore:
        1. Loads the all_data split.
        2. Filters by the internal split column.
        3. Identifies VQA/captioning/grounding examples.
        4. Converts records to the SatQuery format.

    Training is intentionally NOT implemented here.
    """

    DATASET_NAME = ("BIFOLD-BigEarthNetv2-0/BigEarthNet.txt")
    VALID_SPLITS = {"train","validation","test","all",}

    def __init__(self,split: str = "train",cache_dir: str | None = None):

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of: "
                f"{sorted(self.VALID_SPLITS)}"
            )

        self.requested_split = split
        self.cache_dir = cache_dir

        print("=" * 60)
        print("Loading BigEarthNet.txt")
        print("=" * 60)
        print(f"\nDataset: {self.DATASET_NAME}")
        print("\nHugging Face split: all_data")
        print(f"Requested internal split: {split}")

        self.dataset = load_dataset(self.DATASET_NAME,split="all_data",cache_dir=self.cache_dir)

        print("\nDataset loaded successfully.")
        print(f"Total examples: {len(self.dataset)}")
        print(f"Columns: {self.dataset.column_names}")

        if split == "all":
            self.filtered_dataset = self.dataset
        else:
            self.filtered_dataset = self.dataset.filter(lambda example:example["split"] == split)

        print(
            f"\nExamples in '{split}': "
            f"{len(self.filtered_dataset)}"
        )

    def __len__(self):
        return len(self.filtered_dataset)

    def __getitem__(self, index: int):
        return self.filtered_dataset[index]

    # ============================================================
    # SHOW RAW EXAMPLE
    # ============================================================

    def show_example(self,index: int = 0,):

        example = self.filtered_dataset[index]
        print("\n" + "=" * 60)
        print(f"BIGEARTHNET EXAMPLE #{index}")
        print("=" * 60)

        print(
            json.dumps(
                example,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

        return example

    @staticmethod
    def identify_task(example: dict) -> str:
        """
        Convert the BigEarthNet annotation type into
        a SatQuery task.

        Dataset examples observed so far:

            binary       -> VQA
            mcq          -> VQA
            captioning   -> Captioning
            bounding box -> Grounding
        """

        example_type = str(example.get("type","",)).lower()
        category = str(example.get("category","",)).lower()

        # --------------------------------------------------------
        # CAPTIONING
        # --------------------------------------------------------

        if (
            "caption" in example_type
            or "caption" in category
        ):
            return "captioning"

        # --------------------------------------------------------
        # GROUNDING
        # --------------------------------------------------------

        if (
            "bounding" in example_type
            or "ground" in example_type
            or "box" in example_type
            or "ground" in category
            or "localiz" in category
        ):
            return "grounding"

        # --------------------------------------------------------
        # VQA
        # --------------------------------------------------------

        if (
            example_type in {
                "binary",
                "mcq",
                "vqa",
            }
        ):
            return "vqa"

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        instruction = str(example.get("input","",)).lower()

        if (
            "describe" in instruction
            or "description" in instruction
            or "caption" in instruction
            or "overview" in instruction
        ):
            return "captioning"

        if (
            "bounding box" in instruction
            or "locate" in instruction
            or "highlight" in instruction
        ):
            return "grounding"

        if instruction:
            return "vqa"

        return "unknown"

    # ============================================================
    # CONVERT EXAMPLE
    # ============================================================

    @classmethod
    def convert_example(cls,example: dict,) -> dict:

        task = cls.identify_task(example)

        return {
            "id": example.get("ID"),
            "task": task,
            "instruction": example.get("input","",),
            "answer": example.get("output","",),
            "patch_id": example.get("patch_id"),
            "s1_name": example.get("s1_name"),
            "split": example.get("split"),
            "latitude": example.get("latitude"),
            "longitude": example.get("longitude"),
            "country": example.get("country"),
            "season": example.get("season"),
            "climate_zone": example.get("climate_zone"),
        }

    def get_converted_example(self,index: int = 0) -> dict:

        example = self.filtered_dataset[index]
        return self.convert_example(example)

    def count_tasks(self,max_examples: int | None = None,) -> dict:
        counts = {
            "vqa": 0,
            "captioning": 0,
            "grounding": 0,
            "unknown": 0,
        }

        total = len(self.filtered_dataset)

        if max_examples is not None:
            total = min(total,max_examples,)

        print(f"\nInspecting {total} examples...")

        for index in range(total):
            example = self.filtered_dataset[index]
            task = self.identify_task(example)
            counts[task] += 1
        return counts

    def get_task_example(self,task: str,):
        for index in range(len(self.filtered_dataset)):
            example = self.filtered_dataset[index]
            detected_task = (self.identify_task(example))
            if detected_task == task:
                return self.convert_example(example)
        return None

if __name__ == "__main__":

    dataset = BigEarthNetDataset(split="train")
    
    dataset.show_example(0)

    print("\n" + "=" * 60)
    print("CONVERTED EXAMPLE")
    print("=" * 60)
    converted = (dataset.get_converted_example(0))
    print(
        json.dumps(
            converted,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    counts = dataset.count_tasks(max_examples=1000)
    print("\n" + "=" * 60)
    print("TASK COUNTS — FIRST 1000 TRAIN EXAMPLES")
    print("=" * 60)
    for task, count in counts.items():
        print(f"{task:15s}: {count}")