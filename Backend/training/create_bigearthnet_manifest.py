import json
from pathlib import Path
import sys

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------

PROJECT_ROOT = (Path(__file__).resolve().parent.parent)
sys.path.append(str(PROJECT_ROOT))
from training.bigearthnet_dataset import (BigEarthNetDataset,)


MAX_ANNOTATIONS = 1000

OUTPUT_DIR = (PROJECT_ROOT/ "data"/ "bigearthnet")
MANIFEST_PATH = (OUTPUT_DIR/ "training_manifest.json")


# ------------------------------------------------------------
# CREATE MANIFEST
# ------------------------------------------------------------

def create_manifest():

    print("=" * 60)
    print("BIGEARTHNET TRAINING MANIFEST")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD TRAINING ANNOTATIONS
    # --------------------------------------------------------

    dataset = BigEarthNetDataset(split="train")
    total = min(MAX_ANNOTATIONS,len(dataset))
    print(f"\nReading first {total} training annotations...")

    # --------------------------------------------------------
    # UNIQUE PATCHES
    # --------------------------------------------------------

    patches = {}

    # --------------------------------------------------------
    # PROCESS ANNOTATIONS
    # --------------------------------------------------------

    for index in range(total):
        example = dataset[index]
        patch_id = example.get("patch_id")

        s1_name = example.get("s1_name")

        if not patch_id:
            continue

        if patch_id not in patches:

            patches[patch_id] = {
                "patch_id": patch_id,
                "s1_name": s1_name,
                "annotations": [],
            }

        patches[patch_id]["annotations"].append(
            {
                "id": example.get("ID"),

                "input": example.get(
                    "input",
                    "",
                ),

                "output": example.get(
                    "output",
                    "",
                ),

                "type": example.get(
                    "type"
                ),

                "category": example.get(
                    "category"
                ),

                "split": example.get(
                    "split"
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True,exist_ok=True,)

    manifest = {
        "dataset": (
            "BIFOLD-BigEarthNetv2-0/"
            "BigEarthNet.txt"
        ),

        "annotation_count": total,

        "unique_patch_count": len(
            patches
        ),

        "patches": list(
            patches.values()
        ),
    }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nManifest created successfully.")
    print(f"\nAnnotations processed: {total}")
    print("Unique patches: "f"{len(patches)}")
    print("\nManifest:")
    print(MANIFEST_PATH)
    print("\n" + "=" * 60)
    print("FIRST PATCHES")
    print("=" * 60)

    for patch_index, patch in enumerate(list(patches.values())[:5]):
        print(f"\nPatch #{patch_index + 1}")
        print(f"S2: {patch['patch_id']}")
        print(f"S1: {patch['s1_name']}")
        print("Annotations: "f"{len(patch['annotations'])}")

if __name__ == "__main__":
    create_manifest()