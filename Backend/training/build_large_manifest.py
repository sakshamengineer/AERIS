import json
import random
from pathlib import Path
from datasets import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "bigearthnet"
    / "training_manifest_large.json"
)

# Number of BigEarthNet.txt records to inspect
SCAN_LIMIT = 100_000

# Target number of unique patches
TARGET_PATCHES = 500

RANDOM_SEED = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_bigearthnet():

    print("=" * 60)
    print("BUILDING LARGE BIGEARTHNET MANIFEST")
    print("=" * 60)

    print("\nLoading BigEarthNet.txt...")

    dataset = load_dataset(
        DATASET_NAME,
        split="all_data"
    )

    print(
        f"Dataset loaded: {len(dataset):,} records"
    )

    return dataset


# ============================================================
# COLLECT TRAINING PATCHES
# ============================================================

def collect_patches(dataset):

    print(
        f"\nScanning up to "
        f"{SCAN_LIMIT:,} annotations..."
    )

    random.seed(RANDOM_SEED)

    patches = {}

    scanned = 0

    for record in dataset:

        scanned += 1

        # ----------------------------------------------------
        # Only use training records
        # ----------------------------------------------------

        if record.get("split") != "train":
            continue

        patch_id = record.get("patch_id")

        if not patch_id:
            continue

        # ----------------------------------------------------
        # Create patch entry
        # ----------------------------------------------------

        if patch_id not in patches:

            patches[patch_id] = {
                "patch_id": patch_id,
                "s1_name": record.get("s1_name"),
                "annotations": []
            }

        # ----------------------------------------------------
        # Add annotation
        # ----------------------------------------------------

        annotation = {
            "id": record.get("ID"),
            "input": record.get("input"),
            "output": record.get("output"),
            "type": record.get("type"),
            "category": record.get("category"),
            "split": record.get("split")
        }

        patches[patch_id]["annotations"].append(
            annotation
        )

        # ----------------------------------------------------
        # Stop once we have enough patches
        # ----------------------------------------------------

        if len(patches) >= TARGET_PATCHES:

            break

        # ----------------------------------------------------
        # Safety limit
        # ----------------------------------------------------

        if scanned >= SCAN_LIMIT:

            break

    print(
        f"\nAnnotations scanned: {scanned:,}"
    )

    print(
        f"Unique patches found: {len(patches)}"
    )

    return list(patches.values())


# ============================================================
# SHUFFLE PATCHES
# ============================================================

def prepare_manifest(patches):

    random.seed(RANDOM_SEED)

    random.shuffle(patches)

    # Limit to target
    patches = patches[:TARGET_PATCHES]

    return patches


# ============================================================
# SAVE
# ============================================================

def save_manifest(patches):

    manifest = {
        "dataset": DATASET_NAME,
        "annotation_count": sum(
            len(p["annotations"])
            for p in patches
        ),
        "unique_patch_count": len(patches),
        "patches": patches
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nManifest saved to:\n"
        f"{OUTPUT_PATH}"
    )

    return manifest


# ============================================================
# SUMMARY
# ============================================================

def print_summary(manifest):

    print("\n" + "=" * 60)
    print("MANIFEST SUMMARY")
    print("=" * 60)

    print(
        f"Unique patches: "
        f"{manifest['unique_patch_count']}"
    )

    print(
        f"Total annotations: "
        f"{manifest['annotation_count']}"
    )

    task_distribution = {}

    for patch in manifest["patches"]:

        for annotation in patch["annotations"]:

            annotation_type = (
                annotation["type"]
                or "unknown"
            ).lower()

            if annotation_type == "captioning":
                task = "captioning"

            elif annotation_type == "bounding box":
                task = "grounding"

            else:
                task = "vqa"

            task_distribution[task] = (
                task_distribution.get(task, 0) + 1
            )

    print("\nTask distribution:")

    for task, count in sorted(
        task_distribution.items()
    ):

        print(
            f"  {task}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_bigearthnet()

    patches = collect_patches(
        dataset
    )

    patches = prepare_manifest(
        patches
    )

    manifest = save_manifest(
        patches
    )

    print_summary(
        manifest
    )

    print("\n" + "=" * 60)
    print("LARGE MANIFEST CREATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()