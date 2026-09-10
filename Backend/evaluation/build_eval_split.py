"""
Carves a held-out evaluation split out of
data/bigearthnet/vlm_training.json.

That file is the (only) labeled dataset in this repo for the VQA /
captioning / grounding tasks the fine-tuned VLM handles, and it
currently has no train/test split - training/vlm_train.py uses all
of it. This script deterministically holds out ~15% of examples
(stratified by task, seeded) into data/bigearthnet/eval_split.json
so there is finally a proper, reusable set to score the model
against, instead of "evaluating" on data it was trained on.

Run once. If you retrain the model, make sure vlm_train.py excludes
patch_ids present in eval_split.json.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = BACKEND_ROOT / "dataset" / "bigearthnet" / "vlm_training.json"
OUTPUT_PATH = BACKEND_ROOT / "dataset" / "bigearthnet" / "eval_split.json"

RANDOM_SEED = 42
HOLDOUT_FRACTION = 0.15


def build_split():
    with open(SOURCE_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    by_task = defaultdict(list)

    for record in records:
        by_task[record.get("task", "unknown")].append(record)

    rng = random.Random(RANDOM_SEED)
    held_out = []

    for task, task_records in by_task.items():
        task_records = task_records[:]
        rng.shuffle(task_records)

        cutoff = max(1, int(len(task_records) * HOLDOUT_FRACTION))
        held_out.extend(task_records[:cutoff])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(held_out, file, indent=2)

    return held_out


if __name__ == "__main__":
    split = build_split()

    print(f"Wrote {len(split)} held-out examples to {OUTPUT_PATH}")
