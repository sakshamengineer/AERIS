import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.router import (identify_task,get_task_description)

test_cases = [
    (
        "What type of land cover is visible?",
        1,
    ),
    (
        "Describe the scene in this image.",
        1,
    ),
    (
        "What changed between these two images?",
        2,
    ),
    (
        "Use the optical and SAR images together.",
        2,
    ),
]


for query, number_of_images in test_cases:

    task = identify_task(
        query,
        number_of_images,
    )

    description = get_task_description(
        task
    )

    print("=" * 60)
    print("Query:", query)
    print("Images:", number_of_images)
    print("Task:", task)
    print("Description:", description)