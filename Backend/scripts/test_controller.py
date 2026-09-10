import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from agent.controller import controller


IMAGE_BEFORE = "data/samples/test1.png"
IMAGE_AFTER = "data/samples/test.tiff"


result = controller.analyze(
    query="caption this image",
    images=[
        IMAGE_BEFORE,
    ],
)


print("\n" + "=" * 60)
print("SATQUERY AI - CHANGE DETECTION CONTROLLER TEST")
print("=" * 60)

print("\nSuccess:")
print(result["success"])

print("\nTask:")
print(result.get("task"))

print("\nTask Description:")
print(result.get("task_description"))

print("\nModel:")
print(result.get("model"))

print("\nModel Status:")
print(result.get("model_status"))

print("\nAnswer:")
print(result.get("answer"))

print("\nEvidence:")
print(result.get("evidence"))

print("\nConfidence:")
print(result.get("confidence"))

print("\nExecution Trace:")
print("-" * 60)

for step in result["trace"]:

    print(
        f"[{step['status'].upper()}] "
        f"{step['step']}: "
        f"{step['details']}"
    )