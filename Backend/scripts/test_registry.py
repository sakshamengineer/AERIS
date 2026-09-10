import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent.registry import (
    get_model_info,
    get_model,
    is_model_loaded,
    get_available_tasks,
)


print("Available tasks:")
print(get_available_tasks())


print("\nVQA model information:")
print(get_model_info("vqa"))


print("\nVQA model:")
print(get_model("vqa"))


print("\nIs VQA model loaded?")
print(is_model_loaded("vqa"))