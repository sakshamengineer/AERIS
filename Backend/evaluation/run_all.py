"""Run the current AERIS model evaluators and write one JSON payload."""

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

OUTPUT_PATH = BACKEND_ROOT / "outputs" / "evaluation_metrics.json"

EVALUATORS = [
    ("evaluation.evaluate_change_detection", "SiameseTemporalCD"),
    ("evaluation.evaluate_optical_sar", "OpticalSARChangeNet"),
    ("evaluation.evaluate_vlm", "Shared VLM (VQA/Captioning/Grounding)"),
]


def run_all():
    results = []

    for module_name, label in EVALUATORS:
        print(f"Running evaluator: {label} ({module_name})")
        try:
            module = __import__(module_name, fromlist=["run"])
            results.append(module.run())
        except Exception as error:  # noqa: BLE001
            print(f"  FAILED: {error}")
            traceback.print_exc()
            results.append({
                "model": module_name.rsplit(".", 1)[-1].replace("evaluate_", ""),
                "display_name": label,
                "status": "error",
                "error": str(error),
                "metrics": {},
            })

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"\nWrote combined results to {OUTPUT_PATH}")
    return payload


if __name__ == "__main__":
    run_all()
