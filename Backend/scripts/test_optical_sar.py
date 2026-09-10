"""Smoke test for the trained four-input OpticalSARChangeNet."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.optical_sar_inference import get_optical_sar_model


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python scripts/test_optical_sar.py "
            "<optical_t1> <optical_t2> <sar_t1> <sar_t2>"
        )

    model = get_optical_sar_model()
    result = model.predict(
        optical_t1=sys.argv[1],
        optical_t2=sys.argv[2],
        sar_t1=sys.argv[3],
        sar_t2=sys.argv[4],
    )

    for key in (
        "success",
        "model",
        "checkpoint",
        "change_percentage",
        "mean_change_probability",
        "max_change_probability",
        "confidence",
        "threshold",
        "change_map",
        "probability_map",
        "sar_fusion_map",
    ):
        print(f"{key}: {result.get(key)}")


if __name__ == "__main__":
    main()
