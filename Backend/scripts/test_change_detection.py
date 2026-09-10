import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.temporal_change_detection import get_temporal_change_detection_model

model = get_temporal_change_detection_model()

result = model.predict(
    image1="dataset/samples/test.jpg",
    image2="dataset/samples/test.tiff",
    output_path="outputs/change_map.png",
)

print("\nSiameseTemporalCD Change Detection Result")
print("=" * 50)

for key, value in result.items():
    print(f"{key}: {value}")
