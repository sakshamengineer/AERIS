import sys
from pathlib import Path
from preprocessing.loader import load_image, get_image_info

sys.path.append(str(Path(__file__).resolve().parent.parent))

image, metadata = load_image("data/samples/test.jpg")

print("Image shape:")
print(image.shape)

print("\nMetadata:")
print(get_image_info(metadata))