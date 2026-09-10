import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from agent.compatibility import validate_inputs

def main():

    print("=" * 60)
    print("SATQUERY AI - COMPATIBILITY CHECK TEST")
    print("=" * 60)

    result = validate_inputs(
        images=[
            "data/samples/test.jpg",
            "data/samples/test.tiff",
        ],
        query="Analyze these optical and SAR images together.",
        task="optical_sar",
    )

    print("\nValid:")
    print(result["valid"])

    print("\nMessage:")
    print(result.get("message"))

    print("\nDimension Status:")
    print(result.get("dimension_status"))

    print("\nCRS Status:")
    print(result.get("crs_status"))

    print("\nImage Metadata:")
    for metadata in result.get("metadata", []):
        print("-" * 40)
        for key, value in metadata.items():
            print(f"{key}: {value}")

    if not result["valid"]:
        print("\nError:")
        print(result.get("error"))


if __name__ == "__main__":
    main()