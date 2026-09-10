from agent.modality import (
    detect_modality,
    check_optical_sar_pair,
)


def main():

    print("=" * 60)
    print("SATQUERY AI - MODALITY DETECTION TEST")
    print("=" * 60)

    images = [
        "data/samples/test.jpg",
        "data/samples/test.tiff",
    ]

    print("\nIndividual Modality Detection")
    print("-" * 60)

    for image in images:

        result = detect_modality(image)

        print(f"\nImage: {image}")
        print(f"Modality: {result.get('modality')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Reason: {result.get('reason')}")

    print("\n")
    print("=" * 60)
    print("OPTICAL + SAR PAIR CHECK")
    print("=" * 60)

    pair_result = check_optical_sar_pair(
        images,
        declared_modalities=[
            "optical",
            "sar",
        ],
    )
    print("\nValid:")
    print(pair_result.get("valid"))

    print("\nIs Optical + SAR:")
    print(pair_result.get("is_optical_sar"))

    print("\nRequires Confirmation:")
    print(pair_result.get("requires_confirmation"))

    print("\nMessage:")
    print(pair_result.get("message"))

    print("\nModalities:")

    for result in pair_result.get("modalities", []):
        print("-" * 40)

        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()