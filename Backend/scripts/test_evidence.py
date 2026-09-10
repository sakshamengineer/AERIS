from agent.evidence import (
    create_input_evidence,
    create_change_map_evidence,
    create_optical_sar_evidence,
    create_multi_image_evidence,
)


def main():

    print("=" * 60)
    print("SATQUERY AI - EVIDENCE LAYER TEST")
    print("=" * 60)

    print("\nInput Evidence")
    print("-" * 60)

    print(
        create_input_evidence(
            "data/samples/test.jpg"
        )
    )

    print("\nChange Map Evidence")
    print("-" * 60)

    print(
        create_change_map_evidence(
            "outputs/change_map.png"
        )
    )

    print("\nOptical-SAR Evidence")
    print("-" * 60)

    print(
        create_optical_sar_evidence(
            "outputs/optical_sar_fusion.png"
        )
    )

    print("\nMulti-Image Evidence")
    print("-" * 60)

    print(
        create_multi_image_evidence(
            [
                "data/samples/test.jpg",
                "data/samples/test.tiff",
            ]
        )
    )


if __name__ == "__main__":
    main()