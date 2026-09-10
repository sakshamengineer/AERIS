from preprocessing.optical_sar_preprocessor import (
    OpticalSARPreprocessor
)


OPTICAL_IMAGE = "data/optical/sample_optical.tif"
SAR_IMAGE = "data/sar/sample_sar.tif"


def main():

    print("=" * 60)
    print("SATQUERY AI")
    print("OPTICAL + SAR PREPROCESSING TEST")
    print("=" * 60)

    processor = OpticalSARPreprocessor()

    result = processor.process(
        optical_path=OPTICAL_IMAGE,
        sar_path=SAR_IMAGE
    )

    print("\nProcessing Result")
    print("=" * 60)

    print(
        "Success:",
        result["success"]
    )

    print(
        "Optical shape:",
        result["optical"].shape
    )

    print(
        "SAR shape:",
        result["sar"].shape
    )

    print(
        "Output size:",
        result["output_shape"]
    )

    print(
        "Warnings:",
        result["warnings"]
    )

    processor.save_numpy(
        result
    )


if __name__ == "__main__":
    main()