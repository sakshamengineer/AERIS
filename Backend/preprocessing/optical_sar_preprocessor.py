import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


class OpticalSARPreprocessor:
    """
    Preprocessing pipeline for paired Optical + SAR satellite imagery.

    Expected:
        Optical -> GeoTIFF
        SAR     -> GeoTIFF

    Output:
        Aligned and normalized NumPy arrays.
    """

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # LOAD GEOTIFF
    # ---------------------------------------------------------

    def load_geotiff(self, path):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        with rasterio.open(path) as src:

            data = src.read().astype(np.float32)

            profile = src.profile.copy()

            metadata = {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "crs": str(src.crs) if src.crs else None,
                "transform": list(src.transform),
                "bounds": tuple(src.bounds),
                "dtype": str(src.dtypes[0])
            }

        return data, metadata, profile

    # ---------------------------------------------------------
    # VALIDATE GEOSPATIAL INFORMATION
    # ---------------------------------------------------------

    def validate_metadata(
        self,
        optical_metadata,
        sar_metadata
    ):

        warnings = []

        if optical_metadata["crs"] is None:
            warnings.append(
                "Optical image has no CRS."
            )

        if sar_metadata["crs"] is None:
            warnings.append(
                "SAR image has no CRS."
            )

        if (
            optical_metadata["width"] <= 0
            or optical_metadata["height"] <= 0
        ):
            warnings.append(
                "Invalid Optical dimensions."
            )

        if (
            sar_metadata["width"] <= 0
            or sar_metadata["height"] <= 0
        ):
            warnings.append(
                "Invalid SAR dimensions."
            )

        return warnings

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    def normalize(self, image):

        image = image.astype(np.float32)

        output = np.zeros_like(image)

        for i in range(image.shape[0]):

            band = image[i]

            valid = np.isfinite(band)

            if not np.any(valid):
                continue

            values = band[valid]

            low = np.percentile(values, 2)
            high = np.percentile(values, 98)

            if high - low < 1e-8:

                output[i] = 0

            else:

                clipped = np.clip(
                    band,
                    low,
                    high
                )

                output[i] = (
                    clipped - low
                ) / (
                    high - low
                )

        return output

    # ---------------------------------------------------------
    # RESAMPLE SAR TO OPTICAL GRID
    # ---------------------------------------------------------

    def align_sar_to_optical(
        self,
        optical,
        optical_profile,
        sar,
        sar_profile
    ):

        target_height = optical.shape[1]
        target_width = optical.shape[2]

        aligned_sar = np.zeros(
            (
                sar.shape[0],
                target_height,
                target_width
            ),
            dtype=np.float32
        )

        for band in range(sar.shape[0]):

            reproject(
                source=sar[band],
                destination=aligned_sar[band],

                src_transform=sar_profile["transform"],
                src_crs=sar_profile["crs"],

                dst_transform=optical_profile["transform"],
                dst_crs=optical_profile["crs"],

                resampling=Resampling.bilinear
            )

        return aligned_sar

    # ---------------------------------------------------------
    # SELECT OPTICAL BANDS
    # ---------------------------------------------------------

    def prepare_optical(self, optical):

        if optical.shape[0] >= 3:

            # RGB-style first three bands
            optical = optical[:3]

        elif optical.shape[0] == 1:

            optical = np.repeat(
                optical,
                3,
                axis=0
            )

        else:

            raise ValueError(
                "Optical image must contain at least one band."
            )

        return optical

    # ---------------------------------------------------------
    # SELECT SAR BANDS
    # ---------------------------------------------------------

    def prepare_sar(self, sar):

        if sar.shape[0] == 0:

            raise ValueError(
                "SAR image contains no bands."
            )

        # Baseline:
        # use first SAR band.

        sar = sar[:1]

        return sar

    # ---------------------------------------------------------
    # FULL PIPELINE
    # ---------------------------------------------------------

    def process(self,optical_path,sar_path):

        print("\nLoading Optical image...")

        optical, optical_metadata, optical_profile = (self.load_geotiff(optical_path))

        print(f"Optical shape: {optical.shape}")

        print("\nLoading SAR image...")

        sar, sar_metadata, sar_profile = (self.load_geotiff(sar_path))
        
        print(f"SAR shape: {sar.shape}")

        # Metadata validation

        warnings = self.validate_metadata(
            optical_metadata,
            sar_metadata
        )

        if warnings:

            print("\nWarnings:")

            for warning in warnings:
                print(f" - {warning}")

        # Check CRS before geospatial reprojection

        if (
            optical_profile["crs"] is None
            or sar_profile["crs"] is None
        ):

            raise ValueError(
                "Both Optical and SAR images must have "
                "valid CRS information for geospatial alignment."
            )

        # Prepare channels

        optical = self.prepare_optical(
            optical
        )

        sar = self.prepare_sar(
            sar
        )

        # Normalize

        print("\nNormalizing Optical image...")

        optical = self.normalize(
            optical
        )

        print("Normalizing SAR image...")

        sar = self.normalize(
            sar
        )

        # Align SAR

        print(
            "\nAligning SAR to Optical grid..."
        )

        sar = self.align_sar_to_optical(
            optical,
            optical_profile,
            sar,
            sar_profile
        )

        print(
            f"Aligned SAR shape: {sar.shape}"
        )

        return {
            "success": True,

            "optical": optical,

            "sar": sar,

            "optical_metadata":
                optical_metadata,

            "sar_metadata":
                sar_metadata,

            "output_shape": {
                "height": optical.shape[1],
                "width": optical.shape[2]
            },

            "warnings": warnings
        }

    # ---------------------------------------------------------
    # SAVE OUTPUT
    # ---------------------------------------------------------

    def save_numpy(
        self,
        result,
        output_directory="data/processed/optical_sar"
    ):

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        optical_path = os.path.join(
            output_directory,
            "optical.npy"
        )

        sar_path = os.path.join(
            output_directory,
            "sar.npy"
        )

        np.save(
            optical_path,
            result["optical"]
        )

        np.save(
            sar_path,
            result["sar"]
        )

        print(
            "\nSaved processed data:"
        )

        print(
            f"Optical: {optical_path}"
        )

        print(
            f"SAR: {sar_path}"
        )

        return {
            "optical_path": optical_path,
            "sar_path": sar_path
        }