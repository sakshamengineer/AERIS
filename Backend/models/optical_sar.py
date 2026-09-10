import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from PIL import Image


class OpticalSARFusionNetwork(nn.Module):
    """
    Lightweight dual-stream Optical + SAR fusion network.

    Optical and SAR are processed independently before
    their feature representations are fused.
    """

    def __init__(self, optical_channels=3, sar_channels=1):
        super().__init__()

        self.optical_encoder = nn.Sequential(
            nn.Conv2d(
                optical_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
        )

        self.sar_encoder = nn.Sequential(
            nn.Conv2d(
                sar_channels,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                64,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
        )

        self.output_layer = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, optical, sar):

        optical_features = self.optical_encoder(optical)

        sar_features = self.sar_encoder(sar)

        fused_features = torch.cat(
            [
                optical_features,
                sar_features
            ],
            dim=1
        )

        fused_features = self.fusion(
            fused_features
        )

        output = self.output_layer(
            fused_features
        )

        return output, fused_features


class OpticalSARProcessor:

    def __init__(self, device=None):

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = device

        self.model = OpticalSARFusionNetwork()

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"Optical-SAR fusion model initialized "
            f"on {self.device}"
        )

    # --------------------------------------------------
    # IMAGE LOADING
    # --------------------------------------------------

    def _load_image(self, path):

        extension = os.path.splitext(
            path
        )[1].lower()

        if extension in [".tif", ".tiff"]:

            with rasterio.open(path) as src:

                image = src.read()

                metadata = {
                    "width": src.width,
                    "height": src.height,
                    "bands": src.count,
                    "crs": str(src.crs),
                }

            return image, metadata

        else:

            image = Image.open(path).convert(
                "RGB"
            )

            image = np.array(image)

            image = np.transpose(
                image,
                (2, 0, 1)
            )

            metadata = {
                "width": image.shape[2],
                "height": image.shape[1],
                "bands": image.shape[0],
                "crs": None,
            }

            return image, metadata

    # --------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------

    def _normalize(self, image):

        image = image.astype(
            np.float32
        )

        normalized = np.zeros_like(
            image
        )

        for band in range(
            image.shape[0]
        ):

            band_data = image[band]

            min_val = np.nanmin(
                band_data
            )

            max_val = np.nanmax(
                band_data
            )

            if max_val - min_val > 1e-8:

                normalized[band] = (
                    band_data - min_val
                ) / (
                    max_val - min_val
                )

            else:

                normalized[band] = 0.0

        return normalized

    # --------------------------------------------------
    # RESIZE / ALIGN
    # --------------------------------------------------

    def _align_images(
        self,
        optical,
        sar
    ):

        height = min(
            optical.shape[1],
            sar.shape[1]
        )

        width = min(
            optical.shape[2],
            sar.shape[2]
        )

        optical = optical[
            :,
            :height,
            :width
        ]

        sar = sar[
            :,
            :height,
            :width
        ]

        return optical, sar

    # --------------------------------------------------
    # BAND PREPARATION
    # --------------------------------------------------

    def _prepare_optical(
        self,
        optical
    ):

        if optical.shape[0] >= 3:

            optical = optical[:3]

        elif optical.shape[0] == 1:

            optical = np.repeat(
                optical,
                3,
                axis=0
            )

        else:

            raise ValueError(
                "Optical image must contain "
                "at least 1 band."
            )

        return optical

    def _prepare_sar(
        self,
        sar
    ):

        if sar.shape[0] >= 1:

            sar = sar[:1]

        else:

            raise ValueError(
                "SAR image contains no bands."
            )

        return sar

    # --------------------------------------------------
    # SAVE VISUAL EVIDENCE
    # --------------------------------------------------

    def _save_fusion_evidence(
        self,
        fused_output,
        output_path="outputs/optical_sar_fusion.png"
    ):

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        image = np.asarray(
            fused_output,
            dtype=np.float32
        )

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        )

        image = np.clip(
            image,
            0.0,
            1.0
        )

        image_uint8 = (
            image * 255
        ).astype(
            np.uint8
        )

        Image.fromarray(
            image_uint8
        ).save(
            output_path
        )

        return output_path

    # --------------------------------------------------
    # FUSION
    # --------------------------------------------------

    def predict(
        self,
        optical_image,
        sar_image
    ):

        try:

            print(
                "Loading optical image..."
            )

            optical, optical_metadata = (
                self._load_image(
                    optical_image
                )
            )

            print(
                "Loading SAR image..."
            )

            sar, sar_metadata = (
                self._load_image(
                    sar_image
                )
            )

            optical = self._prepare_optical(
                optical
            )

            sar = self._prepare_sar(
                sar
            )

            optical = self._normalize(
                optical
            )

            sar = self._normalize(
                sar
            )

            optical, sar = (
                self._align_images(
                    optical,
                    sar
                )
            )

            optical_tensor = (
                torch.from_numpy(
                    optical
                )
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            sar_tensor = (
                torch.from_numpy(
                    sar
                )
                .unsqueeze(0)
                .float()
                .to(self.device)
            )

            print(
                "Running Optical-SAR fusion..."
            )

            with torch.no_grad():

                output, fused_features = (
                    self.model(
                        optical_tensor,
                        sar_tensor
                    )
                )

            fused_output = (
                output
                .squeeze()
                .cpu()
                .numpy()
            )

            fused_features_np = (
                fused_features
                .squeeze()
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Normalize output
            # ------------------------------------------------

            min_val = fused_output.min()
            max_val = fused_output.max()

            if max_val - min_val > 1e-8:

                fused_output = (
                    fused_output - min_val
                ) / (
                    max_val - min_val
                )

            # ------------------------------------------------
            # Heuristic diagnostic score
            # ------------------------------------------------

            confidence = float(
                np.mean(
                    np.abs(
                        fused_output - 0.5
                    )
                ) * 2
            )

            confidence = min(
                max(
                    confidence,
                    0.0
                ),
                1.0
            )

            # ------------------------------------------------
            # Save visual evidence
            # ------------------------------------------------

            evidence_path = (
                self._save_fusion_evidence(
                    fused_output
                )
            )

            return {

                "success": True,

                "task": "optical_sar",

                "optical_image":
                    optical_image,

                "sar_image":
                    sar_image,

                "optical_metadata":
                    optical_metadata,

                "sar_metadata":
                    sar_metadata,

                "aligned_size": {
                    "width": int(
                        optical.shape[2]
                    ),
                    "height": int(
                        optical.shape[1]
                    )
                },

                "fused_features_shape":
                    list(
                        fused_features_np.shape
                    ),

                "evidence":
                    evidence_path,

                "fused_output":
                    fused_output,

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "confidence_type":
                    "heuristic_diagnostic",

            }

        except Exception as e:

            return {

                "success": False,

                "task": "optical_sar",

                "error": str(e)

            }


# --------------------------------------------------
# SINGLETON
# --------------------------------------------------

_optical_sar_model = None


def get_optical_sar_model():

    global _optical_sar_model

    if _optical_sar_model is None:

        _optical_sar_model = (
            OpticalSARProcessor()
        )

    return _optical_sar_model