"""Production adapter for the trained SiameseTemporalCD change detector."""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.temporal_optical import SiameseTemporalCD


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "optical" / "best.pt"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "optical.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "change_maps"


class SiameseTemporalCDInference:
    """Load and run the trained SiameseTemporalCD checkpoint.

    The training config in this project defines ImageNet-style statistics for
    raw 8-bit optical imagery, so preprocessing intentionally keeps pixel
    values in the 0-255 domain before mean/std normalization.
    """

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SiameseTemporalCD checkpoint not found: {self.checkpoint_path}"
            )

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        self.checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        cfg = self.checkpoint.get("config", {}) if isinstance(self.checkpoint, dict) else {}
        self.config = cfg
        model_cfg = dict(cfg.get("model", {}))
        dataset_cfg = dict(cfg.get("dataset", {}))
        eval_cfg = dict(cfg.get("eval", {}))

        self.model_cfg = model_cfg
        self.dataset_cfg = dataset_cfg
        self.threshold = float(eval_cfg.get("threshold", 0.5))
        self.in_channels = int(
            model_cfg.get("in_channels", dataset_cfg.get("in_channels", 3))
        )
        self.mean = self._stats(dataset_cfg.get("mean"), self.in_channels, 0.0)
        self.std = self._stats(dataset_cfg.get("std"), self.in_channels, 1.0)
        self.stride = int(max(model_cfg.get("out_strides", [4, 8, 16, 32])))

        self.model = SiameseTemporalCD(**model_cfg) if model_cfg else SiameseTemporalCD()
        state = self.checkpoint.get("model", self.checkpoint)
        if not isinstance(state, dict):
            raise ValueError("Invalid SiameseTemporalCD checkpoint: missing model state_dict.")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def _stats(values: Any, channels: int, fallback: float) -> list[float]:
        if values is None:
            return [fallback] * channels
        values = [float(v) for v in values]
        if len(values) == channels:
            return values
        if len(values) == 1:
            return values * channels
        if len(values) > channels:
            return values[:channels]
        return (values + [values[-1]] * channels)[:channels]

    @staticmethod
    def _read_image(path: str) -> np.ndarray:
        from preprocessing.loader import load_image

        image, _ = load_image(path)
        image = np.asarray(image)

        if image.ndim == 2:
            image = image[..., None]
        elif image.ndim == 3 and image.shape[0] < image.shape[-1] and image.shape[0] <= 32:
            # Rasterio returns CHW; standard images are HWC.
            image = np.transpose(image, (1, 2, 0))

        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] < 3:
            raise ValueError(f"Optical image must have at least 3 channels: {path}")

        # The trained optical model expects the first N optical channels.
        image = image[..., :3] if image.shape[-1] >= 3 else image
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return image

    @staticmethod
    def _resize(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        h, w = hw
        if image.shape[:2] == (h, w):
            return image
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        # IMPORTANT: training used raw 0-255 optical values with ImageNet
        # mean/std. Do not divide by 255 before applying these statistics.
        if image.shape[-1] != self.in_channels:
            if self.in_channels == 3 and image.shape[-1] >= 3:
                image = image[..., :3]
            elif image.shape[-1] == 1:
                image = np.repeat(image, self.in_channels, axis=-1)
            else:
                raise ValueError(
                    f"Checkpoint expects {self.in_channels} channels, input has {image.shape[-1]}."
                )

        mean = np.asarray(self.mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(self.std, dtype=np.float32).reshape(1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        normalized = (image - mean) / std
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1)))
        return tensor.unsqueeze(0).float()

    @staticmethod
    def _pad_to_stride(x: torch.Tensor, stride: int) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        ph = (stride - h % stride) % stride
        pw = (stride - w % stride) % stride
        if ph or pw:
            mode = "reflect" if h > 1 and w > 1 else "replicate"
            x = F.pad(x, (0, pw, 0, ph), mode=mode)
        return x, (h, w)

    @staticmethod
    def _rgb_preview(image: np.ndarray) -> np.ndarray:
        image = image[..., :3].astype(np.float32)
        if image.max() > 1.0:
            # Uploaded optical imagery is normally 8-bit. For other ranges,
            # robustly stretch only for visualization, never for model input.
            lo = np.nanpercentile(image, 2, axis=(0, 1), keepdims=True)
            hi = np.nanpercentile(image, 98, axis=(0, 1), keepdims=True)
            image = (image - lo) / np.maximum(hi - lo, 1e-6)
        return np.clip(image, 0.0, 1.0)

    @staticmethod
    def _save_heatmap(prob: np.ndarray, path: Path) -> None:
        normalized = np.clip(prob, 0.0, 1.0)
        heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        Image.fromarray(heat).save(path)

    @staticmethod
    def _save_overlay(base: np.ndarray, prob: np.ndarray, mask: np.ndarray, path: Path) -> None:
        rgb = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        # Probability visualization remains useful even when the hard mask is
        # empty. Strong predictions are blended more strongly.
        heat_bgr = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
        alpha = np.clip(prob[..., None] * 0.75, 0.0, 0.75)
        overlay = rgb.astype(np.float32) * (1.0 - alpha) + heat.astype(np.float32) * alpha

        # Hard predictions get a crisp boundary on top of the probability map.
        if np.any(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(path)

    @staticmethod
    def _region_overlay(base: np.ndarray, mask: np.ndarray, path: Path) -> list[dict[str, int]]:
        canvas = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
        regions = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(20, int(mask.size * 0.00002)):
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            regions.append({"label": label, "x": x, "y": y, "w": w, "h": h, "area": area})
        regions.sort(key=lambda r: r["area"], reverse=True)
        regions = regions[:20]

        for index, region in enumerate(regions, 1):
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]
            region_mask = labels == region["label"]
            canvas[region_mask] = (
                canvas[region_mask].astype(np.float32) * 0.45
                + np.array([255, 60, 60], dtype=np.float32) * 0.55
            ).astype(np.uint8)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(canvas, f"R{index}: change", (x + 4, max(20, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        Image.fromarray(canvas).save(path)
        return regions

    def predict(self, image1: str, image2: str) -> dict[str, Any]:
        t1_raw = self._read_image(image1)
        t2_raw = self._read_image(image2)

        # Use T1 as the reference grid, matching the model's bi-temporal setup.
        target_hw = t1_raw.shape[:2]
        t2_raw = self._resize(t2_raw, target_hw)

        t1 = self._preprocess(t1_raw)
        t2 = self._preprocess(t2_raw)
        t1_pad, original_hw = self._pad_to_stride(t1.to(self.device), self.stride)
        t2_pad, _ = self._pad_to_stride(t2.to(self.device), self.stride)

        with torch.inference_mode():
            logits = self.model(t1_pad, t2_pad)
            if logits.shape[-2:] != t1_pad.shape[-2:]:
                logits = F.interpolate(logits, size=t1_pad.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)[0, 0, :original_hw[0], :original_hw[1]].detach().cpu().numpy()

        # Keep the trained threshold for the actual binary prediction.
        pred = (prob >= self.threshold).astype(np.uint8)

        run_dir = OUTPUT_ROOT / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        t1_rgb = self._rgb_preview(t1_raw)
        t2_rgb = self._rgb_preview(t2_raw)
        if t1_rgb.shape[:2] != pred.shape:
            t1_rgb = self._resize(t1_rgb, pred.shape)
            t2_rgb = self._resize(t2_rgb, pred.shape)

        Image.fromarray((t1_rgb * 255).astype(np.uint8)).save(run_dir / "image_t1.png")
        Image.fromarray((t2_rgb * 255).astype(np.uint8)).save(run_dir / "image_t2.png")
        Image.fromarray((pred * 255).astype(np.uint8)).save(run_dir / "prediction_mask.png")
        self._save_heatmap(prob, run_dir / "probability_map.png")
        self._save_overlay(t2_rgb, prob, pred, run_dir / "change_map.png")
        regions = self._region_overlay(t2_rgb, pred, run_dir / "changed_regions.png")
        np.save(run_dir / "probability.npy", prob.astype(np.float32))

        total_pixels = int(pred.size)
        changed_pixels = int(pred.sum())
        change_percentage = round((changed_pixels / max(total_pixels, 1)) * 100.0, 2)

        # A balanced pixel-confidence diagnostic, explicitly not accuracy.
        pixel_confidence = np.where(pred > 0, prob, 1.0 - prob)
        confidence = float(np.mean(pixel_confidence))

        return {
            "success": True,
            "task": "change_detection",
            "model_name": "SiameseTemporalCD",
            "checkpoint": str(self.checkpoint_path),
            "change_map": str(run_dir / "change_map.png"),
            "probability_map": str(run_dir / "probability_map.png"),
            "prediction_mask": str(run_dir / "prediction_mask.png"),
            "changed_regions": str(run_dir / "changed_regions.png"),
            "image_t1": str(run_dir / "image_t1.png"),
            "image_t2": str(run_dir / "image_t2.png"),
            "regions": regions,
            "image_size": {"width": int(pred.shape[1]), "height": int(pred.shape[0])},
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "change_percentage": change_percentage,
            "threshold": self.threshold,
            "mean_change_probability": round(float(prob.mean()), 4),
            "max_change_probability": round(float(prob.max()), 4),
            "confidence": round(confidence, 4),
            "confidence_type": "pixel_prediction_diagnostic",
        }


_model_instance: SiameseTemporalCDInference | None = None


def get_temporal_change_detection_model() -> SiameseTemporalCDInference:
    global _model_instance
    if _model_instance is None:
        checkpoint = os.environ.get("AERIS_CHANGE_DETECTION_CHECKPOINT")
        _model_instance = SiameseTemporalCDInference(checkpoint_path=checkpoint)
    return _model_instance
