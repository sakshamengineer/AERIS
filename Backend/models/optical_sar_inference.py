"""Production inference adapter for the trained OpticalSARChangeNet."""
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
import rasterio

from models.optical_sar_multimodal import OpticalSARChangeNet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "runs" / "optical_sar" / "best.pt"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "optical_sar"


class OpticalSARChangeNetInference:
    """Run the trained four-input OpticalSARChangeNet.

    Expected inputs, in order:
      optical_t1, optical_t2, sar_t1, sar_t2

    The checkpoint's embedded model/dataset/eval configuration is used so
    inference matches training instead of duplicating hyperparameters in code.
    """

    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str] | None = None,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"OpticalSARChangeNet checkpoint not found: {self.checkpoint_path}"
            )

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        checkpoint = torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid OpticalSARChangeNet checkpoint format.")

        self.checkpoint = checkpoint
        self.config = checkpoint.get("config", {}) or {}
        self.model_cfg = dict(self.config.get("model", {}) or {})
        self.dataset_cfg = dict(self.config.get("dataset", {}) or {})
        self.eval_cfg = dict(self.config.get("eval", {}) or {})

        self.optical_channels = int(self.model_cfg.get("optical_channels", self.dataset_cfg.get("optical_channels", 3)))
        self.sar_channels = int(self.model_cfg.get("sar_channels", self.dataset_cfg.get("sar_channels", 2)))
        self.threshold = float(self.eval_cfg.get("threshold", 0.5))
        self.stride = int(max(self.model_cfg.get("out_strides", [8, 16, 32])))

        self.optical_mean = self._stats(self.dataset_cfg.get("optical_mean"), self.optical_channels, 0.0)
        self.optical_std = self._stats(self.dataset_cfg.get("optical_std"), self.optical_channels, 1.0)
        self.sar_mean = self._stats(self.dataset_cfg.get("sar_mean"), self.sar_channels, 0.0)
        self.sar_std = self._stats(self.dataset_cfg.get("sar_std"), self.sar_channels, 1.0)
        self.sar_clip = self.dataset_cfg.get("sar_clip")
        self.sar_log_db = bool(self.dataset_cfg.get("sar_log_db", False))

        self.model = OpticalSARChangeNet(**self.model_cfg) if self.model_cfg else OpticalSARChangeNet()
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError("OpticalSARChangeNet checkpoint is missing 'model' state_dict.")
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
    def _read_raw(path: str) -> np.ndarray:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if p.suffix.lower() in {".tif", ".tiff"}:
            with rasterio.open(p) as src:
                arr = src.read().astype(np.float32)  # CHW
            if arr.ndim == 2:
                arr = arr[None, ...]
            return np.transpose(arr, (1, 2, 0))

        with Image.open(p) as im:
            # Preserve grayscale SAR instead of forcing everything to RGB.
            arr = np.asarray(im)
            if arr.ndim == 2:
                arr = arr[..., None]
            elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
                arr = arr[..., :3]
            else:
                raise ValueError(f"Unsupported image channel layout: {path}")
            return arr.astype(np.float32)

    @staticmethod
    def _resize(image: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        h, w = hw
        if image.shape[:2] == (h, w):
            return image
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _adapt_channels(image: np.ndarray, channels: int, name: str) -> np.ndarray:
        c = image.shape[-1]
        if c == channels:
            return image
        if c > channels:
            return image[..., :channels]
        if c == 1:
            return np.repeat(image, channels, axis=-1)
        raise ValueError(f"{name} has {c} channels but checkpoint expects {channels}.")

    @staticmethod
    def _normalize(image: np.ndarray, mean: list[float], std: list[float], clip: list[float] | None = None, log_db: bool = False) -> np.ndarray:
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if log_db:
            image = 10.0 * np.log10(np.maximum(image, 1e-10))
        if clip is not None and len(clip) == 2:
            image = np.clip(image, float(clip[0]), float(clip[1]))
        mean_arr = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
        std_arr = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
        std_arr = np.where(std_arr == 0, 1.0, std_arr)
        return (image - mean_arr) / std_arr

    def _prep_optical(self, image: np.ndarray) -> torch.Tensor:
        image = self._adapt_channels(image, self.optical_channels, "Optical image")
        image = self._normalize(image, self.optical_mean, self.optical_std)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()

    def _prep_sar(self, image: np.ndarray) -> torch.Tensor:
        image = self._adapt_channels(image, self.sar_channels, "SAR image")
        image = self._normalize(image, self.sar_mean, self.sar_std, self.sar_clip, self.sar_log_db)
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).float()

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
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        image = image[..., :3].astype(np.float32)
        lo = np.nanpercentile(image, 2, axis=(0, 1), keepdims=True)
        hi = np.nanpercentile(image, 98, axis=(0, 1), keepdims=True)
        return np.clip((image - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0)

    @staticmethod
    def _sar_preview(image: np.ndarray) -> np.ndarray:
        if image.shape[-1] == 1:
            gray = image[..., 0]
        else:
            gray = np.mean(image[..., :2], axis=-1)
        lo, hi = np.nanpercentile(gray, 2), np.nanpercentile(gray, 98)
        gray = np.clip((gray - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        return np.repeat(gray[..., None], 3, axis=-1)

    @staticmethod
    def _save_heatmap(prob: np.ndarray, path: Path) -> None:
        heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        Image.fromarray(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)).save(path)

    @staticmethod
    def _save_overlay(base: np.ndarray, prob: np.ndarray, mask: np.ndarray, path: Path) -> None:
        rgb = (np.clip(base, 0, 1) * 255).astype(np.uint8)
        heat = cv2.applyColorMap((np.clip(prob, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        alpha = np.clip(prob[..., None] * 0.9, 0.0, 0.9)
        out = rgb.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha
        if np.any(mask):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(path)

    def predict(self, optical_t1: str, optical_t2: str, sar_t1: str, sar_t2: str) -> dict[str, Any]:
        raw_opt1 = self._read_raw(optical_t1)
        raw_opt2 = self._read_raw(optical_t2)
        raw_sar1 = self._read_raw(sar_t1)
        raw_sar2 = self._read_raw(sar_t2)

        target_hw = raw_opt1.shape[:2]
        raw_opt2 = self._resize(raw_opt2, target_hw)
        raw_sar1 = self._resize(raw_sar1, target_hw)
        raw_sar2 = self._resize(raw_sar2, target_hw)

        tensors = [
            self._prep_optical(raw_opt1),
            self._prep_optical(raw_opt2),
            self._prep_sar(raw_sar1),
            self._prep_sar(raw_sar2),
        ]
        padded = [self._pad_to_stride(x.to(self.device), self.stride)[0] for x in tensors]

        with torch.inference_mode():
            logits = self.model(*padded)
            if logits.shape[-2:] != padded[0].shape[-2:]:
                logits = F.interpolate(logits, size=padded[0].shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)[0, 0, :target_hw[0], :target_hw[1]].cpu().numpy()

        pred = (prob >= self.threshold).astype(np.uint8)
        run_dir = OUTPUT_ROOT / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=True)

        opt1_preview = self._rgb_preview(raw_opt1)
        opt2_preview = self._rgb_preview(raw_opt2)
        sar1_preview = self._sar_preview(raw_sar1)
        sar2_preview = self._sar_preview(raw_sar2)

        Image.fromarray((opt1_preview * 255).astype(np.uint8)).save(run_dir / "optical_t1.png")
        Image.fromarray((opt2_preview * 255).astype(np.uint8)).save(run_dir / "optical_t2.png")
        Image.fromarray((sar1_preview * 255).astype(np.uint8)).save(run_dir / "sar_t1.png")
        Image.fromarray((sar2_preview * 255).astype(np.uint8)).save(run_dir / "sar_t2.png")
        Image.fromarray((pred * 255).astype(np.uint8)).save(run_dir / "prediction_mask.png")
        self._save_heatmap(prob, run_dir / "probability_map.png")
        self._save_overlay(opt2_preview, prob, pred, run_dir / "change_map.png")

        # A simple multimodal visualization: optical T2 luminance blended with
        # SAR T2 intensity. It is evidence only, not another model prediction.
        fusion = np.clip(0.65 * opt2_preview + 0.35 * sar2_preview, 0, 1)
        Image.fromarray((fusion * 255).astype(np.uint8)).save(run_dir / "sar_fusion_map.png")
        np.save(run_dir / "probability.npy", prob.astype(np.float32))

        total = int(pred.size)
        changed = int(pred.sum())
        change_percentage = round(100.0 * changed / max(total, 1), 2)
        mean_probability = float(prob.mean())
        max_probability = float(prob.max())
        # Confidence is the average confidence in the binary decision. This is
        # a model-derived diagnostic, not a calibrated accuracy probability.
        confidence = float(np.mean(np.where(pred > 0, prob, 1.0 - prob)))

        return {
            "success": True,
            "task": "optical_sar",
            "model": "OpticalSARChangeNet",
            "checkpoint": str(self.checkpoint_path),
            "threshold": self.threshold,
            "image_size": {"width": int(target_hw[1]), "height": int(target_hw[0])},
            "change_percentage": change_percentage,
            "mean_change_probability": round(mean_probability * 100.0, 2),
            "max_change_probability": round(max_probability * 100.0, 2),
            "confidence": round(confidence, 4),
            "confidence_type": "model_prediction_diagnostic",
            "change_map": str(run_dir / "change_map.png"),
            "probability_map": str(run_dir / "probability_map.png"),
            "prediction_mask": str(run_dir / "prediction_mask.png"),
            "sar_fusion_map": str(run_dir / "sar_fusion_map.png"),
            "input_previews": {
                "optical_t1": str(run_dir / "optical_t1.png"),
                "optical_t2": str(run_dir / "optical_t2.png"),
                "sar_t1": str(run_dir / "sar_t1.png"),
                "sar_t2": str(run_dir / "sar_t2.png"),
            },
        }


_optical_sar_model = None


def get_optical_sar_model() -> OpticalSARChangeNetInference:
    global _optical_sar_model
    if _optical_sar_model is None:
        _optical_sar_model = OpticalSARChangeNetInference()
    return _optical_sar_model
