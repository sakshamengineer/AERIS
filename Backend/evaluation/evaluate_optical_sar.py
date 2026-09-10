"""Evaluate the current OpticalSARChangeNet checkpoint.

The old evaluator measured an untrained OpticalSARFusionNetwork. That model is
no longer part of AERIS. This evaluator targets runs/optical_sar/best.pt and
uses the configured OpticalSARDataset validation split when available.
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from models.optical_sar_multimodal import OpticalSARChangeNet  # noqa: E402
from training.common import forward_batch, get_device  # noqa: E402
from training.metrics import ChangeMetrics  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

CONFIG_PATH = BACKEND_ROOT / "configs" / "optical_sar.yaml"
CHECKPOINT_PATH = BACKEND_ROOT / "runs" / "optical_sar" / "best.pt"


def _load_config():
    import yaml
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _history_result(ckpt, cfg):
    history = ckpt.get("history", []) if isinstance(ckpt, dict) else []
    if not history:
        return None
    select_metric = cfg.get("eval", {}).get("select_metric", "f1")
    row = max(history, key=lambda item: float(item.get(select_metric, -1)))
    metric_map = {
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "iou": row.get("iou"),
        "dice": row.get("dice"),
        "accuracy": row.get("accuracy"),
    }
    metric_map = {k: round(float(v), 4) for k, v in metric_map.items() if v is not None}
    return row, metric_map


def _evaluate_dataset(cfg, model, device):
    from dataset.optical_sar_dataset import OpticalSARDataset

    dcfg = cfg["dataset"]
    common = dict(
        root=dcfg["root"],
        dataset_type=dcfg.get("type", "slag"),
        optical_channels=int(dcfg.get("optical_channels", 3)),
        sar_channels=int(dcfg.get("sar_channels", 2)),
        optical_mean=dcfg.get("optical_mean"),
        optical_std=dcfg.get("optical_std"),
        sar_mean=dcfg.get("sar_mean"),
        sar_std=dcfg.get("sar_std"),
        sar_clip=dcfg.get("sar_clip"),
        sar_log_db=bool(dcfg.get("sar_log_db", False)),
        check_georef=bool(dcfg.get("check_georef", True)),
    )
    split = dcfg.get("val_split", "val")
    dataset = OpticalSARDataset(split=split, augment=None, **common)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("training", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    threshold = float(cfg.get("eval", {}).get("threshold", 0.5))
    metrics = ChangeMetrics(threshold=threshold)
    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            logits = forward_batch(model, batch)
            metrics.update(logits.float(), batch["mask"])

    result = metrics.compute()
    return {k: round(float(v), 4) for k, v in result.items()}, len(dataset), split


def run():
    cfg = _load_config()

    if not CHECKPOINT_PATH.exists():
        return {
            "model": "optical_sar_change_net",
            "display_name": "OpticalSARChangeNet",
            "model_type": "Bi-temporal optical + SAR change detection network",
            "status": "pending",
            "pending_reason": f"Checkpoint not found: {CHECKPOINT_PATH}",
            "sample_size": 0,
            "metrics": {},
        }

    device = get_device(cfg.get("device", "auto"))
    model = OpticalSARChangeNet(**cfg["model"]).to(device)
    ckpt = load_checkpoint(str(CHECKPOINT_PATH), model, map_location="cpu")

    try:
        dataset_metrics, sample_size, split = _evaluate_dataset(cfg, model, device)
        return {
            "model": "optical_sar_change_net",
            "display_name": "OpticalSARChangeNet",
            "model_type": "Bi-temporal optical + SAR change detection network",
            "status": "evaluated",
            "checkpoint": str(CHECKPOINT_PATH.relative_to(BACKEND_ROOT)),
            "evaluation_basis": "Live evaluation on the configured validation split.",
            "dataset_split": split,
            "sample_size": sample_size,
            "threshold": float(cfg.get("eval", {}).get("threshold", 0.5)),
            "metrics": dataset_metrics,
        }
    except Exception as error:
        recorded = _history_result(ckpt, cfg)
        if recorded:
            row, metric_map = recorded
            return {
                "model": "optical_sar_change_net",
                "display_name": "OpticalSARChangeNet",
                "model_type": "Bi-temporal optical + SAR change detection network",
                "status": "evaluated",
                "checkpoint": str(CHECKPOINT_PATH.relative_to(BACKEND_ROOT)),
                "evaluation_basis": "Metrics recorded by the best checkpoint during validation. Live re-evaluation was unavailable.",
                "best_epoch": row.get("epoch"),
                "sample_size": None,
                "threshold": float(cfg.get("eval", {}).get("threshold", 0.5)),
                "metrics": metric_map,
                "evaluation_note": f"Live validation could not be rerun: {error}",
            }

        return {
            "model": "optical_sar_change_net",
            "display_name": "OpticalSARChangeNet",
            "model_type": "Bi-temporal optical + SAR change detection network",
            "status": "error",
            "checkpoint": str(CHECKPOINT_PATH.relative_to(BACKEND_ROOT)),
            "error": str(error),
            "metrics": {},
        }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
