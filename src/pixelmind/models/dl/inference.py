"""Checkpoint loading and tiled inference, exposed as a protocol scorer."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from pixelmind.models.dl.dataset import to_tensor
from pixelmind.models.dl.networks import build_model


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], cfg["base_width"], cfg.get("fusion", "diff"))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


@torch.inference_mode()
def predict_proba(
    model: nn.Module, t1: np.ndarray, t2: np.ndarray, device: torch.device,
    tile: int = 512, batch: int = 4,
) -> np.ndarray:  # fmt: skip
    """Probability map HxW, computed tile by tile (bounded memory)."""
    h, w = t1.shape[:2]
    ph, pw = -h % tile, -w % tile  # pad to a multiple of the tile size
    a = torch.nn.functional.pad(to_tensor(t1), (0, pw, 0, ph), mode="reflect")
    b = torch.nn.functional.pad(to_tensor(t2), (0, pw, 0, ph), mode="reflect")
    H, W = a.shape[1:]
    coords = [(y, x) for y in range(0, H, tile) for x in range(0, W, tile)]
    out = torch.empty(H, W)
    for i in range(0, len(coords), batch):
        cs = coords[i : i + batch]
        xa = torch.stack([a[:, y : y + tile, x : x + tile] for y, x in cs]).to(device)
        xb = torch.stack([b[:, y : y + tile, x : x + tile] for y, x in cs]).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(xa, xb).float()).squeeze(1).cpu()
        for (y, x), pt in zip(cs, p, strict=True):
            out[y : y + tile, x : x + tile] = pt
    return out[:h, :w].numpy()


def make_scorer(model: nn.Module, device: torch.device, tile: int = 512, batch: int = 4):
    def score(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        p = predict_proba(model, t1, t2, device, tile, batch)
        return (p * 255).round().astype(np.uint8)

    return score
