"""Train a change-detection network on LEVIR-CD, tracked with MLflow.

python -m pixelmind.models.dl.train --config configs/siamese_unet.yaml [--set epochs=2 ...]

- Model selection on val F1 (threshold tuned on val every epoch), early stopping.
- Best checkpoint -> {out_dir}/{run_name}_best.pt (weights + config + val metrics).
- MLflow: params, per-epoch metrics, git commit, manifest sha256, checkpoint.
"""

import argparse
import hashlib
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pixelmind.data.loader import load_pair, usable
from pixelmind.data.manifest import read_manifest
from pixelmind.evaluation.metrics import ScoreHistogram
from pixelmind.models.dl.dataset import CropDataset
from pixelmind.models.dl.inference import make_scorer, pick_device
from pixelmind.models.dl.networks import BCEDiceLoss, build_model
from pixelmind.provenance import git_commit


@dataclass
class TrainConfig:
    model: str = "siamese_unet"  # unet | siamese_unet
    fusion: str = "diff"  # siamese only: diff | concat
    base_width: int = 32
    crop: int = 256
    crops_per_image: int = 8
    batch_size: int = 16
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dice_weight: float = 1.0
    patience: int = 15  # epochs without val F1 improvement
    num_workers: int = 4
    cache: bool = False  # keep decoded images in RAM
    tile: int = 512  # validation inference tile
    seed: int = 42
    device: str = "auto"
    max_train_images: int | None = None  # smoke tests only
    max_val_images: int | None = None
    root: str = "data/raw"
    manifest: str = "data/manifests/levircd.jsonl"
    out_dir: str = "artifacts/models"
    run_name: str | None = None
    mlflow_experiment: str = "pixelmind-levircd"

    @classmethod
    def load(cls, path: Path | None, overrides: list[str]) -> "TrainConfig":
        data = yaml.safe_load(path.read_text()) if path else {}
        for item in overrides:  # key=value, parsed as YAML scalars
            key, _, value = item.partition("=")
            data[key] = yaml.safe_load(value)
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate(model, records, root: Path, device, tile: int) -> dict:
    model.eval()
    scorer = make_scorer(model, device, tile=tile)
    hist = ScoreHistogram()
    for r in records:
        t1, t2, mask = load_pair(root, r)
        hist.add(scorer(t1, t2), mask)
    t = hist.best_threshold()
    return {**hist.metrics_at(t), "threshold": t, "ap": hist.average_precision()}


def train(cfg: TrainConfig) -> Path:
    import mlflow

    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    root, out_dir = Path(cfg.root), Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg.run_name or (cfg.model if cfg.model == "unet" else f"{cfg.model}_{cfg.fusion}")
    manifest_path = Path(cfg.manifest)
    records = read_manifest(manifest_path)
    train_recs = usable(records, "train")[: cfg.max_train_images]
    val_recs = usable(records, "val")[: cfg.max_val_images]

    ds = CropDataset(root, train_recs, cfg.crop, cfg.crops_per_image, True, cfg.cache, cfg.seed)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda", drop_last=len(ds) > cfg.batch_size,
        persistent_workers=cfg.num_workers > 0,
        generator=torch.Generator().manual_seed(cfg.seed),
    )  # fmt: skip
    model = build_model(cfg.model, cfg.base_width, cfg.fusion).to(device)
    criterion = BCEDiceLoss(cfg.dice_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # local SQLite store (MLflow >= 3 no longer accepts the ./mlruns file store)
    default_uri = f"sqlite:///{(out_dir / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", default_uri))
    if mlflow.get_experiment_by_name(cfg.mlflow_experiment) is None:
        # keep run artifacts next to the database: one folder to copy back from Kaggle
        location = (out_dir / "mlruns").resolve().as_uri()
        mlflow.create_experiment(cfg.mlflow_experiment, artifact_location=location)
    mlflow.set_experiment(cfg.mlflow_experiment)
    best_path = out_dir / f"{run_name}_best.pt"
    best_f1, stale = -1.0, 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(asdict(cfg))
        mlflow.set_tags({
            "git_commit": git_commit(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "device": str(device) + (f" ({torch.cuda.get_device_name()})" if use_amp else ""),
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_pairs": len(train_recs), "val_pairs": len(val_recs),
        })  # fmt: skip

        for epoch in range(cfg.epochs):
            t0 = time.perf_counter()
            ds.set_epoch(epoch)
            model.train()
            total, n = 0.0, 0
            for t1, t2, mask in loader:
                t1, t2, mask = (x.to(device, non_blocking=True) for x in (t1, t2, mask))
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=use_amp):
                    loss = criterion(model(t1, t2).float(), mask)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item() * len(t1)
                n += len(t1)
            sched.step()

            val = validate(model, val_recs, root, device, cfg.tile)
            metrics = {
                "train_loss": total / n, "lr": sched.get_last_lr()[0],
                **{f"val_{k}": val[k] for k in ("f1", "iou", "precision", "recall", "ap")},
                "val_threshold": val["threshold"], "epoch_seconds": time.perf_counter() - t0,
            }  # fmt: skip
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None}, step=epoch)
            f1 = val["f1"] or 0.0
            print(
                f"epoch {epoch:3d} loss {metrics['train_loss']:.4f} val F1 {f1:.4f} "
                f"IoU {val['iou'] or 0:.4f} AP {val['ap']:.4f} ({metrics['epoch_seconds']:.0f}s)",
                flush=True,
            )
            if f1 > best_f1:
                best_f1, stale = f1, 0
                torch.save(
                    {"state_dict": model.state_dict(), "config": asdict(cfg), "epoch": epoch,
                     "val": val, "git_commit": git_commit()},
                    best_path,
                )  # fmt: skip
            else:
                stale += 1
                if stale >= cfg.patience:
                    print(f"early stopping at epoch {epoch}", flush=True)
                    break

        mlflow.log_metric("best_val_f1", best_f1)
        mlflow.log_artifact(str(best_path))
    print(f"best val F1 {best_f1:.4f} -> {best_path}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    train(TrainConfig.load(args.config, args.set))


if __name__ == "__main__":
    main()
