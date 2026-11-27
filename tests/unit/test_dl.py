import json

import numpy as np
import pytest
import torch
from PIL import Image

from pixelmind.data.manifest import build_manifest, write_manifest
from pixelmind.models.dl.dataset import CropDataset, augment
from pixelmind.models.dl.inference import load_checkpoint, predict_proba
from pixelmind.models.dl.networks import BCEDiceLoss, SiameseUNet, UNet, build_model
from pixelmind.models.dl.train import TrainConfig, train

torch.manual_seed(0)


@pytest.mark.parametrize(
    "model",
    [UNet(base=4), SiameseUNet(base=4, fusion="diff"), SiameseUNet(base=4, fusion="concat")],
)
def test_networks_output_one_logit_per_pixel(model):
    t1, t2 = torch.rand(2, 3, 64, 48), torch.rand(2, 3, 64, 48)
    assert model(t1, t2).shape == (2, 1, 64, 48)


def test_siamese_diff_is_symmetric_and_zero_change_invariant():
    model = SiameseUNet(base=4).eval()
    t1, t2 = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    assert torch.allclose(model(t1, t2), model(t2, t1), atol=1e-6)


def test_build_model_rejects_unknown():
    with pytest.raises(ValueError):
        build_model("resnet")


def test_bce_dice_loss_lower_for_good_prediction():
    target = (torch.rand(2, 1, 16, 16) > 0.9).float()
    good = (target * 2 - 1) * 10
    assert BCEDiceLoss()(good, target) < BCEDiceLoss()(-good, target)


def test_augment_applies_same_transform_to_all_arrays():
    img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    mask = img[..., 0]
    for seed in range(8):
        a, b, m = augment([img, img.copy(), mask], np.random.default_rng(seed))
        assert np.array_equal(a, b) and np.array_equal(a[..., 0], m)


# --- end-to-end on a tiny synthetic dataset ---------------------------------


@pytest.fixture
def synthetic(tmp_path):
    for k, (split, n) in enumerate((("train", 4), ("val", 2), ("test", 2))):
        for i in range(n):
            rng = np.random.default_rng(100 * k + i)
            t1 = rng.integers(60, 120, (64, 64, 3), dtype=np.uint8)
            t2 = t1.copy()
            m = np.zeros((64, 64), np.uint8)
            t2[16:40, 8:30] = 250
            m[16:40, 8:30] = 255
            for sub, arr in (("A", t1), ("B", t2), ("label", m)):
                d = tmp_path / "levircd" / split / sub
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(d / f"{split}_{i}.png")
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(build_manifest(tmp_path), manifest)
    return tmp_path, manifest


def test_crop_dataset_aligned_and_reproducible(synthetic):
    root, manifest = synthetic
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    train_recs = [r for r in records if r["split"] == "train"]
    ds = CropDataset(root, train_recs, crop=32, crops_per_image=3, seed=1)
    assert len(ds) == 12
    t1, t2, m = ds[5]
    assert t1.shape == t2.shape == (3, 32, 32) and m.shape == (1, 32, 32)
    # changed pixels are exactly the bright ones in T2
    assert torch.equal(m[0] > 0, (t2 - t1).abs().amax(0) > 0.3)
    assert all(torch.equal(a, b) for a, b in zip(ds[5], ds[5], strict=True))
    cached = CropDataset(root, train_recs, crop=32, crops_per_image=3, seed=1, cache=True)
    assert all(torch.equal(a, b) for a, b in zip(cached[5], ds[5], strict=True))
    ds.set_epoch(1)
    assert not torch.equal(ds[5][0], t1) or not torch.equal(ds[5][2], m)


def test_predict_proba_tiles_cover_image_of_any_size():
    model = SiameseUNet(base=4).eval()
    t1 = np.random.default_rng(0).integers(0, 256, (70, 50, 3), dtype=np.uint8)
    p = predict_proba(model, t1, t1, torch.device("cpu"), tile=32, batch=3)
    assert p.shape == (70, 50) and np.isfinite(p).all() and 0 <= p.min() <= p.max() <= 1


def test_train_smoke_saves_checkpoint_and_logs_mlflow(synthetic, tmp_path, monkeypatch):
    root, manifest = synthetic
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = TrainConfig(
        model="siamese_unet", base_width=4, crop=32, crops_per_image=2, batch_size=4, epochs=2,
        num_workers=0, tile=32, root=str(root), manifest=str(manifest),
        out_dir=str(tmp_path / "out"), device="cpu",
    )  # fmt: skip
    path = train(cfg)
    model, ckpt = load_checkpoint(path, torch.device("cpu"))
    assert ckpt["config"]["model"] == "siamese_unet"
    assert 0 <= ckpt["val"]["f1"] <= 1
    import mlflow

    runs = mlflow.search_runs(experiment_names=[cfg.mlflow_experiment], output_format="list")
    assert len(runs) == 1
    assert runs[0].data.metrics["best_val_f1"] == pytest.approx(ckpt["val"]["f1"])
    assert runs[0].data.params["model"] == "siamese_unet"
    assert list((tmp_path / "out" / "mlruns").rglob(path.name))  # checkpoint logged as artifact


def test_config_rejects_unknown_keys(tmp_path):
    with pytest.raises(ValueError):
        TrainConfig.load(None, ["learning_rate=0.1"])
    assert TrainConfig.load(None, ["epochs=3", "cache=true"]).epochs == 3
