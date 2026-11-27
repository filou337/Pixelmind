import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from pixelmind.data.download import md5sum
from pixelmind.data.eda import render_markdown, summarize
from pixelmind.data.manifest import build_manifest, read_manifest, write_manifest
from pixelmind.data.quality import (
    Issue,
    Status,
    binarize_mask,
    check_duplicates,
    check_pair,
    overall_status,
)

H = W = 8


def rgb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


def mask(changed: int = 4) -> np.ndarray:
    m = np.zeros((H, W), dtype=np.uint8)
    m.flat[:changed] = 255
    return m


def codes(issues: list[Issue]) -> set[str]:
    return {i.code for i in issues}


# --- per-pair rules ---------------------------------------------------------


def test_valid_pair_passes():
    assert check_pair(rgb(1), rgb(2), mask()) == []


def test_size_mismatch_fails():
    issues = check_pair(rgb(1), rgb(2)[:4], mask())
    assert "size_mismatch" in codes(issues)
    assert overall_status(issues) is Status.FAIL


def test_mask_with_many_stray_values_fails():
    m = mask()
    m[0, :] = 128  # 8/64 pixels
    assert "mask_not_binary" in codes(check_pair(rgb(1), rgb(2), m))


def test_mask_with_few_stray_values_warns():
    m = np.zeros((100, 100), dtype=np.uint8)
    m[:10] = 255
    m[50, 50] = 254  # 1/10000 pixels, like LEVIR-CD artefacts
    t1 = np.random.default_rng(1).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    t2 = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    issues = check_pair(t1, t2, m)
    assert codes(issues) == {"mask_near_binary"}
    assert overall_status(issues) is Status.WARN


def test_binarize_mask():
    assert binarize_mask(np.array([0, 1])).tolist() == [False, True]
    assert binarize_mask(np.array([0, 127, 128, 254, 255])).tolist() == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_grayscale_image_fails():
    assert "t1_not_rgb" in codes(check_pair(rgb(1)[..., 0], rgb(2), mask()))


def test_multichannel_identical_mask_is_warning_only():
    m3 = np.stack([mask()] * 3, axis=-1)
    issues = check_pair(rgb(1), rgb(2), m3)
    assert codes(issues) == {"mask_multichannel"}
    assert overall_status(issues) is Status.WARN


def test_identical_t1_t2_warns():
    assert "t1_equals_t2" in codes(check_pair(rgb(1), rgb(1), mask()))


# --- dataset-level rules ----------------------------------------------------


def rec(pair_id, split, t1, t2):
    return {"pair_id": pair_id, "split": split, "sha256": {"t1": t1, "t2": t2}}


def test_duplicate_across_splits_is_leakage_fail():
    out = check_duplicates([rec("a", "train", "x", "y"), rec("b", "test", "x", "z")])
    assert {i.code for i in out["a"]} == {"duplicate_across_splits"}
    assert overall_status(out["b"]) is Status.FAIL


def test_duplicate_within_split_warns():
    out = check_duplicates([rec("a", "train", "x", "y"), rec("b", "train", "x", "z")])
    assert overall_status(out["a"]) is Status.WARN


# --- manifest end-to-end on a tiny fake dataset -----------------------------


def write_pair(root, split, scene, t1, t2, m):
    for sub, arr in (("A", t1), ("B", t2), ("label", m)):
        d = root / "levircd" / split / sub
        d.mkdir(parents=True, exist_ok=True)
        if arr is not None:
            Image.fromarray(arr).save(d / scene)


@pytest.fixture
def fake_root(tmp_path):
    write_pair(tmp_path, "train", "train_1.png", rgb(1), rgb(2), mask(16))
    write_pair(tmp_path, "train", "train_2.png", rgb(3), rgb(4), mask(0))
    write_pair(tmp_path, "test", "test_1.png", rgb(1), rgb(5), mask(8))  # T1 leaks from train
    write_pair(tmp_path, "val", "val_1.png", rgb(6), rgb(7), None)  # missing mask
    return tmp_path


def test_manifest_records_status_and_stats(fake_root, tmp_path):
    records = {r["pair_id"]: r for r in build_manifest(fake_root)}
    assert set(records) == {
        "levircd_train_train_1",
        "levircd_train_train_2",
        "levircd_test_test_1",
        "levircd_val_val_1",
    }

    ok = records["levircd_train_train_2"]
    assert ok["quality_status"] == "PASS"
    assert ok["changed_pixel_ratio"] == 0.0
    assert ok["paths"]["t1"] == "levircd/train/A/train_2.png"
    assert (ok["width"], ok["height"]) == (W, H)

    assert records["levircd_train_train_1"]["changed_pixel_ratio"] == 16 / 64
    assert records["levircd_test_test_1"]["quality_status"] == "FAIL"
    assert records["levircd_val_val_1"]["quality_status"] == "FAIL"
    assert "mask_missing" in {i["code"] for i in records["levircd_val_val_1"]["quality_issues"]}

    path = tmp_path / "m.jsonl"
    write_manifest(list(records.values()), path)
    assert read_manifest(path) == json.loads(json.dumps(list(records.values())))


def test_eda_summary_and_report(fake_root):
    summary = summarize(build_manifest(fake_root))
    train = summary["by_split"]["train"]
    assert train["pairs"] == 2
    assert train["no_change_pairs"] == 1
    assert train["positive_pixel_share"] == pytest.approx(16 / 128)
    md = render_markdown(summary, None)
    assert "duplicate_across_splits" in md and "| train | 2 |" in md


def test_md5sum_matches_hashlib(tmp_path):
    p = tmp_path / "f"
    data = b"pixelmind" * 200_000  # > 1 chunk
    p.write_bytes(data)
    assert md5sum(p) == hashlib.md5(data).hexdigest()


def test_download_extracts_each_split_into_its_own_folder(tmp_path, monkeypatch):
    import zipfile

    from pixelmind.data import download

    def fake_fetch(url, dest, md5):
        split = dest.stem
        with zipfile.ZipFile(dest, "w") as zf:  # same layout as the real archives
            for sub in ("A", "B", "label"):
                zf.writestr(f"{sub}/{split}_1.png", b"x")

    monkeypatch.setattr(download, "fetch", fake_fetch)
    download.download_levircd(tmp_path, ["train", "test"])
    assert (tmp_path / "levircd/train/A/train_1.png").exists()
    assert (tmp_path / "levircd/test/label/test_1.png").exists()
    assert not (tmp_path / "levircd/A").exists()
