"""Download LEVIR-CD (official train/val/test split) with MD5 verification.

Mirror: Hugging Face, pinned commit (same source and checksums as TorchGeo).
Licence: academic, non-commercial use only. See https://justchenhao.github.io/LEVIR/
"""

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path

_BASE = (
    "https://huggingface.co/datasets/satellite-image-deep-learning/LEVIR-CD/resolve/"
    "6a6bb0a5b389403d81c05e33bf08bc0b9e5f13a6"
)
LEVIRCD_SPLITS = {
    "train": "a638e71f480628652dea78d8544307e4",
    "val": "f7b857978524f9aa8c3bf7f94e3047a4",
    "test": "07d5dd89e46f5c1359e2eca746989ed9",
}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path, md5: str) -> None:
    """Download url to dest unless a file with the right MD5 is already there."""
    if dest.exists() and md5sum(dest) == md5:
        print(f"ok (cached) {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    actual = md5sum(tmp)
    if actual != md5:
        tmp.unlink()
        raise RuntimeError(f"MD5 mismatch for {dest.name}: expected {md5}, got {actual}")
    tmp.rename(dest)
    print(f"ok {dest.name}")


def download_levircd(root: Path, splits: list[str]) -> None:
    """Download and extract to root/levircd/{split}/{A,B,label}/*.png (raw, read-only)."""
    target = root / "levircd"
    archives = root / "_archives"
    archives.mkdir(parents=True, exist_ok=True)
    for split in splits:
        archive = archives / f"{split}.zip"
        fetch(f"{_BASE}/{split}.zip", archive, LEVIRCD_SPLITS[split])
        if (target / split).exists():
            print(f"already extracted: {target / split}")
            continue
        # archives contain A/, B/, label/ at their root (no split folder)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target / split)
        print(f"extracted {split} -> {target / split}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw"))
    parser.add_argument("--splits", nargs="+", default=list(LEVIRCD_SPLITS), choices=LEVIRCD_SPLITS)
    args = parser.parse_args()
    download_levircd(args.root, args.splits)


if __name__ == "__main__":
    main()
