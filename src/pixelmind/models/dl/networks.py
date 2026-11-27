"""U-Net (early fusion) and Siamese U-Net (shared encoder) for change detection.

Both expose forward(t1, t2) -> logits (B, 1, H, W); H and W must be multiples of 16.
"""

import torch
from torch import nn


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    def __init__(self, c_in: int, widths: list[int]) -> None:
        super().__init__()
        chans = [c_in, *widths]
        self.blocks = nn.ModuleList(
            conv_block(a, b) for a, b in zip(chans, chans[1:], strict=False)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for i, block in enumerate(self.blocks):
            x = block(x if i == 0 else self.pool(x))
            feats.append(x)
        return feats  # high -> low resolution


class Decoder(nn.Module):
    def __init__(self, skip_channels: list[int], widths: list[int]) -> None:
        super().__init__()
        # skip_channels / widths are given high -> low resolution
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        c = skip_channels[-1]
        for skip, w in zip(reversed(skip_channels[:-1]), reversed(widths[:-1]), strict=True):
            self.ups.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.blocks.append(conv_block(w + skip, w))
            c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = feats[-1]
        for up, block, skip in zip(self.ups, self.blocks, reversed(feats[:-1]), strict=True):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.head(x)


def _widths(base: int) -> list[int]:
    return [base, base * 2, base * 4, base * 8, base * 16]


class UNet(nn.Module):
    """Baseline DL 1: T1 and T2 concatenated as a 6-channel input."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        w = _widths(base)
        self.encoder = Encoder(6, w)
        self.decoder = Decoder(w, w)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(torch.cat([t1, t2], dim=1)))


class SiameseUNet(nn.Module):
    """DL 2: one encoder with shared weights for T1 and T2, fused at every scale.

    fusion="diff": |f1 - f2| (symmetric, explicit comparison)
    fusion="concat": [f1, f2] (the decoder learns the comparison)
    """

    def __init__(self, base: int = 32, fusion: str = "diff") -> None:
        super().__init__()
        if fusion not in ("diff", "concat"):
            raise ValueError(f"unknown fusion {fusion!r}")
        w = _widths(base)
        self.fusion = fusion
        self.encoder = Encoder(3, w)
        skip = w if fusion == "diff" else [2 * c for c in w]
        self.decoder = Decoder(skip, w)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        f1, f2 = self.encoder(t1), self.encoder(t2)
        if self.fusion == "diff":
            fused = [torch.abs(a - b) for a, b in zip(f1, f2, strict=True)]
        else:
            fused = [torch.cat([a, b], dim=1) for a, b in zip(f1, f2, strict=True)]
        return self.decoder(fused)


def build_model(name: str, base: int = 32, fusion: str = "diff") -> nn.Module:
    if name == "unet":
        return UNet(base)
    if name == "siamese_unet":
        return SiameseUNet(base, fusion)
    raise ValueError(f"unknown model {name!r}")


class BCEDiceLoss(nn.Module):
    """BCEWithLogits + soft Dice: Dice counters the ~5 % positive-pixel imbalance."""

    def __init__(self, dice_weight: float = 1.0, eps: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight, self.eps = dice_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        dice = 1 - (2 * inter + self.eps) / (p.sum() + target.sum() + self.eps)
        return self.bce(logits, target) + self.dice_weight * dice
