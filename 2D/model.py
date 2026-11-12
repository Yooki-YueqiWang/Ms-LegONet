"""Network of the 2D Ms-LegONet block.

Input: coefficient patches kappa of shape (B, 16, 16) (float32). The network reads the
normalized log-coefficient (log kappa minus its patch mean, divided by log 1e4) and two
coordinate channels. Output: (B, 225, 63), i.e. 63 nodal fields on the 225 interior nodes of
the 16 x 16 Q1 patch (zero trace). The fields span the learned local space; they are
energy-orthonormalized downstream (`losses.orthonormalize`, `solve.metric_basis`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Q = 63
CELLS = 16
INTERIOR = 225

ARCHITECTURES = {
    "MEDIUM": {
        "encoder_widths": [48, 96, 192, 288],
        "fpn_width": 160,
        "global_width": 288,
        "decoder_widths": [384, 384, 192],
        "residual_blocks_per_level": 2,
        "bottleneck_blocks": 2,
    },
    "LARGE": {
        "encoder_widths": [64, 128, 256, 384],
        "fpn_width": 192,
        "global_width": 384,
        "decoder_widths": [512, 512, 256],
        "residual_blocks_per_level": 2,
        "bottleneck_blocks": 2,
    },
}


def sample_level(level: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling of a feature map at the interior nodes of the patch."""
    height, width = level.shape[-2:]
    x = 0.5 * (axis + 1) * (width - 1)
    y = 0.5 * (axis + 1) * (height - 1)
    xn = torch.arange(width, device=level.device, dtype=level.dtype)
    yn = torch.arange(height, device=level.device, dtype=level.dtype)
    xw = torch.clamp(1 - torch.abs(x[:, None] - xn[None]), min=0)
    yw = torch.clamp(1 - torch.abs(y[:, None] - yn[None]), min=0)
    sampled = torch.einsum("ph,bchw,qw->bpqc", yw, level, xw)
    return sampled.reshape(level.shape[0], len(axis) ** 2, level.shape[1])


def _groups(width: int) -> int:
    for value in (16, 8, 4, 2, 1):
        if width % value == 0:
            return value
    return 1


class PreActResidual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(_groups(width), width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.body(value)


class MultiscaleFPNTrialSpace(nn.Module):
    """Local block: coefficient patch -> q = 63 interior nodal fields spanning the learned space."""

    def __init__(self, variant: str = "LARGE", q: int = Q):
        super().__init__()
        if variant not in ARCHITECTURES:
            raise ValueError(f"unknown architecture variant {variant}")
        if q != Q:
            raise ValueError("the 2D block uses q=63")
        self.variant = variant
        self.q = q
        config = ARCHITECTURES[variant]
        c0, c1, c2, c3 = config["encoder_widths"]
        fpn = config["fpn_width"]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, padding=1),
            nn.GroupNorm(_groups(c0), c0),
            nn.SiLU(),
        )
        self.e0 = nn.Sequential(PreActResidual(c0), PreActResidual(c0))
        self.down1 = nn.Conv2d(c0, c1, 3, stride=2, padding=1)
        self.e1 = nn.Sequential(PreActResidual(c1), PreActResidual(c1))
        self.down2 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.e2 = nn.Sequential(PreActResidual(c2), PreActResidual(c2))
        self.down3 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.e3 = nn.Sequential(PreActResidual(c3), PreActResidual(c3))
        self.bottleneck = nn.Sequential(PreActResidual(c3), PreActResidual(c3))

        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(c0, fpn, 1),
                nn.Conv2d(c1, fpn, 1),
                nn.Conv2d(c2, fpn, 1),
                nn.Conv2d(c3, fpn, 1),
            ]
        )
        self.fpn_fusion = nn.ModuleList([PreActResidual(fpn) for _ in range(4)])

        descriptor = fpn + config["global_width"] + 10
        h0, h1, h2 = config["decoder_widths"]
        self.decoder = nn.Sequential(
            nn.Linear(descriptor, h0),
            nn.SiLU(),
            nn.Linear(h0, h1),
            nn.SiLU(),
            nn.Linear(h1, h2),
            nn.SiLU(),
            nn.Linear(h2, q),
        )

    @staticmethod
    def network_input(coefficient: torch.Tensor) -> torch.Tensor:
        logk = torch.log(coefficient)
        scaled = (logk - logk.mean((-2, -1), keepdim=True)) / math.log(1e4)
        axis = torch.linspace(-1 + 1 / 16, 1 - 1 / 16, 16, device=coefficient.device, dtype=coefficient.dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack((xx, yy), 0).unsqueeze(0).expand(len(coefficient), -1, -1, -1)
        return torch.cat((scaled[:, None], coords), 1)

    @staticmethod
    def physical_position(
        batch: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        interior = torch.linspace(-1 + 2 / CELLS, 1 - 2 / CELLS, CELLS - 1, device=device, dtype=dtype)
        iy, ix = torch.meshgrid(interior, interior, indexing="ij")
        points = torch.stack((ix.reshape(-1), iy.reshape(-1)), 1)
        px, py = points[:, 0], points[:, 1]
        position = torch.stack(
            (
                px,
                py,
                torch.sin(math.pi * px),
                torch.cos(math.pi * px),
                torch.sin(2 * math.pi * px),
                torch.cos(2 * math.pi * px),
                torch.sin(math.pi * py),
                torch.cos(math.pi * py),
                torch.sin(2 * math.pi * py),
                torch.cos(2 * math.pi * py),
            ),
            1,
        )[None].expand(batch, -1, -1)
        return interior, position

    def forward(self, coefficient: torch.Tensor, cells: int = CELLS) -> torch.Tensor:
        if cells != CELLS:
            raise ValueError("the 2D block is defined for 16x16 coefficients")
        value = self.network_input(coefficient)
        e0 = self.e0(self.stem(value))
        e1 = self.e1(self.down1(e0))
        e2 = self.e2(self.down2(e1))
        e3 = self.bottleneck(self.e3(self.down3(e2)))
        levels = (e0, e1, e2, e3)
        pyramid = self.fpn_fusion[3](self.lateral[3](e3))
        for index in (2, 1, 0):
            pyramid = F.interpolate(
                pyramid, size=levels[index].shape[-2:], mode="bilinear", align_corners=True
            )
            pyramid = self.fpn_fusion[index](pyramid + self.lateral[index](levels[index]))
        interior, position = self.physical_position(len(value), value.device, value.dtype)
        local = sample_level(pyramid, interior)
        pooled = e3.mean((-2, -1))[:, None].expand(-1, INTERIOR, -1)
        return self.decoder(torch.cat((local, pooled, position), -1))


def parameter_breakdown(model: MultiscaleFPNTrialSpace) -> dict[str, int]:
    groups = {"stem": 0, "encoder": 0, "fpn": 0, "decoder": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        if name.startswith("stem"):
            groups["stem"] += count
        elif name.startswith(("lateral", "fpn_fusion")):
            groups["fpn"] += count
        elif name.startswith("decoder"):
            groups["decoder"] += count
        else:
            groups["encoder"] += count
    groups["total"] = sum(groups.values())
    return groups


def load_model(path, device="cpu") -> MultiscaleFPNTrialSpace:
    """Load a checkpoint written by train.py (any dict with "model" and optionally "architecture")."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = MultiscaleFPNTrialSpace(state.get("architecture", "LARGE"))
    model.load_state_dict(state["model"])
    return model.to(device).eval()
