"""Uncertainty-guided residual fusion. Research hypothesis, not a claimed SOTA."""
import common
import torch
from torch import nn
from torch.nn import functional as F
from octa.model import UNet


class RefineScale(nn.Module):
    def __init__(self, a_channels, o_channels, hidden=16):
        super().__init__()
        self.a = nn.Conv2d(a_channels, hidden, 1)
        self.o = nn.Conv2d(o_channels, hidden, 1)
        self.mix = nn.Sequential(nn.Conv2d(3 * hidden + 1, hidden, 3, padding=1),
                                 nn.GroupNorm(4, hidden), nn.SiLU())
        self.gate = nn.Conv2d(hidden, 1, 1)
        self.residual = nn.Conv2d(hidden, 1, 1)
        # Exactly recover the OCTA anchor at initialization.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, a, o, uncertainty, use_gate):
        a, o = self.a(a), self.o(o)
        u = F.interpolate(uncertainty, a.shape[-2:], mode="bilinear", align_corners=False)
        feature = self.mix(torch.cat((a, o, (a - o).abs(), u), dim=1))
        gate = torch.sigmoid(self.gate(feature)) if use_gate else torch.ones_like(u)
        return gate * self.residual(feature), gate


class UGRFusion(nn.Module):
    def __init__(self, base=32, oct_base=8, uncertainty=True, use_gate=True,
                 correction_limit=2.0, uncertainty_floor=0.1, aux_source="oct"):
        super().__init__()
        if correction_limit <= 0 or not 0 <= uncertainty_floor <= 1:
            raise ValueError("Invalid correction limit or uncertainty floor")
        if aux_source not in ("oct", "octa"):
            raise ValueError(aux_source)
        self.anchor = UNet(in_channels=1, base=base)
        self.uncertainty, self.use_gate = uncertainty, use_gate
        self.limit, self.floor, self.aux_source = correction_limit, uncertainty_floor, aux_source
        self.oct_blocks, self.refiners = nn.ModuleList(), nn.ModuleList()
        previous = 1
        for level in range(5):
            c = oct_base * 2 ** level
            self.oct_blocks.append(nn.Sequential(
                nn.Conv2d(previous, c, 3, padding=1, bias=False), nn.GroupNorm(1, c), nn.SiLU(),
                nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False), nn.GroupNorm(1, c), nn.SiLU()))
            self.refiners.append(RefineScale(base * 2 ** level, c))
            previous = c

    def forward_details(self, x, strength=1.0):
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError("Expected [B,2,H,W], channel order OCTA then OCT")
        net = self.anchor
        features = [net.stem(x[:, :1])]
        for down in (net.down1, net.down2, net.down3, net.down4):
            features.append(down(features[-1]))
        d = features[-1]
        for up, skip in zip((net.up1, net.up2, net.up3, net.up4), reversed(features[:-1])):
            d = up(d, skip)
        anchor = net.head(d)
        p = torch.sigmoid(anchor.detach().float())
        u = 4 * p * (1 - p) if self.uncertainty else torch.ones_like(p)
        availability = self.floor + (1 - self.floor) * u
        o = x[:, 1:2] if self.aux_source == "oct" else x[:, :1]
        residuals, gates = [], []
        for level, (block, refine, a) in enumerate(zip(self.oct_blocks, self.refiners, features)):
            if level:
                o = F.avg_pool2d(o, 2)
            o = block(o)
            residual, gate = refine(a, o, u.to(a.dtype), self.use_gate)
            residuals.append(F.interpolate(residual.float(), anchor.shape[-2:], mode="bilinear", align_corners=False))
            gates.append(F.interpolate(gate.float(), anchor.shape[-2:], mode="bilinear", align_corners=False))
        raw = torch.stack(residuals).mean(0)
        correction = strength * availability * self.limit * torch.tanh(raw / self.limit)
        return {"logits": anchor.float() + correction, "anchor": anchor,
                "uncertainty": u, "gate": torch.stack(gates).mean(0), "correction": correction}

    def forward(self, x):
        return self.forward_details(x)["logits"]


class Control(nn.Module):
    def __init__(self, mode, base):
        super().__init__()
        self.mode = mode
        self.net = UNet(in_channels=1 if mode == "octa" else 2, base=base)

    def forward_details(self, x, strength=1.0):
        logits = self.net(x[:, :1] if self.mode == "octa" else x)
        return {"logits": logits, "anchor": logits}

    def forward(self, x):
        return self.forward_details(x)["logits"]


def build_model(config):
    if config["model"] != "ugr":
        return Control(config["model"], config["base_channels"])
    return UGRFusion(base=config["base_channels"], oct_base=config["oct_base"],
                     uncertainty=config["uncertainty"], use_gate=config["gate"],
                     correction_limit=config["correction_limit"], aux_source=config["aux_source"])
