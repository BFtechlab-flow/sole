import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        g = 8 if cout % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(g, cout),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiScaleDilatedContext(nn.Module):
    """Innovation #2: depthwise multi-dilation context for long curved structures."""

    def __init__(self, channels, dilations=(1, 3, 6, 9)):
        super().__init__()
        self.branches = nn.ModuleList()
        groups = 8 if channels % 8 == 0 else 1
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        3,
                        padding=d,
                        dilation=d,
                        groups=channels,
                        bias=False,
                    ),
                    nn.Conv2d(channels, channels, 1, bias=False),
                    nn.GroupNorm(groups, channels),
                    nn.SiLU(),
                )
            )
        self.logits = nn.Parameter(torch.zeros(len(dilations)))
        self.out = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        weights = torch.softmax(self.logits, dim=0)
        z = sum(w * branch(x) for w, branch in zip(weights, self.branches))
        return self.out(z)


class FrequencyResidualBranch(nn.Module):
    """Innovation #3: explicit high-frequency/ridge branch for thin filament threads."""

    def __init__(self, out_channels):
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, size):
        raw = x[:, :1]
        clahe = x[:, 1:2]
        ridge = x[:, 2:3]
        hp = raw - F.avg_pool2d(raw, 9, stride=1, padding=4)
        freq = torch.cat([hp, clahe - raw, ridge], dim=1)
        freq = F.interpolate(freq, size=size, mode="bilinear", align_corners=False)
        return self.net(freq)


class SolarLimbGate(nn.Module):
    """Innovation #1: radial/limb-aware gating using the existing radial input channel."""

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(2, max(8, channels // 8), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(8, channels // 8), 1, 1),
        )

    def forward(self, z, x):
        radial = F.interpolate(x[:, 3:4], size=z.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate(torch.cat([radial, 1.0 - radial], dim=1)))
        return z * (0.5 + gate)


class AdaptiveFeatureMixer(nn.Module):
    """Per-pixel fusion of FPN, context and frequency evidence."""

    def __init__(self, channels):
        super().__init__()
        self.score = nn.Conv2d(channels * 3, 3, 1)
        self.out = ConvBlock(channels, channels)

    def forward(self, base, context, frequency):
        stack = torch.cat([base, context, frequency], dim=1)
        weights = torch.softmax(self.score(stack), dim=1)
        fused = (
            base * weights[:, 0:1]
            + context * weights[:, 1:2]
            + frequency * weights[:, 2:3]
        )
        return self.out(fused)


class FILANet(nn.Module):
    def __init__(
        self,
        encoder="convnext_tiny.fb_in1k",
        pretrained=True,
        fpn_channels=128,
        in_chans=4,
        advanced=None,
    ):
        super().__init__()
        self.advanced = dict(advanced or {})
        self.advanced_enabled = bool(self.advanced.get("enabled", False))

        self.encoder = timm.create_model(
            encoder,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        cs = self.encoder.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, fpn_channels, 1) for c in cs])
        self.ref = nn.ModuleList([ConvBlock(fpn_channels, fpn_channels) for _ in cs])
        self.fuse = ConvBlock(fpn_channels * len(cs), fpn_channels)

        if self.advanced_enabled:
            self.context = MultiScaleDilatedContext(fpn_channels)
            self.frequency = FrequencyResidualBranch(fpn_channels)
            self.mixer = AdaptiveFeatureMixer(fpn_channels)
            self.limb_gate = SolarLimbGate(fpn_channels)

        self.heads = nn.ModuleDict(
            {
                k: nn.Conv2d(fpn_channels, 1, 1)
                for k in ("region", "centerline", "boundary", "distance")
            }
        )
        self.orientation_head = nn.Conv2d(fpn_channels, 2, 1)

        if self.advanced_enabled:
            # Innovations #4-#7 and #10 can be ablated independently.
            mapping = {
                "width": "width_head",
                "curvature": "curvature_head",
                "endpoint": "endpoint_head",
                "junction": "junction_head",
                "uncertainty": "uncertainty_head",
            }
            self.advanced_heads = nn.ModuleDict({
                name: nn.Conv2d(fpn_channels, 1, 1)
                for name, flag in mapping.items()
                if bool(self.advanced.get(flag, True))
            })

    def forward(self, x):
        size = x.shape[-2:]
        f = self.encoder(x)
        p = [None] * len(f)
        p[-1] = self.lat[-1](f[-1])
        for i in range(len(f) - 2, -1, -1):
            p[i] = self.lat[i](f[i]) + F.interpolate(
                p[i + 1],
                size=f[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        p = [self.ref[i](p[i]) for i in range(len(p))]
        base_size = p[0].shape[-2:]
        z = self.fuse(
            torch.cat(
                [
                    F.interpolate(q, size=base_size, mode="bilinear", align_corners=False)
                    for q in p
                ],
                1,
            )
        )

        if self.advanced_enabled:
            use_context = bool(self.advanced.get("multiscale_context", True))
            use_frequency = bool(self.advanced.get("frequency_branch", True))
            context = self.context(z) if use_context else z
            frequency = self.frequency(x, z.shape[-2:]) if use_frequency else z
            if use_context or use_frequency:
                z = self.mixer(z, context, frequency)
            if bool(self.advanced.get("limb_gate", True)):
                z = self.limb_gate(z, x)

        z = F.interpolate(z, size=size, mode="bilinear", align_corners=False)
        out = {k: h(z) for k, h in self.heads.items()}
        out["orientation"] = self.orientation_head(z)
        if self.advanced_enabled:
            out.update({k: h(z) for k, h in self.advanced_heads.items()})
        return out
