"""UNet with built-in input normalisation, and checkpoint I/O.

One class covers every UNet in the paper; they differ only in the input channels and in how raw
uint16 values are normalised (see `NormalizingUNet` and `channels.INPUTS`). Parameter and buffer
names match the original checkpoints, so those load directly.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "canopy_height/1"


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) x 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """2x2 max-pool, then DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsample, concatenate the skip connection, then DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class NormalizingUNet(nn.Module):
    """UNet whose forward pass normalises raw uint16-valued inputs.

    The first `n_embed` channels (Earth embedding) become x / 10000 - 1; the remaining channels are
    standardised with the `mean_t` / `std_t` buffers. With `zero_restore`, cells whose raw value is 0
    (no data) are set back to 0 after normalisation. The input tensor is modified in place.
    """

    def __init__(self, n_channels, n_classes=1, mean=None, std=None, n_embed=64, zero_restore=True,
                 bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_embed = n_embed
        self.zero_restore = zero_restore
        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        mean = torch.zeros(n_channels) if mean is None else torch.as_tensor(mean, dtype=torch.float32)
        std = torch.ones(n_channels) if std is None else torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("mean_t", mean.to(torch.float32))
        self.register_buffer("std_t", std.to(torch.float32))

    def normalize(self, x):
        C = x.size(1)
        e = min(self.n_embed, C)
        mask = x == 0
        if e > 0:
            x[:, :e] = x[:, :e] / 10000.0 - 1.0
        if C > e:
            x[:, e:] = (x[:, e:] - self.mean_t[None, e:C, None, None]) / \
                       (self.std_t[None, e:C, None, None] + 1e-12)
        if self.zero_restore:
            x[mask] = 0
        return x

    def forward(self, x):
        x = self.normalize(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ---------------------------------------------------------------- configuration and checkpoints
def model_config(input="AE", n_channels=None, n_embed=None, zero_restore=None, **extra):
    """Configuration stored with every checkpoint. `input` names a channel layout in channels.INPUTS."""
    from . import channels
    input = channels.canonical(input)
    spec = channels.INPUTS[input]
    return dict(arch="unet", input=input,
                n_channels=n_channels if n_channels is not None else spec["n_channels"],
                n_embed=n_embed if n_embed is not None else spec["n_embed"],
                zero_restore=bool(spec["zero_restore"] if zero_restore is None else zero_restore), **extra)


def build_model(config, mean=None, std=None):
    if config.get("arch", "unet") != "unet":
        raise ValueError(f"unsupported architecture {config['arch']}")
    return NormalizingUNet(config["n_channels"], 1, mean, std, n_embed=config["n_embed"],
                           zero_restore=config["zero_restore"])


def save_checkpoint(path, model, config, **meta):
    m = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({"format": CHECKPOINT_FORMAT, "config": config, "meta": meta, "state_dict": m.state_dict()}, path)


def load_checkpoint(path, config=None, map_location="cpu"):
    """Returns (state_dict, config). Plain state dicts (original checkpoints) need `config`."""
    obj = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and obj.get("format") == CHECKPOINT_FORMAT:
        state, config = obj["state_dict"], (config or obj["config"])
    elif config is None:
        raise ValueError(f"{path} is a plain state dict; pass its model config")
    else:
        state = obj
    if config.get("input"):                      # aliases IE, I (the paper's checkpoints) -> AE, A
        from . import channels
        config = dict(config, input=channels.canonical(config["input"]))
    return state, config


def load_model(path, config=None, device="cpu", zero_restore=None):
    """Model ready for inference or fine-tuning. `zero_restore` overrides the stored setting."""
    state, cfg = load_checkpoint(path, config)
    cfg = dict(cfg)
    if zero_restore is not None:
        cfg["zero_restore"] = bool(zero_restore)
    if "inc.double_conv.0.weight" in state:
        n = state["inc.double_conv.0.weight"].shape[1]
        assert n == cfg["n_channels"], f"checkpoint has {n} input channels, config says {cfg['n_channels']}"
    mean, std = state.get("mean_t"), state.get("std_t")
    m = build_model(cfg, mean, std)
    missing, unexpected = m.load_state_dict(state, strict=False)
    missing = [k for k in missing if k not in ("mean_t", "std_t")]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return m.to(device), cfg


def describe(path):
    _, cfg = load_checkpoint(path)
    return json.dumps(cfg, indent=1)

