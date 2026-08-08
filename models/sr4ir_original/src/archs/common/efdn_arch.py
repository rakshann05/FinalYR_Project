'''
EFDN: Edge-enhanced Feature Distillation Network for Efficient Super-Resolution
Code Reference (official): https://github.com/icandle/EFDN

This file integrates EFDN as an SR backbone option for SR4IR.

SR4IR builds SR networks via `archs.common.<name>_arch.build_network(**opt)`, and
expects a module whose forward maps a low-resolution image in [0, 1] to a
super-resolved image in [0, 1] at `scale` x. EFDN satisfies this directly
(its forward clamps the output to [0, 1]).

EFDN has two modes:
  * multi-branch *training* mode  -> each EDBB block runs 5 parallel branches
    (3x3 rep_conv + 1x1 + Sobel-x + Sobel-y + Laplacian). This is what we train,
    and what the pixel / EG / TDP losses back-propagate through.
  * re-parameterized *deploy* mode -> every EDBB block is folded into a single
    3x3 conv + PReLU (EDBB_deploy-equivalent). This is what you MUST switch to
    before measuring params / FLOPs / runtime, otherwise you measure train-time
    compute instead of deploy-time compute.

The multi-branch block (EDBB) and the folding math (`switch_to_deploy`) are taken
verbatim from EFDN's `models/unitv2.py`, vendored here as `efdn_unitv2.py`, so the
reparameterization is bit-for-bit the authors' implementation (no re-derivation).

The surrounding EFDN topology (ESA attention, Cell, head/body/tail) is taken from
EFDN's `models/EFDN_deploy.py`; the only change is that `Cell` uses the trainable
multi-branch `EDBB` instead of the already-folded `EDBB_deploy` when `deploy=False`.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from .efdn_unitv2 import EDBB, EDBB_deploy


def build_network(**kwargs):
    # SR4IR pops `name`/`trainable` and injects `scale`; the rest are EFDN kwargs.
    return EFDN(**kwargs)


# --------------------------------------------------------------------------- #
# ESA attention + 1x1-PReLU conv, verbatim from EFDN/models/EFDN_deploy.py
# --------------------------------------------------------------------------- #
class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class conv(nn.Module):
    def __init__(self, n_feats):
        super(conv, self).__init__()
        self.conv1x1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        self.act = nn.PReLU(num_parameters=n_feats)

    def forward(self, x):
        return self.act(self.conv1x1(x))


# --------------------------------------------------------------------------- #
# Cell: EFDN body block. Uses multi-branch EDBB in training mode.
# --------------------------------------------------------------------------- #
class Cell(nn.Module):
    def __init__(self, n_feats=48, deploy=False, act_type='prelu'):
        super(Cell, self).__init__()

        self.conv1 = conv(n_feats)
        if deploy:
            # already-folded single-conv blocks (inference topology)
            self.conv2 = EDBB_deploy(n_feats, n_feats)
            self.conv3 = EDBB_deploy(n_feats, n_feats)
        else:
            # multi-branch trainable blocks (default EFDN EDBB branch set:
            # rep_conv 3x3 + 1x1 + Sobel-x + Sobel-y + Laplacian, then PReLU)
            self.conv2 = EDBB(n_feats, n_feats, act_type=act_type)
            self.conv3 = EDBB(n_feats, n_feats, act_type=act_type)

        self.fuse = nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0)
        self.att = ESA(n_feats, nn.Conv2d)
        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats // 2, 1, 1, 0) for _ in range(4)])

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out = self.fuse(torch.cat([self.branch[0](x), self.branch[1](out1),
                                   self.branch[2](out2), self.branch[3](out3)], dim=1))
        out = self.att(out)
        out += x
        return out


# --------------------------------------------------------------------------- #
# EFDN network
# --------------------------------------------------------------------------- #
class EFDN(nn.Module):
    """EFDN super-resolution backbone.

    Args:
        scale (int): SR upscaling factor (injected by SR4IR from `opt['scale']`).
        in_channels (int): input channels. Default: 3.
        n_feats (int): feature width. Default: 48 (EFDN paper setting).
        out_channels (int): output channels. Default: 3.
        deploy (bool): if True, build the already-folded single-conv topology
            (for loading reparameterized weights / benchmarking). Default: False.
    """

    def __init__(self, scale=4, in_channels=3, n_feats=48, out_channels=3, deploy=False):
        super(EFDN, self).__init__()
        self.scale = scale
        self.deploy = deploy

        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        self.cells = nn.ModuleList([Cell(n_feats, deploy=deploy) for _ in range(4)])
        self.local_fuse = nn.ModuleList([nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0) for _ in range(3)])
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        out0 = self.head(x)
        out1 = self.cells[0](out0)
        out2 = self.cells[1](out1)
        out2_fuse = self.local_fuse[0](torch.cat([out1, out2], dim=1))
        out3 = self.cells[2](out2_fuse)
        out3_fuse = self.local_fuse[1](torch.cat([out2, out3], dim=1))
        out4 = self.cells[3](out3_fuse)
        out4_fuse = self.local_fuse[2](torch.cat([out2, out4], dim=1))
        out = out4_fuse + out0
        out = self.tail(out)
        return out.clamp(0, 1)

    @torch.no_grad()
    def reparameterize(self):
        """Fold every multi-branch EDBB into a single 3x3 conv (in place).

        Uses EFDN's own `EDBB.switch_to_deploy()` (vendored, unmodified), so the
        folded weights are numerically identical to the authors' implementation.
        After this call the network's forward is unchanged numerically but runs
        the deploy-time compute; call it before measuring params/FLOPs/runtime.
        Returns self for chaining.
        """
        for m in self.modules():
            if isinstance(m, EDBB):
                m.switch_to_deploy()
        self.deploy = True
        return self
