'''
EG loss = Edge loss + Gradient-Variance loss, for the EFDN backbone.

IMPORTANT / PROVENANCE:
  EFDN's official repo (https://github.com/icandle/EFDN) ships only the model
  and inference/reparameterization code -- it does NOT release the training-time
  "EG" loss implementation. This module is therefore a re-implementation based on
  the described components, NOT the authors' original code:

    * Gradient-Variance (GV) term: from "Gradient Variance Loss for
      Structure-Enhanced Image Super-Resolution" (Abrahamyan et al., ICASSP 2022).
      Sobel gradient maps are computed on the luminance channel, split into
      non-overlapping patches, and the per-patch variance of the gradient is
      matched between SR and HR with an L2 criterion. This penalizes the loss of
      high-frequency structure that a plain pixel loss over-smooths.
    * Edge term: L1 between Sobel gradient magnitudes of SR and HR.

  Because this is a re-implementation, treat its exact behavior as something to
  verify. It is intentionally configurable (edge_weight / gv_weight) and can be
  ablated away entirely by removing `eg_opt` from the config (the brief's plan
  keeps EG on by default and ablates later).

Weighted total: loss_weight * (edge_weight * L_edge + gv_weight * L_gv).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rgb_to_luma(x):
    # x: (N, 3, H, W) in [0, 1]. Rec.601 luma.
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class EGLoss(nn.Module):
    """Edge-enhanced Gradient-Variance loss.

    Args:
        loss_weight (float): overall weight applied to the combined EG loss.
        edge_weight (float): weight of the Sobel edge-magnitude L1 term.
        gv_weight (float): weight of the gradient-variance term.
        patch_size (int): side of the non-overlapping windows used to compute
            the local gradient variance. Default: 8.
        eps (float): numerical floor for the variance. Default: 1e-6.
    """

    def __init__(self, loss_weight=1.0, edge_weight=1.0, gv_weight=1.0,
                 patch_size=8, eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.edge_weight = edge_weight
        self.gv_weight = gv_weight
        self.patch_size = patch_size
        self.eps = eps

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = sobel_x.t().contiguous()
        # (2, 1, 3, 3): one output channel per Sobel direction
        kernel = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)
        self.register_buffer('sobel', kernel)

    def _gradients(self, luma):
        # luma: (N, 1, H, W) -> gx, gy each (N, 1, H, W)
        g = F.conv2d(luma, self.sobel, padding=1)
        gx, gy = g[:, 0:1], g[:, 1:2]
        return gx, gy

    def _patch_variance(self, grad):
        # grad: (N, 1, H, W) -> per-patch variance map via unfold.
        ps = self.patch_size
        n, c, h, w = grad.shape
        # crop to a multiple of patch_size so windows tile exactly
        h2, w2 = (h // ps) * ps, (w // ps) * ps
        if h2 == 0 or w2 == 0:
            return grad.new_zeros((n, c, 1))
        grad = grad[:, :, :h2, :w2]
        patches = F.unfold(grad, kernel_size=ps, stride=ps)  # (N, ps*ps, L)
        return patches.var(dim=1, unbiased=False)            # (N, L)

    def forward(self, pred, target, **kwargs):
        if self.loss_weight <= 0:
            return pred.new_zeros(())

        luma_p = _rgb_to_luma(pred)
        luma_t = _rgb_to_luma(target)

        gx_p, gy_p = self._gradients(luma_p)
        gx_t, gy_t = self._gradients(luma_t)

        # --- edge magnitude L1 ---
        mag_p = torch.sqrt(gx_p ** 2 + gy_p ** 2 + self.eps)
        mag_t = torch.sqrt(gx_t ** 2 + gy_t ** 2 + self.eps)
        l_edge = F.l1_loss(mag_p, mag_t)

        # --- gradient variance (x and y directions) ---
        vx_p, vx_t = self._patch_variance(gx_p), self._patch_variance(gx_t)
        vy_p, vy_t = self._patch_variance(gy_p), self._patch_variance(gy_t)
        l_gv = F.mse_loss(vx_p, vx_t) + F.mse_loss(vy_p, vy_t)

        return self.loss_weight * (self.edge_weight * l_edge + self.gv_weight * l_gv)
