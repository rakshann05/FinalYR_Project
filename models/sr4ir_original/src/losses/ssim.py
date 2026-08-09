'''
SSIM loss for the SR update (part of the PSFL = pixel + SSIM group).

Turns structural similarity into a minimizable loss: L_ssim = loss_weight * (1 - SSIM).
Uses pytorch_msssim (`pip install pytorch_msssim`). Images are expected in [0, 1],
so data_range defaults to 1.0.
'''
import torch
import torch.nn as nn

try:
    from pytorch_msssim import ssim as _ssim
except ImportError as e:  # pragma: no cover
    _ssim = None
    _import_error = e


class SSIMLoss(nn.Module):
    """1 - SSIM loss.

    Args:
        loss_weight (float): weight applied to (1 - SSIM). Default: 1.0.
        data_range (float): dynamic range of the inputs. Default: 1.0 (images in [0,1]).
        win_size (int): Gaussian window size for SSIM. Default: 11.
        win_sigma (float): Gaussian window sigma. Default: 1.5.
    """

    def __init__(self, loss_weight=1.0, data_range=1.0, win_size=11, win_sigma=1.5):
        super().__init__()
        if _ssim is None:
            raise ImportError(
                "SSIMLoss requires pytorch_msssim. Install it with "
                "`pip install pytorch_msssim`."
            ) from _import_error
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma

    def forward(self, pred, target, **kwargs):
        if self.loss_weight <= 0:
            return pred.new_zeros(())
        s = _ssim(pred, target,
                  data_range=self.data_range,
                  win_size=self.win_size,
                  win_sigma=self.win_sigma,
                  size_average=True)
        return self.loss_weight * (1.0 - s)
