import torch
import numpy as np
import torch.nn.functional as F
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)


def gen_LoG_kernel(sigma=1, size=7):
    X = np.arange(size//2, -size//2, -1)
    Y = np.arange(size//2, -size//2, -1)
    xx, yy = np.meshgrid(X, Y)
    LoG_kernel = 1 / (np.pi * sigma ** 4) * (1 - (xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * np.exp(- (xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return torch.from_numpy(LoG_kernel).type(torch.float32).view(1, 1, size, size)


def laplacian_edge(img):
    kernel = gen_LoG_kernel()
    beta = 0.0001
    # img = renderings[-1]['rgb']
    # Ensure input tensor is in the correct format: [B, C, H, W]
    # if img.dim() == 4 and img.shape[1] != 3:
    #     img = img.permute(0, 3, 1, 2)
    # elif img.dim() == 3:
    #     img = img.unsqueeze(1)
    # img = torch.mean(img, dim=-3, keepdim=True)
    # img = (img - img.mean(dim=[2, 3], keepdim=True)) / (img.std(dim=[2, 3], keepdim=True) + 1e-8)
    B, C, H, W = img.size()
    img_lap = F.conv2d(img, kernel.to(torch.get_device(img)), padding='same')
    blur_loss = - torch.log (torch.sum(img_lap ** 2, dim=[1, 2, 3]) / (H*W - torch.mean(img, dim=[1,2,3])**2) + 1e-8)
    return blur_loss.mean()


def horizontal_total_variation(img: torch.Tensor) -> torch.Tensor:
    """Compute total variation statistics on current batch."""
    if img.ndim != 4:
        raise RuntimeError(f"Expected input `img` to be an 4D tensor, but got {img.shape}")
    # diff1 = img[..., 1:, :] - img[..., :-1, :]
    diff2 = img[..., :, 1:] - img[..., :, :-1]

    # res1 = diff1.abs().sum([1, 2, 3])
    res2 = diff2.abs().sum([1, 2, 3])
    score = res2
    return score / img.shape[0]
