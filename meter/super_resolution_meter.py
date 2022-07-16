from typing import List, Sequence, Optional, Any

import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image import PSNR, SSIM


class PSNR_SHAVE(PSNR):
    def __init__(self, scale, gray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.gray = gray

    def update(self, preds: Tensor, target: Tensor) -> None:
        shave = self.scale
        if self.gray:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = preds.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            preds = preds.mul(convert).sum(dim=1)
            target = target.mul(convert).sum(dim=1)
        else:
            shave = self.scale + 6
        preds = preds[..., shave:-shave, shave:-shave]
        target = target[..., shave:-shave, shave:-shave]
        if preds.nelement() == 0:
            return
        super().update(preds, target)


class SSIM_SHAVE(SSIM):
    def __init__(self, scale, gray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale
        self.gray = gray

    def update(self, preds: Tensor, target: Tensor) -> None:
        shave = self.scale
        if self.gray:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = preds.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            preds = preds.mul(convert).sum(dim=1, keepdim=True)
            target = target.mul(convert).sum(dim=1, keepdim=True)
        else:
            shave = self.scale + 6
        preds = preds[..., shave:-shave, shave:-shave]
        target = target[..., shave:-shave, shave:-shave]
        if preds.nelement() == 0:
            return
        super().update(preds, target)

# class SSIM:
#     """Structure Similarity
#     img1, img2: [0, 255]"""
#
#     def __init__(self):
#         self.name = "SSIM"
#
#     @staticmethod
#     def __call__(img1, img2):
#         if not img1.shape == img2.shape:
#             raise ValueError("Input images must have the same dimensions.")
#         if img1.ndim == 2:  # Grey or Y-channel image
#             return SSIM._ssim(img1, img2)
#         elif img1.ndim == 3:
#             if img1.shape[2] == 3:
#                 ssims = []
#                 for i in range(3):
#                     ssims.append(SSIM()(img1[:, :, i], img2[:, :, i]))
#                 return np.array(ssims).mean()
#             elif img1.shape[2] == 1:
#                 return SSIM._ssim(np.squeeze(img1), np.squeeze(img2))
#         else:
#             raise ValueError("Wrong input image dimensions.")
#
#     @staticmethod
#     def _ssim(img1, img2):
#         from cv2 import filter2D, getGaussianKernel
#         C1 = (0.01 * 255) ** 2
#         C2 = (0.03 * 255) ** 2
#
#         img1 = img1.astype(np.float64)
#         img2 = img2.astype(np.float64)
#         kernel = getGaussianKernel(11, 1.5)
#         window = np.outer(kernel, kernel.transpose())
#
#         mu1 = filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#         mu2 = filter2D(img2, -1, window)[5:-5, 5:-5]
#         mu1_sq = mu1 ** 2
#         mu2_sq = mu2 ** 2
#         mu1_mu2 = mu1 * mu2
#         sigma1_sq = filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#         sigma2_sq = filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#         sigma12 = filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#         ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
#                 (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
#         )
#         return ssim_map.mean()


# class SSIM(Metric):
#     """Computes Structual Similarity Index Measure (SSIM_).
#
#     Args:
#         kernel_size: size of the gaussian kernel (default: (11, 11))
#         sigma: Standard deviation of the gaussian kernel (default: (1.5, 1.5))
#         k1: Parameter of SSIM. Default: 0.01
#         k2: Parameter of SSIM. Default: 0.03
#
#     Return:
#         Tensor with SSIM score
#     """
#
#     preds: List[Tensor]
#     target: List[Tensor]
#
#     def __init__(
#             self,
#             kernel_size: Sequence[int] = (11, 11),
#             sigma: float = 1.5,
#             k1: float = 0.01,
#             k2: float = 0.03,
#             compute_on_step: bool = True,
#             dist_sync_on_step: bool = False,
#             process_group: Optional[Any] = None,
#     ) -> None:
#         super().__init__(
#             compute_on_step=compute_on_step,
#             dist_sync_on_step=dist_sync_on_step,
#             process_group=process_group,
#         )
#         self.add_state("preds", default=[], dist_reduce_fx="cat")
#         self.add_state("target", default=[], dist_reduce_fx="cat")
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#         self.k1 = k1
#         self.k2 = k2
#
#     def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
#         """Update state with predictions and targets.
#
#         Args:
#             preds: Predictions from model
#             target: Ground truth values
#         """
#         assert preds.shape == target.shape and len(preds.shape) == 4
#         self.preds.append(preds)
#         self.target.append(target)
#
#     def compute(self) -> Tensor:
#         """Computes explained variance over state."""
#         # from skimage.measure import compare_ssim as ssim
#         from skimage.metrics import structural_similarity as ssim
#         ret = []
#         for preds, target in zip(self.preds, self.target):
#             for p, t in zip(preds, target):
#                 ret.append(ssim(p.permute((1, 2, 0)).cpu().numpy(), t.permute((1, 2, 0)).cpu().numpy(), K1=self.k1, K2=self.k2,
#                                 gaussian_weights=True, sigma=self.sigma,
#                                 use_sample_covariance=False, multichannel=p.shape[0] == 3))
#         return torch.tensor(ret).mean()
