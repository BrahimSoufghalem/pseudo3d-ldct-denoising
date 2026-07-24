"""
LDCT Project — Loss Functions
================================
VGG Perceptual Loss, Sobel Edge Loss, and the combined Hybrid Loss.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import SSIMLoss 

from config import LAMBDA_L1, LAMBDA_SSIM, LAMBDA_EDGE 



# ═══════════════════════════════════════════
# SOBEL EDGE LOSS
# ═══════════════════════════════════════════
class SobelEdgeLoss(nn.Module):
    """
    Computes L1 loss on Sobel-filtered edge maps.
    Encourages the model to preserve sharp boundaries.
    """
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        sobel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def get_edges(self, x):
        c = x.shape[1]
        sobel_x = self.sobel_x.repeat(c, 1, 1, 1)
        sobel_y = self.sobel_y.repeat(c, 1, 1, 1)
        gx = F.conv2d(x, sobel_x, padding=1, groups=c)
        gy = F.conv2d(x, sobel_y, padding=1, groups=c)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return grad_mag

    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        return F.l1_loss(pred_edges, target_edges)


# ═══════════════════════════════════════════
# MONAI HYBRID LOSS
# ═══════════════════════════════════════════
class MONAIHybridLoss(nn.Module):
    """

    Combines L1 + SSIM + Edge losses
    with configurable weights for LDCT denoising.
    """
    def __init__(
        self,
        lambda_l1=LAMBDA_L1,
        lambda_ssim=LAMBDA_SSIM,
        lambda_edge=LAMBDA_EDGE,
        spatial_dims=2,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_edge = lambda_edge

        self.ssim_loss = SSIMLoss(spatial_dims=spatial_dims, data_range=1.0)
        self.edge_loss = SobelEdgeLoss()

    def forward(self, pred_img, target_img):
        l1 = F.l1_loss(pred_img, target_img)
        ssim = self.ssim_loss(pred_img, target_img)
        edge = self.edge_loss(pred_img, target_img)

        total = (
            self.lambda_l1 * l1
            + self.lambda_ssim * ssim
            + self.lambda_edge * edge
        )

        return total, {
            "L1": l1.item(),
            "SSIM": ssim.item(),
            "Edge": edge.item(),
            "Total": total.item(),
        }
