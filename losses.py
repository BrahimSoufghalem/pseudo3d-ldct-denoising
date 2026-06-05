"""
LDCT Project — Loss Functions
================================
VGG Perceptual Loss, Sobel Edge Loss, and the combined Hybrid Loss.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from monai.losses import SSIMLoss

from config import LAMBDA_L1, LAMBDA_SSIM, LAMBDA_PERC, LAMBDA_EDGE


# ═══════════════════════════════════════════
# VGG PERCEPTUAL LOSS
# ═══════════════════════════════════════════
class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual loss using intermediate VGG-19 feature maps.
    Measures high-level structural similarity between prediction and target.
    """
    def __init__(self, feature_layers=(4, 9, 18), weights=(1.0, 1.0, 1.0)):
        super().__init__()
        self.feature_layers = feature_layers
        self.weights = weights

        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.slices = nn.ModuleList()
        prev_idx = 0
        for layer_idx in feature_layers:
            self.slices.append(nn.Sequential(*list(vgg.children())[prev_idx:layer_idx]))
            prev_idx = layer_idx

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _preprocess(self, x):
        """Normalize input to ImageNet statistics. Repeat channels if grayscale."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred_v = self._preprocess(pred)
        target_v = self._preprocess(target.detach())
        total_loss = 0.0
        x_p, x_t = pred_v, target_v
        for slice_net, w in zip(self.slices, self.weights):
            x_p = slice_net(x_p)
            x_t = slice_net(x_t)
            total_loss += w * F.l1_loss(x_p, x_t)
        return total_loss


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
    Combines L1 + SSIM + Perceptual + Edge losses
    with configurable weights for LDCT denoising.
    """
    def __init__(
        self,
        lambda_l1=LAMBDA_L1,
        lambda_ssim=LAMBDA_SSIM,
        lambda_perc=LAMBDA_PERC,
        lambda_edge=LAMBDA_EDGE,
        spatial_dims=2,
    ):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perc = lambda_perc
        self.lambda_edge = lambda_edge

        self.ssim_loss = SSIMLoss(spatial_dims=spatial_dims, data_range=1.0)
        self.perc_loss = VGGPerceptualLoss()
        self.edge_loss = SobelEdgeLoss()

    def forward(self, pred_img, target_img):
        l1 = F.l1_loss(pred_img, target_img)
        ssim = self.ssim_loss(pred_img, target_img)
        perc = self.perc_loss(pred_img, target_img)
        edge = self.edge_loss(pred_img, target_img)

        total = (
            self.lambda_l1 * l1
            + self.lambda_ssim * ssim
            + self.lambda_perc * perc
            + self.lambda_edge * edge
        )

        return total, {
            "L1": l1.item(),
            "SSIM": ssim.item(),
            "Perceptual": perc.item(),
            "Edge": edge.item(),
            "Total": total.item(),
        }
