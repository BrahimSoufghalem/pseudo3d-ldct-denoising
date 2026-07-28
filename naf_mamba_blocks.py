"""
NAF-MambaUNet Building Blocks
================================================================
Derived from:
  1. Megvii NAFNet (Megvii-Research/NAFNet): NAFBlock, SimpleGate, SCA, LayerNorm2d
  2. VMamba / MambaVision: 4-way cross-scan selective state-space (SS2D)
  3. Anatomy-guided attention skip gates.

Selective scan backend
----------------------
The S6 recurrence is delegated to the OFFICIAL `mamba_ssm` CUDA kernel
(`selective_scan_fn`) whenever it is importable and the tensors live on CUDA.
A chunked pure-PyTorch reference implementation is kept as a fallback so the
model still runs on CPU (unit tests, debugging) - it is correct but slow.

Install the fast path with:
    pip install mamba-ssm --no-build-isolation
"""

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional official Mamba kernel
# ---------------------------------------------------------------------------
try:  # pragma: no cover - depends on the local environment
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _selective_scan_cuda
    HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover
    _selective_scan_cuda = None
    HAS_MAMBA_SSM = False


# ══════════════════════════════════════════════════════════════
# 1. MEGVII NAFNET COMPONENTS (Activation-Free Restoration)
# ══════════════════════════════════════════════════════════════

class LayerNormFunction(torch.autograd.Function):
    """Memory-efficient channel LayerNorm (CUDA fast path)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_tensors
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


def _layer_norm_2d_ref(x, weight, bias, eps):
    """Device-agnostic channel LayerNorm (CPU / MPS path)."""
    dtype = x.dtype
    x = x.float()
    mu = x.mean(dim=1, keepdim=True)
    var = x.var(dim=1, keepdim=True, unbiased=False)
    y = (x - mu) / torch.sqrt(var + eps)
    y = weight.float().view(1, -1, 1, 1) * y + bias.float().view(1, -1, 1, 1)
    return y.to(dtype)


class LayerNorm2d(nn.Module):
    """2D spatial (channel) LayerNorm from Megvii NAFNet.

    Dispatches to a custom autograd kernel on CUDA and to a plain PyTorch
    implementation elsewhere, so CPU inference and unit tests work.
    """

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        if x.is_cuda:
            return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
        return _layer_norm_2d_ref(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Element-wise multiplication (x1 * x2) replacing non-linear activations."""

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Channel attention with 1x1 Conv from Megvii NAFNet."""

    def __init__(self, c):
        super().__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

    def forward(self, x):
        return x * self.sca(x)


class NAFBlock(nn.Module):
    """
    Non-Linear Activation-Free (NAF) block from Megvii NAFNet.

    `beta`/`gamma` are zero-initialised exactly as in the reference NAFNet
    implementation, so every block starts as an identity mapping. Gradients
    still reach the internal convolutions on the very first step (d(x*beta)/d(beta) = x),
    which is why the original authors chose zeros for training stability.
    """

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.sca = SimplifiedChannelAttention(dw_channel // 2)
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


def make_naf_stage(channels, num_blocks):
    """Stack `num_blocks` NAF blocks into a single sequential stage."""
    num_blocks = max(1, int(num_blocks))
    return nn.Sequential(*[NAFBlock(channels) for _ in range(num_blocks)])


# ══════════════════════════════════════════════════════════════
# 2. PSEUDO-3D (Z-AXIS) INPUT STEMS
# ══════════════════════════════════════════════════════════════

class Slice2DStem(nn.Module):
    """Plain 2D stem: a single centre slice in, `width` feature maps out."""

    def __init__(self, in_channels, width):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class PseudoDepthStem(nn.Module):
    """
    Explicit z-axis stem for 2.5D input.

    The original code fed (prev, curr, next) straight into a 2D conv, so the
    "pseudo-3D" claim reduced to "three input channels". Here the slice stack is
    treated as a real depth axis:

      branch A : Conv3d over (D=3, H, W) -> learns inter-slice relations, then
                 collapses the depth axis to a single plane.
      branch B : Conv2d on the centre slice only -> guarantees an intact 2D path
                 (the model can ignore the neighbours if they do not help).

    The two branches are concatenated to `width` channels.
    """

    def __init__(self, in_channels, width):
        super().__init__()
        if in_channels < 2:
            raise ValueError("PseudoDepthStem expects at least 2 input slices")
        self.in_channels = in_channels
        self.centre_index = in_channels // 2

        depth_width = width // 2
        centre_width = width - depth_width

        self.depth_conv = nn.Conv3d(
            1, depth_width,
            kernel_size=(in_channels, 3, 3),
            padding=(0, 1, 1),
            bias=True,
        )
        self.centre_conv = nn.Conv2d(1, centre_width, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        centre = x[:, self.centre_index:self.centre_index + 1]
        depth = self.depth_conv(x.unsqueeze(1)).squeeze(2)   # [B, dw, H, W]
        return torch.cat([depth, self.centre_conv(centre)], dim=1)


def build_stem(in_channels, width, input_mode="2.5d"):
    """Factory returning the stem matching the requested input mode."""
    if str(input_mode).lower().replace(".", "") == "2d" or in_channels == 1:
        return Slice2DStem(in_channels, width)
    return PseudoDepthStem(in_channels, width)


# ══════════════════════════════════════════════════════════════
# 3. ANATOMY-GUIDED ATTENTION SKIP GATE
# ══════════════════════════════════════════════════════════════

class AnatomyAttentionGate2D(nn.Module):
    """
    Context-guided attention gate with tanh scaling in [0.5, 1.5].

    Allows both noise suppression (<1.0) and clean-signal amplification (>1.0),
    initialised at exact identity (psi is zero-initialised).

    IMPORTANT: `g` must be a genuinely deeper/coarser context tensor coming from
    the bottleneck or the decoder path. Feeding a strided copy of `x` itself
    (as an earlier revision did) makes the gate carry no new information.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            LayerNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh(),
        )
        nn.init.zeros_(self.psi[0].weight)
        nn.init.zeros_(self.psi[0].bias)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        alpha = self.psi(F.gelu(g1 + x1))
        return x * (1.0 + 0.5 * alpha)


# ══════════════════════════════════════════════════════════════
# 4. SELECTIVE SCAN (S6) BACKENDS
# ══════════════════════════════════════════════════════════════

def selective_scan_ref(u, delta, A, B, C, D=None, delta_bias=None,
                       delta_softplus=True, chunk_size=32):
    """
    Chunked pure-PyTorch selective scan (fallback for CPU / no mamba-ssm).

    Shapes
    ------
    u, delta : [b, d, l]
    A        : [d, n]
    B, C     : [b, g, n, l]   (g groups, d must be divisible by g)
    D        : [d]
    """
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    b, d, l = u.shape
    g = B.shape[1]
    n = A.shape[1]
    if d % g != 0:
        raise ValueError("channel count must be divisible by the number of groups")
    per_group = d // g

    h = u.new_zeros((b, d, n))
    outputs = []

    for start in range(0, l, chunk_size):
        end = min(start + chunk_size, l)
        u_c = u[:, :, start:end]                                     # [b,d,lc]
        dt_c = delta[:, :, start:end]                                # [b,d,lc]
        B_c = B[:, :, :, start:end].repeat_interleave(per_group, dim=1)   # [b,d,n,lc]
        C_c = C[:, :, :, start:end].repeat_interleave(per_group, dim=1)   # [b,d,n,lc]

        dA = torch.exp(dt_c.unsqueeze(-1) * A.view(1, d, 1, n))      # [b,d,lc,n]
        dBu = dt_c.unsqueeze(-1) * B_c.permute(0, 1, 3, 2) * u_c.unsqueeze(-1)

        states = []
        for t in range(end - start):
            h = dA[:, :, t] * h + dBu[:, :, t]
            states.append(h)
        hs = torch.stack(states, dim=2)                              # [b,d,lc,n]
        outputs.append((hs * C_c.permute(0, 1, 3, 2)).sum(dim=-1))   # [b,d,lc]

    y = torch.cat(outputs, dim=2)
    if D is not None:
        y = y + u * D.view(1, -1, 1)
    return y


def selective_scan(u, delta, A, B, C, D=None, delta_bias=None,
                   delta_softplus=True, backend="auto", chunk_size=32):
    """Dispatch to the official CUDA kernel when possible, else the reference scan."""
    backend = str(backend).lower()
    can_use_cuda = HAS_MAMBA_SSM and u.is_cuda

    if backend == "cuda" and not can_use_cuda:
        raise RuntimeError(
            "SCAN_BACKEND='cuda' but the official mamba_ssm kernel is unavailable "
            "(install with `pip install mamba-ssm --no-build-isolation`) or the "
            "tensors are not on a CUDA device."
        )

    if backend in ("auto", "cuda") and can_use_cuda:
        return _selective_scan_cuda(
            u, delta, A, B, C, D, None, delta_bias, delta_softplus,
        )

    return selective_scan_ref(
        u, delta, A, B, C, D=D, delta_bias=delta_bias,
        delta_softplus=delta_softplus, chunk_size=chunk_size,
    )


# ══════════════════════════════════════════════════════════════
# 5. 2D CROSS-SCAN SELECTIVE STATE-SPACE (SS2D)
# ══════════════════════════════════════════════════════════════

class CrossScan2D(nn.Module):
    """
    4-way cross scan of a 2D feature map -> [B, K, C, L].
      k=0 horizontal forward, k=1 horizontal backward,
      k=2 vertical forward,   k=3 vertical backward.
    """

    def forward(self, x):
        x1 = x.flatten(2)
        x2 = torch.flip(x1, dims=[-1])
        x3 = x.transpose(2, 3).flatten(2)
        x4 = torch.flip(x3, dims=[-1])
        return torch.stack([x1, x2, x3, x4], dim=1)


class CrossMerge2D(nn.Module):
    """Inverse of CrossScan2D: [B, K, C, L] -> [B, C, H, W]."""

    def forward(self, ys, H, W):
        B, K, C, L = ys.shape
        y1 = ys[:, 0]
        y2 = torch.flip(ys[:, 1], dims=[-1])
        y3 = ys[:, 2].reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
        y4 = torch.flip(ys[:, 3], dims=[-1]).reshape(B, C, W, H).transpose(2, 3).reshape(B, C, L)
        y = (y1 + y2 + y3 + y4) * 0.25
        return y.reshape(B, C, H, W)


class SS2DMambaBottleneck(nn.Module):
    """
    2D cross-scan selective state-space (SS2D) block.

    Each of the K scan directions owns its OWN S6 parameter set
    (`x_proj`, `dt_proj`, `A`, `D`), matching the VMamba formulation. The four
    directions are executed in a single fused selective-scan call by mapping
    them onto the `groups` dimension of the kernel.

    `dt` is produced with softplus only - no upper clamp - so the step size is
    free to grow; the official Mamba initialisation already places it in a sane
    range and clamping would zero the gradient outside the bounds.
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank=None,
                 n_directions=4, scan_backend="auto", scan_chunk_size=32,
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.K = n_directions
        self.dt_rank = dt_rank or math.ceil(d_model / 16)
        self.scan_backend = scan_backend
        self.scan_chunk_size = scan_chunk_size

        self.norm = LayerNorm2d(d_model)
        self.in_proj = nn.Conv2d(d_model, self.d_inner * 2, kernel_size=1, bias=False)
        self.dw_conv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                 padding=d_conv // 2, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()

        self.cross_scan = CrossScan2D()
        self.cross_merge = CrossMerge2D()

        # ---- per-direction S6 parameters -------------------------------
        x_proj_weight = torch.empty(self.K, self.dt_rank + 2 * d_state, self.d_inner)
        for k in range(self.K):
            nn.init.kaiming_uniform_(x_proj_weight[k], a=math.sqrt(5))
        self.x_proj_weight = nn.Parameter(x_proj_weight)

        dt_init_std = self.dt_rank ** -0.5
        dt_projs_weight = torch.empty(self.K, self.d_inner, self.dt_rank).uniform_(-dt_init_std, dt_init_std)
        self.dt_projs_weight = nn.Parameter(dt_projs_weight)

        dt = torch.exp(
            torch.rand(self.K, self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))      # inverse of softplus
        self.dt_projs_bias = nn.Parameter(inv_dt)
        self.dt_projs_bias._no_weight_decay = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.view(1, 1, d_state).repeat(self.K, self.d_inner, 1).reshape(self.K * self.d_inner, d_state)
        self.A_logs = nn.Parameter(torch.log(A))
        self.A_logs._no_weight_decay = True

        self.Ds = nn.Parameter(torch.ones(self.K * self.d_inner))
        self.Ds._no_weight_decay = True
        # ----------------------------------------------------------------

        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, d_model, kernel_size=1, bias=False)
        nn.init.zeros_(self.out_proj.weight)      # starts as an identity residual block

    def _scan(self, x_conv):
        B, C, H, W = x_conv.shape
        L = H * W
        K, Dn, N = self.K, self.d_inner, self.d_state

        xs = self.cross_scan(x_conv).float()                                  # [B,K,C,L]
        x_dbl = torch.einsum("bkcl,kdc->bkdl", xs, self.x_proj_weight.float())
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, N, N], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_projs_weight.float())

        u = xs.reshape(B, K * Dn, L).contiguous()
        dts = dts.reshape(B, K * Dn, L).contiguous()
        As = -torch.exp(self.A_logs.float())                                  # [K*C, N]
        Bs = Bs.contiguous()                                                  # [B,K,N,L]
        Cs = Cs.contiguous()

        ys = selective_scan(
            u, dts, As, Bs, Cs,
            D=self.Ds.float(),
            delta_bias=self.dt_projs_bias.float().reshape(-1),
            delta_softplus=True,
            backend=self.scan_backend,
            chunk_size=self.scan_chunk_size,
        )
        return self.cross_merge(ys.view(B, K, Dn, L), H, W)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        H, W = x.shape[2], x.shape[3]

        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=1)
        x_conv = self.act(self.dw_conv(x_branch))

        # The SSM recurrence is numerically sensitive: always run it in fp32.
        with torch.autocast(device_type=x.device.type, enabled=False):
            merged = self._scan(x_conv.float())

        merged = merged.to(residual.dtype)
        out = self.out_norm(merged) * self.act(z_branch)
        return residual + self.out_proj(out)


class ResidualMambaBottleneck(nn.Module):
    """
    Two sequentially stacked SS2D blocks.

    Each block already carries its own internal residual connection
    (`return residual + out`), so no extra addition is inserted between them.
    Used when MAMBA_MODE is "residual" or "full".
    """

    def __init__(self, d_model, d_state=16, **kwargs):
        super().__init__()
        self.mamba1 = SS2DMambaBottleneck(d_model, d_state=d_state, **kwargs)
        self.mamba2 = SS2DMambaBottleneck(d_model, d_state=d_state, **kwargs)

    def forward(self, x):
        return self.mamba2(self.mamba1(x))


# ══════════════════════════════════════════════════════════════
# 6. MULTI-SCALE SPATIAL FUSION (1/16 Mamba <-> 1/8 NAF)
# ══════════════════════════════════════════════════════════════

class MultiScaleSpatialFusion(nn.Module):
    """
    Fuses low-resolution Mamba state-space features (1/16) with high-resolution
    NAF features (1/8). Zero-initialised output conv, so the block is an exact
    identity at step 0 - consistent with the rest of the network and keeping the
    ablation between modes clean.
    """

    def __init__(self, in_c_low, in_c_high, out_c):
        super().__init__()
        self.upsample_low = nn.Sequential(
            nn.Conv2d(in_c_low, in_c_high * 4, kernel_size=1, bias=False),
            nn.PixelShuffle(2),
            LayerNorm2d(in_c_high),
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_c_high * 2, out_c, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_c),
            SimpleGate(),
            nn.Conv2d(out_c // 2, out_c, kernel_size=1, bias=False),
        )
        nn.init.zeros_(self.fuse_conv[-1].weight)

    def forward(self, feat_low, feat_high):
        up_low = self.upsample_low(feat_low)
        if up_low.shape[2:] != feat_high.shape[2:]:
            up_low = F.interpolate(up_low, size=feat_high.shape[2:], mode='bilinear', align_corners=False)
        concat = torch.cat([up_low, feat_high], dim=1)
        return feat_high + self.fuse_conv(concat)


def warn_if_slow_scan():
    """Emit a one-time warning when the slow PyTorch fallback will be used."""
    if not HAS_MAMBA_SSM:
        warnings.warn(
            "mamba-ssm is not installed: falling back to the chunked PyTorch "
            "selective scan, which is correct but MUCH slower. "
            "Install with `pip install mamba-ssm --no-build-isolation`.",
            RuntimeWarning,
            stacklevel=2,
        )
