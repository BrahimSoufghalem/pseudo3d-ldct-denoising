"""
CPU smoke test for MS-NAFMambaNet
===================================
Verifies, for every (INPUT_MODE x MAMBA_MODE) combination:
  * the forward pass runs end-to-end on CPU (exercising the PyTorch fallback
    selective scan, i.e. SCAN_BACKEND="ref"),
  * the output shape equals the input spatial shape,
  * non-multiple-of-16 inputs work thanks to the padding wrapper,
  * the model is an exact identity residual at initialisation (zero head),
  * a backward pass produces finite gradients.

Run with a tiny width so it finishes in seconds:
    python test_shapes.py
"""

import itertools

import torch

import config as cfg
from model import MSNAFMambaNet
from utils import build_model_input, extract_centre_slice

WIDTH = 8
D_STATE = 4
SIZES = [(32, 32), (30, 45)]     # the second one is not a multiple of 16


def run_case(input_mode, mamba_mode, h, w):
    torch.manual_seed(0)
    model = MSNAFMambaNet(
        input_mode=input_mode,
        mamba_mode=mamba_mode,
        width=WIDTH,
        d_state=D_STATE,
        scan_backend="ref",
        verbose=False,
    )
    in_ch = cfg.in_channels_for(input_mode)
    x = torch.rand(2, in_ch, h, w)

    out = model(x)
    assert out.shape == (2, cfg.OUT_CHANNELS, h, w), f"bad shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite output"

    # zero-initialised head -> pure identity residual at step 0
    assert out.abs().max().item() == 0.0, "model is not identity at init"

    centre = extract_centre_slice(x)
    assert centre.shape == (2, 1, h, w)

    loss = (centre + out).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK  input={input_mode:5s} mamba={mamba_mode:11s} "
          f"size={h}x{w}  out={tuple(out.shape)}  params={n_params:,}")


def test_build_model_input():
    prev, curr, nxt = (torch.rand(1, 1, 16, 16) * 2000 - 1024 for _ in range(3))
    x2d = build_model_input(prev, curr, nxt, input_mode="2d")
    x25 = build_model_input(prev, curr, nxt, input_mode="2.5d")
    assert x2d.shape == (1, 1, 16, 16), x2d.shape
    assert x25.shape == (1, 3, 16, 16), x25.shape
    assert 0.0 <= x25.min().item() and x25.max().item() <= 1.0
    print("  OK  build_model_input 2d / 2.5d")


def main():
    torch.set_grad_enabled(True)
    print("build_model_input:")
    test_build_model_input()

    print("forward/backward smoke tests:")
    for input_mode, mamba_mode in itertools.product(cfg.VALID_INPUT_MODES, cfg.VALID_MAMBA_MODES):
        for (h, w) in SIZES:
            run_case(input_mode, mamba_mode, h, w)

    print("\nAll shape tests passed.")


if __name__ == "__main__":
    main()
