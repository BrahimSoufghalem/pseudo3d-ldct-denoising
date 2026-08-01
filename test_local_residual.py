"""CPU smoke tests for the benchmark mean/std local residual control."""

import torch
import torch.nn.functional as F

from local_residual_data import denormalize_to_pixel, standardize_hu
from local_residual_model import LocalResidualNet


def main():
    hu = torch.tensor([-1024.0, -500.0, 0.0, 1000.0, 2500.0])
    restored = denormalize_to_pixel(standardize_hu(hu)) - 1024.0
    assert torch.allclose(restored, hu, atol=1e-4)

    torch.manual_seed(0)
    model = LocalResidualNet(channels=16, blocks=2, groups=8, verbose=False)
    x = torch.randn(3, 1, 32, 40)
    y = torch.randn_like(x)
    pred = model(x)
    assert pred.shape == x.shape
    loss = F.mse_loss(pred, y)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert model.receptive_field() == 19
    print(f"Local residual mean/std smoke test passed | loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
