"""CPU forward/backward smoke tests for the physics-spectral research path."""

import torch

from physics_losses import PhysicsInformedCTLoss
from physics_spectral_model import PhysicsSpectralNet


def run(spectral, lambda_nps, lambda_hu):
    torch.manual_seed(0)
    model = PhysicsSpectralNet(
        channels=16, band_channels=4, groups=2,
        dilations=(1, 2), spectral=spectral, verbose=False,
    )
    x = torch.rand(3, 1, 32, 40)
    y = torch.rand_like(x)
    correction = model(x)
    assert correction.shape == x.shape
    assert correction.abs().max().item() == 0.0
    pred = x + correction
    loss_fn = PhysicsInformedCTLoss(-1024, 1900, lambda_nps, lambda_hu)
    loss, info = loss_fn(pred, y, x)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    if spectral:
        low, mid, high = model.decomposition(x)
        assert torch.allclose(low + mid + high, x, atol=1e-6)

    print(f"OK spectral={spectral} nps={lambda_nps} hu={lambda_hu} loss={loss.item():.6f}")


def main():
    for spectral in (False, True):
        for nps, hu in ((0, 0), (0.01, 0), (0, 0.1), (0.01, 0.1)):
            run(spectral, nps, hu)
    print("All physics-spectral tests passed.")


if __name__ == "__main__":
    main()
