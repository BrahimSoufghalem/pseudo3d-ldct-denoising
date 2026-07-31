"""
Selective-scan micro-benchmark
================================
Times ONLY the S6 selective scan at the exact shapes used by the bottleneck,
so the optimal SCAN_CHUNK_SIZE can be measured instead of guessed.

The chunk size does NOT change the number of sequential timesteps (that is
always L). It only changes the size of the per-chunk temporaries
[B, K*d_inner, chunk, N]. Large chunks therefore cost extra memory traffic for
no algorithmic benefit, which is why smaller chunks are often faster.

Usage:
    python bench_scan.py                       # forward only, batch 16
    python bench_scan.py --batch 64 --backward
"""

import argparse
import time

import torch

import config as cfg
from naf_mamba_blocks import HAS_MAMBA_SSM, selective_scan


def make_inputs(b, d, n, l, g, device, requires_grad=False):
    u = torch.randn(b, d, l, device=device, requires_grad=requires_grad)
    delta = torch.randn(b, d, l, device=device, requires_grad=requires_grad)
    A = -torch.rand(d, n, device=device) - 0.5
    B = torch.randn(b, g, n, l, device=device)
    C = torch.randn(b, g, n, l, device=device)
    D = torch.ones(d, device=device)
    bias = torch.zeros(d, device=device)
    return u, delta, A, B, C, D, bias


def timeit(fn, iters=10, warmup=3, device="cuda"):
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--width", type=int, default=cfg.MODEL_WIDTH)
    p.add_argument("--spatial", type=int, default=cfg.SPATIAL_SIZE[0])
    p.add_argument("--backward", action="store_true", help="also time the backward pass")
    p.add_argument("--chunks", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Bottleneck geometry: stages are w, 2w, 4w, 8w, 16w and the SS2D block
    # expands by 2 internally.
    d_model = args.width * 16
    d_inner = 2 * d_model
    K = cfg.N_SCAN_DIRECTIONS
    d = K * d_inner
    n = cfg.D_STATE
    side = args.spatial // cfg.SIZE_DIVISOR
    l = side * side

    print(f"device={device} | official kernel available: {HAS_MAMBA_SSM}")
    print(f"batch={args.batch} d_model={d_model} d_inner={d_inner} K={K} "
          f"d={d} N={n} L={l} ({side}x{side})")
    print(f"backward: {'on' if args.backward else 'off'}\n")

    u, delta, A, B, C, D, bias = make_inputs(
        args.batch, d, n, l, K, device, requires_grad=args.backward
    )

    def run(backend, chunk):
        def step():
            y = selective_scan(u, delta, A, B, C, D=D, delta_bias=bias,
                               delta_softplus=True, backend=backend, chunk_size=chunk)
            if args.backward:
                y.sum().backward(retain_graph=True)
        return step

    print("PyTorch fallback (backend='ref'):")
    best = None
    for chunk in args.chunks:
        try:
            ms = timeit(run("ref", chunk), device=device)
        except torch.cuda.OutOfMemoryError:
            print(f"  chunk={chunk:4d}  OOM")
            torch.cuda.empty_cache()
            continue
        marker = ""
        if best is None or ms < best[1]:
            best = (chunk, ms)
            marker = "  <-- best so far"
        print(f"  chunk={chunk:4d}  {ms:8.2f} ms/iter{marker}")

    if best is not None:
        print(f"\nFastest fallback chunk: {best[0]} ({best[1]:.2f} ms/iter)")
        print(f"Set SCAN_CHUNK_SIZE = {best[0]} in config.py")

    if HAS_MAMBA_SSM and device == "cuda":
        ms = timeit(run("cuda", 0), device=device)
        print(f"\nOfficial mamba_ssm kernel: {ms:8.2f} ms/iter")
        if best is not None:
            print(f"Speedup over the best fallback: {best[1] / ms:.1f}x")
            print("SCAN_CHUNK_SIZE is IGNORED while the official kernel is active.")
    else:
        print("\nOfficial kernel not available - install it for the real speedup:")
        print("  pip install 'mamba-ssm==2.2.5' --no-build-isolation --no-deps")


if __name__ == "__main__":
    main()
