"""Run several SMALL pilot trainings sequentially and summarize them.

Purpose: screen many configurations cheaply (few patients, few iterations)
before spending a full paper-protocol run on the winner. All pilots share the
same seed, the same deterministic patient subset and the same budget, so
their RANKING is meaningful. The absolute numbers are NOT reportable.

Usage
-----
    python run_pilot.py --data-dir dataset \\
        --train-patients 8 --val-patients 4 \\
        --iters 8000 --val-every 1000

Custom config list (one per line: "name | extra train_20p.py flags"):
    python run_pilot.py --data-dir dataset --configs my_pilots.txt

Default configs screened when --configs is not given:
    C0-pilot   |                                    (pure RED-CNN)
    S1-pilot   | --use-spectral-head
    S2-pilot   | --use-spectral-head --nps-weight 0.05
    S2h-pilot  | --use-spectral-head --nps-weight 0.05 --hu-bin-loss 0.1

Each pilot trains into <output-root>/<name>/<arch>/best_model.pt. Existing
results are skipped (delete the folder or pass --force to retrain). At the
end a ranked summary table is printed and saved to
<output-root>/pilot_summary.csv.
"""

import argparse
import csv
import os
import shlex
import subprocess
import sys

import torch

PAPER_LR = "9.583417460320728e-05"

DEFAULT_CONFIGS = """\
# name | extra train_20p.py flags
C0-pilot   |
S1-pilot   | --use-spectral-head
S2-pilot   | --use-spectral-head --nps-weight 0.05
S2h-pilot  | --use-spectral-head --nps-weight 0.05 --hu-bin-loss 0.1
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--arch", default="redcnn",
                   choices=["redcnn", "resnet", "local_residual"])
    p.add_argument("--split", choices=["20p", "100p"], default="100p")
    p.add_argument("--train-patients", type=int, default=8)
    p.add_argument("--val-patients", type=int, default=4)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=73)
    p.add_argument("--patch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--cache-rate", type=float, default=1.0)
    p.add_argument("--output-root", default="runs_pilot")
    p.add_argument("--configs", default=None,
                   help="Path to a config list file. Each non-comment line: "
                        "'name | extra flags'. Defaults to the built-in "
                        "C0/S1/S2/S2h list.")
    p.add_argument("--force", action="store_true",
                   help="Retrain even if best_model.pt already exists.")
    return p.parse_args()


def load_configs(path):
    text = DEFAULT_CONFIGS if path is None else open(path).read()
    configs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, extra = line.split("|", 1)
        else:
            name, extra = line, ""
        name = name.strip()
        if not name:
            continue
        configs.append((name, extra.strip()))
    if not configs:
        raise RuntimeError("No configurations found")
    return configs


def main():
    args = parse_args()
    configs = load_configs(args.configs)
    env = {**os.environ, "HU_RANGE_PRESET": "benchmark"}
    os.makedirs(args.output_root, exist_ok=True)

    print(f"\nPILOT SWEEP: {len(configs)} configs | "
          f"{args.train_patients} train / {args.val_patients} val patients | "
          f"{args.iters} iterations each")
    print("NOTE: pilot numbers rank configs; they are NOT reportable results.\n")

    for name, extra in configs:
        out_root = os.path.join(args.output_root, name)
        best = os.path.join(out_root, args.arch, "best_model.pt")
        if os.path.exists(best) and not args.force:
            print(f"[{name}] already trained -> skipping (use --force to redo)")
            continue
        cmd = [
            sys.executable, "train_20p.py",
            "--arch", args.arch,
            "--data-dir", args.data_dir,
            "--split", args.split,
            "--output-root", out_root,
            "--max-iterations", str(args.iters),
            "--iterations-before-val", str(args.val_every),
            "--batch-size", str(args.batch_size),
            "--patch-size", str(args.patch_size),
            "--val-patch-size", "128",
            "--lr", PAPER_LR,
            "--lr-schedule", "constant",
            "--select-by", "bench_ssim",
            "--num-workers", str(args.num_workers),
            "--cache-rate", str(args.cache_rate),
            "--train-patients", str(args.train_patients),
            "--val-patients", str(args.val_patients),
        ] + shlex.split(extra)
        print(f"\n[{name}] {' '.join(cmd)}\n")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"[{name}] FAILED (returncode {result.returncode}) -- "
                  f"continuing with the next config")

    # ── Summary ───────────────────────────────────────────────────────────
    rows = []
    for name, extra in configs:
        best = os.path.join(args.output_root, name, args.arch, "best_model.pt")
        if not os.path.exists(best):
            print(f"[{name}] no best_model.pt -> excluded from summary")
            continue
        payload = torch.load(best, map_location="cpu", weights_only=False)
        val = payload.get("val_detail", {})
        rows.append({
            "name": name,
            "flags": extra,
            "iter": payload.get("iteration", ""),
            "bench_ssim": val.get("bench_ssim", payload.get("score", 0.0)),
            "psnr": val.get("psnr", 0.0),
            "ssim": val.get("ssim", 0.0),
            "rmse": val.get("rmse", 0.0),
            "chest_ssim": val.get("chest_ssim", 0.0),
            "abdomen_ssim": val.get("abdomen_ssim", 0.0),
        })
    if not rows:
        print("\nNo pilot results to summarize.")
        return
    rows.sort(key=lambda r: r["bench_ssim"], reverse=True)

    print("\n" + "=" * 100)
    print("PILOT SUMMARY (ranked by bench_ssim; ranking only -- "
          "NOT reportable numbers)")
    print("=" * 100)
    hdr = (f"{'config':<12} {'iter':>6} {'bench_ssim':>11} {'PSNR':>8} "
           f"{'SSIM':>8} {'RMSE':>8} {'C-SSIM':>8} {'A-SSIM':>8}  flags")
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<12} {r['iter']:>6} {r['bench_ssim']:>11.5f} "
              f"{r['psnr']:>8.3f} {r['ssim']:>8.5f} {r['rmse']:>8.2f} "
              f"{r['chest_ssim']:>8.5f} {r['abdomen_ssim']:>8.5f}  {r['flags']}")

    csv_path = os.path.join(args.output_root, "pilot_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSummary -> {csv_path}")
    print("Next: retrain the winning config on the FULL split with the "
          "paper budget before reporting any numbers.")


if __name__ == "__main__":
    main()
