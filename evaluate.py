"""
LDCT Project - Evaluation Script (full resolution, ldct-benchmark physics)
=========================================================================
Runs a trained model over the `test/` folder at FULL original resolution
(no cropping, no padding loss - the model pads internally and crops back).

Metrics (ldct-benchmark standard):
- RMSE       : physical HU, clipped to [0, A_MAX + 1024] (see config).
- PSNR/SSIM  : on clinical diagnostic windows (lung for chest, soft tissue for abdomen).
- VIF        : on the physical HU scale.

Every metric is reported THREE ways: for the denoised output, for the raw LDCT
input (the baseline), and as the delta between them. The baseline columns exist
because VIF counts noise as information, so a denoiser can gain 9 dB of PSNR
while losing VIF against its own input. A negative Delta_VIF means the model is
destroying visual information, which is a different disease from simply being
weak, and it needs a different cure.

OPERATING-POINT SWEEP (--blend)
-------------------------------
`--blend a1,a2,...` evaluates alpha * prediction + (1 - alpha) * ldct_input for
each alpha, reusing a single forward pass and a single baseline pass per slice.
This measures the model's own PSNR/VIF trade-off curve without retraining. If
VIF peaks below alpha = 1 the model is over-smoothing, which is a very
different diagnosis from being capacity- or architecture-limited.

Usage:
    python evaluate.py --input-mode 2.5d --mamba-mode full
    python evaluate.py --model runs/25d_full/best_model.pt --save-images
    python evaluate.py --model runs_charb/2d_basic/best_model.pt --no-amp \\
        --blend 0.7,0.8,0.9,1.0 --output eval_blend
"""

import argparse
from pathlib import Path
from glob import glob

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import config as cfg
from config import TEST_DIR, EVAL_OUTPUT_DIR, A_MIN, A_MAX
from utils import (
    setup_reproducibility, get_device, sort_by_instance_number,
    build_model_input, extract_centre_slice, load_state_into,
)
from model import build_model
from metrics import (
    compute_psnr_windowed, compute_ssim_windowed,
    compute_rmse_hu, compute_vif_hu,
    denormalize_to_hu_offset, apply_center_width, CW
)

import pydicom


# ═══════════════════════════════════════════
# DICOM LOADER & NORMALIZATION
# ═══════════════════════════════════════════
def load_dicom_tensor(path):
    """Read one DICOM file and return a float32 tensor in HU."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    return torch.from_numpy(arr)


def normalize(tensor, a_min=A_MIN, a_max=A_MAX):
    """Clip and scale HU -> [0, 1]."""
    return (tensor.clamp(a_min, a_max) - a_min) / (a_max - a_min)


def parse_blends(raw):
    """Parse the --blend argument into a validated, ordered list of alphas."""
    values = []
    for part in str(raw).replace(" ", "").split(","):
        if not part:
            continue
        try:
            a = float(part)
        except ValueError:
            raise ValueError(f"--blend expects comma-separated floats, got '{raw}'") from None
        if not 0.0 <= a <= 1.0:
            raise ValueError(f"--blend values must lie in [0, 1], got {a}")
        if a not in values:
            values.append(a)
    if not values:
        raise ValueError("--blend needs at least one value")
    return values


# ═══════════════════════════════════════════
# PATIENT EVALUATION
# ═══════════════════════════════════════════
@torch.no_grad()
def evaluate_patient(pid, patient_dir, model, device, input_mode="2.5d",
                     save_images=False, output_dir=None, use_amp=True,
                     blends=(1.0,)):
    """Evaluate every slice of one patient at full resolution.

    Each metric is computed twice per slice: once for the prediction and once
    for the unprocessed LDCT centre slice. The baseline pass is what makes
    Delta_VIF available, and Delta_VIF is the number that decides whether the
    remaining gap is a capacity problem or an objective-function problem.

    When several `blends` are given, the model runs ONCE per slice and the
    prediction is mixed back towards the input at each alpha, so the sweep is
    almost free relative to a single evaluation.

    Returns one result row per alpha.
    """
    low_dir = patient_dir / "Low_Dose"
    full_dir = patient_dir / "Full_Dose"

    low_imgs = sort_by_instance_number(glob(str(low_dir / "*.dcm")))
    full_imgs = sort_by_instance_number(glob(str(full_dir / "*.dcm")))

    assert len(low_imgs) == len(full_imgs), \
        f"[{pid}] Mismatch: {len(low_imgs)} low vs {len(full_imgs)} full"

    n = len(low_imgs)
    body_type = "Chest" if pid[0].upper() == "C" else "Abdomen"
    blends = list(blends)

    # Per-alpha accumulators; the baseline is shared across alphas.
    scores = {a: {"psnr": [], "ssim": [], "rmse": [], "vif": []} for a in blends}
    base_psnr_scores, base_ssim_scores, base_rmse_scores, base_vif_scores = [], [], [], []

    viz_slice_idx = n // 2
    viz_triplet = None

    amp_enabled = use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    for i in tqdm(range(n), desc=f"  [{pid}]", leave=False, unit="slice"):
        prev_i = max(i - 1, 0)
        next_i = min(i + 1, n - 1)

        raw_prev = load_dicom_tensor(low_imgs[prev_i])
        raw_curr = load_dicom_tensor(low_imgs[i])
        raw_next = load_dicom_tensor(low_imgs[next_i])
        raw_full = load_dicom_tensor(full_imgs[i])

        # [1, C, H, W] with C = 1 (2d) or 3 (2.5d); HU limits come from config
        inp = build_model_input(raw_prev, raw_curr, raw_next, input_mode=input_mode).to(device)
        lbl = normalize(raw_full).unsqueeze(0).unsqueeze(0).to(device)
        mid = extract_centre_slice(inp)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            pred_res = model(inp)

        mid_f = mid.float()
        res_f = pred_res.float()

        lbl_hu_offset = denormalize_to_hu_offset(lbl.squeeze(), A_MIN, A_MAX)
        mid_hu_offset = denormalize_to_hu_offset(mid_f.squeeze(), A_MIN, A_MAX)

        # Unprocessed LDCT input vs ground truth - identical for every alpha
        base_psnr_scores.append(compute_psnr_windowed(mid_hu_offset, lbl_hu_offset, body_type))
        base_ssim_scores.append(compute_ssim_windowed(mid_hu_offset, lbl_hu_offset, body_type))
        base_rmse_scores.append(compute_rmse_hu(mid_hu_offset, lbl_hu_offset))
        base_vif_scores.append(compute_vif_hu(mid_hu_offset, lbl_hu_offset))

        for a in blends:
            # alpha * prediction + (1 - alpha) * input, which for a residual
            # model is simply a scaled residual.
            pred = torch.clamp(mid_f + a * res_f, 0.0, 1.0)
            pred_hu_offset = denormalize_to_hu_offset(pred.squeeze(), A_MIN, A_MAX)

            scores[a]["psnr"].append(compute_psnr_windowed(pred_hu_offset, lbl_hu_offset, body_type))
            scores[a]["ssim"].append(compute_ssim_windowed(pred_hu_offset, lbl_hu_offset, body_type))
            scores[a]["rmse"].append(compute_rmse_hu(pred_hu_offset, lbl_hu_offset))
            scores[a]["vif"].append(compute_vif_hu(pred_hu_offset, lbl_hu_offset))

            if i == viz_slice_idx and a == blends[0]:
                center, width = CW.get(body_type, CW["Abdomen"])
                viz_triplet = (
                    apply_center_width(mid_hu_offset, center, width),
                    apply_center_width(lbl_hu_offset, center, width),
                    apply_center_width(pred_hu_offset, center, width),
                )

    def avg(lst):
        return sum(lst) / max(len(lst), 1)

    b_psnr, b_ssim, b_rmse, b_vif = (
        avg(base_psnr_scores), avg(base_ssim_scores), avg(base_rmse_scores), avg(base_vif_scores)
    )

    results = []
    for a in blends:
        m_psnr = avg(scores[a]["psnr"])
        m_ssim = avg(scores[a]["ssim"])
        m_rmse = avg(scores[a]["rmse"])
        m_vif = avg(scores[a]["vif"])

        results.append({
            "PatientID": pid,
            "BodyType": body_type,
            "NumSlices": n,
            "Blend": a,

            "PSNR": round(m_psnr, 4),
            "Baseline_PSNR": round(b_psnr, 4),
            "Delta_PSNR": round(m_psnr - b_psnr, 4),

            "SSIM": round(m_ssim, 4),
            "Baseline_SSIM": round(b_ssim, 4),
            "Delta_SSIM": round(m_ssim - b_ssim, 4),

            "RMSE_HU": round(m_rmse, 4),
            "Baseline_RMSE_HU": round(b_rmse, 4),
            # baseline minus prediction, so positive always means "better"
            "Delta_RMSE_HU": round(b_rmse - m_rmse, 4),

            "VIF": round(m_vif, 4),
            "Baseline_VIF": round(b_vif, 4),
            "Delta_VIF": round(m_vif - b_vif, 4),
        })

    if save_images and viz_triplet is not None and output_dir is not None:
        save_patient_viz(pid, body_type, viz_triplet, results[0], output_dir)

    return results


# ═══════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════
def save_patient_viz(pid, body_type, viz_triplet, metrics, output_dir):
    """Save a side-by-side triplet: LDCT | NDCT | Denoised (clinical window)."""
    ldct, ndct, denoised = viz_triplet
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.05)

    window_name = "Lung Window (C=-600, W=1500)" if body_type == "Chest" else "Soft Tissue Window (C=50, W=400)"
    titles = [
        f"LDCT (Input)\nPSNR: {metrics['Baseline_PSNR']:.2f} dB | VIF: {metrics['Baseline_VIF']:.4f}",
        f"NDCT (Ground Truth)\n{window_name}",
        f"Denoised (Output)\nPSNR: {metrics['PSNR']:.2f} dB | SSIM: {metrics['SSIM']:.4f} | VIF: {metrics['VIF']:.4f}",
    ]

    for j, (img, title) in enumerate(zip([ldct, ndct, denoised], titles)):
        ax = fig.add_subplot(gs[j])
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(f"Patient: {pid} | Type: {body_type}", fontsize=12, fontweight="bold")
    plt.savefig(output_dir / f"{pid}_viz.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════
def print_summary(df, title=None, verdict=True):
    """Print per-patient rows, then per-region model / input / delta triplets."""
    print("\n" + "=" * 92)
    print(title or "EVALUATION RESULTS (ldct-benchmark physical standard)")
    print("=" * 92)
    print(
        f"\n{'Patient':<10} {'Type':<9} {'Slices':>6}  {'dPSNR':>7}  {'PSNR':>7}  "
        f"{'SSIM':>7}  {'RMSE(HU)':>9}  {'VIF':>7}  {'dVIF':>8}"
    )
    print("-" * 92)

    for _, row in df.iterrows():
        print(
            f"{row['PatientID']:<10} {row['BodyType']:<9} {row['NumSlices']:>6}  "
            f"{row['Delta_PSNR']:>+7.2f}  {row['PSNR']:>7.2f}  "
            f"{row['SSIM']:>7.4f}  {row['RMSE_HU']:>9.2f}  {row['VIF']:>7.4f}  {row['Delta_VIF']:>+8.4f}"
        )

    print("=" * 92)

    overall_delta_vif = None

    for body_type in ["Chest", "Abdomen", "Overall"]:
        sub = df if body_type == "Overall" else df[df["BodyType"] == body_type]
        if sub.empty:
            continue

        m = (sub["PSNR"].mean(), sub["SSIM"].mean(), sub["RMSE_HU"].mean(), sub["VIF"].mean())
        b = (sub["Baseline_PSNR"].mean(), sub["Baseline_SSIM"].mean(),
             sub["Baseline_RMSE_HU"].mean(), sub["Baseline_VIF"].mean())
        d = (m[0] - b[0], m[1] - b[1], b[2] - m[2], m[3] - b[3])

        if body_type == "Overall":
            overall_delta_vif = d[3]

        print(f"\n  {body_type}")
        print(f"    Denoised  :  PSNR {m[0]:>7.2f} dB  |  SSIM {m[1]:>7.4f}  |  "
              f"RMSE {m[2]:>7.2f} HU  |  VIF {m[3]:>7.4f}")
        print(f"    LDCT input:  PSNR {b[0]:>7.2f} dB  |  SSIM {b[1]:>7.4f}  |  "
              f"RMSE {b[2]:>7.2f} HU  |  VIF {b[3]:>7.4f}")
        print(f"    Delta     :       {d[0]:>+7.2f} dB  |       {d[1]:>+7.4f}  |  "
              f"     {d[2]:>+7.2f} HU  |      {d[3]:>+7.4f}")

    print("\n" + "=" * 92)

    if verdict and overall_delta_vif is not None:
        if overall_delta_vif < 0:
            print(
                "VERDICT: Delta_VIF is NEGATIVE. The model removes more visual information than\n"
                "         it restores. This is an objective-function problem, not a capacity one:\n"
                "         Charbonnier, SSIM and Sobel are all pointwise distances whose optimum is\n"
                "         the posterior mean, i.e. blur wherever texture and quantum noise share a\n"
                "         frequency band. Reweighting them cannot fix it. Changing the objective\n"
                "         class (adversarial / perceptual) or the input (true 2.5D) can."
            )
        else:
            print(
                "VERDICT: Delta_VIF is positive. The model adds visual information; the gap to the\n"
                "         benchmark is a matter of degree."
            )
        print("=" * 92)


def print_blend_curve(df, blends):
    """Print the PSNR/VIF trade-off curve across blend factors.

    This is the whole point of the sweep. Read the VIF columns first: if they
    peak below alpha = 1 the model is over-smoothing, and filtering strength -
    not capacity, not architecture - is the binding constraint.
    """
    print("\n" + "=" * 92)
    print("BLEND SWEEP: alpha * prediction + (1 - alpha) * LDCT input")
    print("=" * 92)
    print(
        f"\n{'alpha':>6} | {'PSNR C':>8} {'SSIM C':>8} {'VIF C':>8} | "
        f"{'PSNR A':>8} {'SSIM A':>8} {'VIF A':>8}"
    )
    print("-" * 92)

    best = {}
    for a in blends:
        sub = df[df["Blend"] == a]
        chest = sub[sub["BodyType"] == "Chest"]
        abdo = sub[sub["BodyType"] == "Abdomen"]

        row = {
            "psnr_c": chest["PSNR"].mean(), "ssim_c": chest["SSIM"].mean(),
            "vif_c": chest["VIF"].mean(),
            "psnr_a": abdo["PSNR"].mean(), "ssim_a": abdo["SSIM"].mean(),
            "vif_a": abdo["VIF"].mean(),
        }
        best[a] = row
        print(
            f"{a:>6.2f} | {row['psnr_c']:>8.2f} {row['ssim_c']:>8.4f} {row['vif_c']:>8.4f} | "
            f"{row['psnr_a']:>8.2f} {row['ssim_a']:>8.4f} {row['vif_a']:>8.4f}"
        )

    print("-" * 92)

    a_vif_c = max(blends, key=lambda a: best[a]["vif_c"])
    a_vif_a = max(blends, key=lambda a: best[a]["vif_a"])
    a_psnr_c = max(blends, key=lambda a: best[a]["psnr_c"])

    print(f"\n  VIF peaks at   alpha = {a_vif_c:.2f} (chest), {a_vif_a:.2f} (abdomen)")
    print(f"  PSNR peaks at  alpha = {a_psnr_c:.2f} (chest)")

    if max(a_vif_c, a_vif_a) < max(blends):
        print(
            "\n  READING: VIF is maximised by holding some of the input back, so the model is\n"
            "  OVER-SMOOTHING at full strength. The constraint is the operating point, not the\n"
            "  architecture. Reducing denoising strength - or training towards a lower one - is\n"
            "  a lever that costs no parameters."
        )
    else:
        print(
            "\n  READING: VIF is highest at full strength, so the model is NOT over-smoothing.\n"
            "  The over-smoothing hypothesis is dead and the deficit is genuinely a matter of\n"
            "  how much information the model recovers, not of how hard it filters."
        )
    print("=" * 92)


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Evaluate the LDCT denoising model with ldct-benchmark metrics.")
    parser.add_argument("--input-mode", default=cfg.INPUT_MODE, choices=list(cfg.VALID_INPUT_MODES))
    parser.add_argument("--mamba-mode", default=cfg.MAMBA_MODE, choices=list(cfg.VALID_MAMBA_MODES))
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained weights (.pt). Defaults to the run folder for the chosen modes.")
    parser.add_argument("--test-dir", type=str, default=TEST_DIR)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--output", type=str, default=EVAL_OUTPUT_DIR)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--blend", type=str, default="1.0",
        help="Comma-separated blend factors alpha in [0, 1]: output = alpha * prediction + "
             "(1 - alpha) * LDCT input. Several alphas share one forward pass. Default 1.0."
    )
    args = parser.parse_args()

    input_mode = cfg.normalize_input_mode(args.input_mode)
    mamba_mode = cfg.normalize_mamba_mode(args.mamba_mode)
    model_path = args.model or cfg.run_paths(mamba_mode=mamba_mode, input_mode=input_mode)["best_model"]
    blends = parse_blends(args.blend)
    sweeping = len(blends) > 1 or blends != [1.0]

    setup_reproducibility()
    device = get_device()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading model: {model_path}")
    print(f"input_mode={input_mode} | mamba_mode={mamba_mode} | HU range [{A_MIN}, {A_MAX}] "
          f"(preset '{cfg.HU_RANGE_PRESET}')")
    if sweeping:
        print(f"Blend sweep: {', '.join(f'{a:g}' for a in blends)}")
    model = build_model(device, mamba_mode=mamba_mode, input_mode=input_mode, data_parallel=False)

    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    load_state_into(model, state)      # handles "module." prefixes and wrapped dicts
    model.eval()
    print("Model loaded successfully.\n")

    test_path = Path(args.test_dir)
    patients = sorted([
        p for p in test_path.iterdir()
        if p.is_dir() and (p / "Low_Dose").exists() and (p / "Full_Dose").exists()
    ])

    if not patients:
        print(f"No valid patients found in '{args.test_dir}'. "
              "Each patient folder must contain 'Low_Dose/' and 'Full_Dose/'.")
        return

    chest_patients = [p for p in patients if p.name[0].upper() == "C"]
    abdomen_patients = [p for p in patients if p.name[0].upper() == "L"]
    print(f"Found {len(patients)} patients: {len(chest_patients)} Chest, {len(abdomen_patients)} Abdomen")
    print("Baseline metrics are computed too, so expect roughly 2x the usual runtime.\n")

    all_results = []
    for patient_dir in patients:
        pid = patient_dir.name
        print(f"Evaluating [{pid}] ({'Chest' if pid[0].upper() == 'C' else 'Abdomen'}) ...")
        try:
            all_results.extend(evaluate_patient(
                pid, patient_dir, model, device,
                input_mode=input_mode,
                save_images=args.save_images,
                output_dir=output_dir,
                use_amp=not args.no_amp,
                blends=blends,
            ))
        except Exception as e:
            print(f"  Failed: {e}")

    if not all_results:
        print("No results collected.")
        return

    df = pd.DataFrame(all_results).sort_values(["Blend", "BodyType", "PatientID"])
    suffix = "_blend" if sweeping else ""
    csv_path = output_dir / f"evaluation_report_{cfg.run_name(mamba_mode, input_mode)}{suffix}.csv"
    df.to_csv(csv_path, index=False)

    if sweeping:
        for a in blends:
            print_summary(
                df[df["Blend"] == a],
                title=f"EVALUATION RESULTS  |  blend alpha = {a:g}",
                verdict=False,
            )
        print_blend_curve(df, blends)
    else:
        print_summary(df)

    print(f"\nFull report saved -> {csv_path}")
    if args.save_images:
        print(f"Images saved      -> {output_dir}/")


if __name__ == "__main__":
    main()
