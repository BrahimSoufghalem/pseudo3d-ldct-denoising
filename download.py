"""
LDCT-and-projection-data Downloader
======================================
Downloads DICOM data from NBIA with:
  - 30 patients → dataset/ (15 Chest + 15 Abdomen)
  -  6 patients → test/    (3 Chest  + 3 Abdomen)
  - Sorted smallest → largest by total Full+Low size
  - Parallel fast downloads with resume support 
  - tqdm progress bars and CSV report
"""

import json
import time
import zipfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from nbiatoolkit import NBIAClient
from tqdm import tqdm

from config import (
    DATA_DIR, TEST_DIR,
    DATASET_CHEST_LIMIT, DATASET_ABDO_LIMIT,
    TEST_CHEST_LIMIT, TEST_ABDO_LIMIT,
    DOWNLOAD_WORKERS, COLLECTION, DOWNLOAD_TIMEOUT,
    CHUNK_SIZE, NBIA_API_URL,
)


# ═══════════════════════════════════════════
# SESSION (thread-local)
# ═══════════════════════════════════════════
def make_session():
    """Create a requests Session with retry logic."""
    s = requests.Session()
    retry = Retry(
        total=5, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
    return s


_tl = threading.local()


def get_session():
    if not hasattr(_tl, "s"):
        _tl.s = make_session()
    return _tl.s


def log(msg):
    tqdm.write(msg)


# ═══════════════════════════════════════════
# STEP 1 — LOAD SERIES + SIZE INFO
# ═══════════════════════════════════════════
def fetch_series_info():
    """
    Query the NBIA API and return a dict mapping PatientID → {Full/Low dose info}.
    """
    print("🔍  Fetching series list from NBIA …", flush=True)

    with NBIAClient() as client:
        all_series = client.getSeries(Collection=COLLECTION)

    print(f"📋  Total series received: {len(all_series)}", flush=True)

    patient_map = {}
    for s in all_series:
        desc = (s.get("SeriesDescription") or "").lower()
        pid = (s.get("PatientID") or "").strip()
        if not pid:
            continue

        if "full dose images" in desc:
            dose = "Full"
        elif "low dose images" in desc:
            dose = "Low"
        else:
            continue

        raw_size = s.get("FileSize") or s.get("TotalSizeInBytes") or 0
        try:
            size_mb = float(raw_size) / (1024 * 1024)
        except Exception:
            size_mb = 0.0

        patient_map.setdefault(pid, {})
        if dose not in patient_map[pid]:
            patient_map[pid][dose] = {"uid": s["SeriesInstanceUID"], "size_mb": size_mb}

    return patient_map


# ═══════════════════════════════════════════
# STEP 2 — FILTER, SORT & DISTRIBUTE
# ═══════════════════════════════════════════
def select_patients(patient_map):
    """
    Filter patients that have both Full+Low dose, sort by size,
    and split into dataset / test groups.

    Returns:
        (complete, selected_pids, pid_to_dest_folder)
    """
    complete = {
        pid: data for pid, data in patient_map.items()
        if "Full" in data and "Low" in data
    }
    print(f"✅  Patients with Full+Low: {len(complete)}", flush=True)

    def total_size(pid):
        d = complete[pid]
        return d["Full"]["size_mb"] + d["Low"]["size_mb"]

    sorted_chest = sorted([p for p in complete if p[0].upper() == "C"], key=total_size)
    sorted_abdomen = sorted([p for p in complete if p[0].upper() == "L"], key=total_size)

    dataset_pids = sorted_chest[:DATASET_CHEST_LIMIT] + sorted_abdomen[:DATASET_ABDO_LIMIT]
    test_pids = (
        sorted_chest[DATASET_CHEST_LIMIT: DATASET_CHEST_LIMIT + TEST_CHEST_LIMIT]
        + sorted_abdomen[DATASET_ABDO_LIMIT: DATASET_ABDO_LIMIT + TEST_ABDO_LIMIT]
    )

    pid_to_dest_folder = {}
    for p in dataset_pids:
        pid_to_dest_folder[p] = Path(DATA_DIR)
    for p in test_pids:
        pid_to_dest_folder[p] = Path(TEST_DIR)

    selected_pids = dataset_pids + test_pids
    total_est_mb = sum(total_size(p) for p in selected_pids)

    print(f"\n📊 Total Target: {len(selected_pids)} patients (~{total_est_mb:.0f} MB estimated)")
    print(f"   📂 'dataset' folder -> {len(dataset_pids)} patients (15 Chest, 15 Abdomen)")
    print(f"   📂 'test'    folder -> {len(test_pids)} patients (3 Chest, 3 Abdomen)\n", flush=True)

    return complete, selected_pids, pid_to_dest_folder


# ═══════════════════════════════════════════
# STEP 3 — RESUME DETECTION
# ═══════════════════════════════════════════
def is_series_complete(folder: Path) -> bool:
    if not folder.exists():
        return False
    if (folder / "data.zip").exists():
        return False
    return len(list(folder.rglob("*.dcm"))) > 0


def check_resume(selected_pids, pid_to_dest_folder):
    """
    Check disk for already-downloaded patients.

    Returns:
        (already_done, to_download, prev_results)
    """
    def patient_disk_status(pid):
        target_dir = pid_to_dest_folder[pid]
        p = target_dir / pid
        ok_f = is_series_complete(p / "Full_Dose")
        ok_l = is_series_complete(p / "Low_Dose")
        if ok_f and ok_l:
            return "done"
        if ok_f or ok_l:
            return "partial"
        return "none"

    already_done = [p for p in selected_pids if patient_disk_status(p) == "done"]
    partial_done = [p for p in selected_pids if patient_disk_status(p) == "partial"]
    need_download = [p for p in selected_pids if patient_disk_status(p) == "none"]

    print("📊  Resume check:")
    print(f"  ✅  Already complete : {len(already_done)}")
    print(f"  🔄  Partial (retry)  : {len(partial_done)}")
    print(f"  ⬇   Not started      : {len(need_download)}\n", flush=True)

    to_download = partial_done + need_download

    # Load previous progress
    prev_results = []
    if Path("progress.json").exists():
        try:
            with open("progress.json") as f:
                prev_results = json.load(f)
            done_set = set(already_done)
            prev_results = [r for r in prev_results if r["PatientID"] in done_set]
            print(f"  📂  Loaded {len(prev_results)} previous results\n", flush=True)
        except Exception:
            prev_results = []

    return already_done, to_download, prev_results


# ═══════════════════════════════════════════
# STEP 4 — DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════
def download_series(uid, folder, label, global_bar):
    """Download and extract a single DICOM series ZIP from NBIA."""
    folder.mkdir(parents=True, exist_ok=True)
    zip_path = folder / "data.zip"
    session = get_session()

    for attempt in range(6):
        try:
            resp = session.get(
                NBIA_API_URL,
                params={"SeriesInstanceUID": uid, "format": "zip"},
                stream=True,
                timeout=DOWNLOAD_TIMEOUT,
            )
            resp.raise_for_status()

            total_bytes = 0
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        n = len(chunk)
                        total_bytes += n
                        global_bar.update(n)

            if total_bytes < 1000:
                raise ValueError(f"Too small ({total_bytes} B)")

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(folder)

            zip_path.unlink(missing_ok=True)
            return True, total_bytes / (1024 * 1024)

        except zipfile.BadZipFile:
            log(f"  ⚠  Bad zip {uid[:20]} (try {attempt + 1})")
            zip_path.unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            log(f"  ⚠  Error {uid[:20]} (try {attempt + 1}): {str(e)[:80]}")
            zip_path.unlink(missing_ok=True)
            time.sleep(3 * (attempt + 1))

    return False, 0.0


def download_patient(pid, complete, pid_to_dest_folder, global_bar):
    """Download both Full Dose and Low Dose for a single patient."""
    data = complete[pid]
    target_dir = pid_to_dest_folder[pid]
    p_dir = target_dir / pid
    full_folder = p_dir / "Full_Dose"
    low_folder = p_dir / "Low_Dose"
    full_already = is_series_complete(full_folder)
    low_already = is_series_complete(low_folder)
    ptype = "Chest" if pid[0].upper() == "C" else "Abdomen"
    est_mb = data["Full"]["size_mb"] + data["Low"]["size_mb"]

    if full_already and low_already:
        actual_mb = sum(f.stat().st_size for f in p_dir.rglob("*.dcm")) / (1024 * 1024)
        log(f"⏭  [{pid}]  skipped (Folder: {target_dir.name})")
        return {
            "PatientID": pid, "Type": ptype, "Destination": target_dir.name,
            "Estimated_MB": round(est_mb, 2), "Downloaded_MB": round(actual_mb, 2),
            "Full_Dose_MB": "skipped", "Low_Dose_MB": "skipped",
            "Avg_Speed_MBps": "-", "status": "success",
            "full_ok": True, "low_ok": True,
        }

    log(f"⬇  [{pid}]  (~{est_mb:.0f} MB) [{ptype}] -> Saving to: {target_dir.name}")
    t0 = time.time()

    if full_already:
        ok_full = True
        mb_full = sum(f.stat().st_size for f in full_folder.rglob("*.dcm")) / (1024 * 1024)
    else:
        ok_full, mb_full = download_series(data["Full"]["uid"], full_folder, f"{pid}/Full", global_bar)

    if low_already:
        ok_low = True
        mb_low = sum(f.stat().st_size for f in low_folder.rglob("*.dcm")) / (1024 * 1024)
    else:
        ok_low, mb_low = download_series(data["Low"]["uid"], low_folder, f"{pid}/Low", global_bar)

    elapsed = max(time.time() - t0, 0.1)
    total_dl = mb_full + mb_low
    speed = total_dl / elapsed

    status = "success" if (ok_full and ok_low) else ("partial" if (ok_full or ok_low) else "failed")
    icon = {"success": "✅", "partial": "🔄", "failed": "❌"}[status]
    log(f"{icon}  [{pid}]  {status}  {total_dl:.1f} MB  @ {speed:.2f} MB/s ({target_dir.name})")

    return {
        "PatientID": pid, "Type": ptype, "Destination": target_dir.name,
        "Estimated_MB": round(est_mb, 2), "Downloaded_MB": round(total_dl, 2),
        "Full_Dose_MB": round(mb_full, 2), "Low_Dose_MB": round(mb_low, 2),
        "Avg_Speed_MBps": round(speed, 2), "status": status,
        "full_ok": ok_full, "low_ok": ok_low,
    }


# ═══════════════════════════════════════════
# STEP 5 — PARALLEL DOWNLOAD + REPORT
# ═══════════════════════════════════════════
def run_downloads(to_download, complete, pid_to_dest_folder, prev_results):
    """Execute parallel downloads and generate a CSV report."""
    def _total_size(pid):
        d = complete[pid]
        return d["Full"]["size_mb"] + d["Low"]["size_mb"]

    remaining_mb = sum(_total_size(p) for p in to_download)
    print(f"🚀  Will download {len(to_download)} patients  (~{remaining_mb:.0f} MB)\n", flush=True)

    global_bar = tqdm(
        total=int(remaining_mb * 1024 * 1024),
        unit="B", unit_scale=True, unit_divisor=1024,
        desc="📥 Dataset", colour="cyan", leave=True, dynamic_ncols=True,
    )

    results = list(prev_results)
    start = time.time()

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(download_patient, pid, complete, pid_to_dest_folder, global_bar): pid
            for pid in to_download
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            with open("progress.json", "w") as f:
                json.dump(results, f, indent=2)

    global_bar.close()

    # ── CSV Report ──
    df = pd.DataFrame(results)
    cols = [
        "PatientID", "Type", "Destination", "Estimated_MB", "Downloaded_MB",
        "Full_Dose_MB", "Low_Dose_MB", "Avg_Speed_MBps", "status", "full_ok", "low_ok",
    ]
    df = df.reindex(columns=cols)
    df = df.sort_values(["Destination", "Type", "Downloaded_MB"])
    df.to_csv("download_report.csv", index=False)

    # ── Summary ──
    total_time = time.time() - start
    overall_mb = df["Downloaded_MB"].apply(lambda x: x if isinstance(x, (int, float)) else 0).sum()
    overall_speed = overall_mb / max(total_time, 1)

    print("\n" + "=" * 55)
    print(f"🎉  DONE in {total_time / 60:.1f} minutes")
    print(f"⚡  Average speed : {overall_speed:.2f} MB/s")
    print(f"📦  Total downloaded : {overall_mb:.0f} MB")
    print(f"📊  Report  : download_report.csv\n")

    print("📁 Summary by Destination Folder:")
    for folder in ["dataset", "test"]:
        sub_df = df[df["Destination"] == folder]
        print(f"  📂  Folder '{folder}': {len(sub_df)} patients downloaded successfully.")


# ═══════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════
def main():
    Path(DATA_DIR).mkdir(exist_ok=True)
    Path(TEST_DIR).mkdir(exist_ok=True)

    patient_map = fetch_series_info()
    complete, selected_pids, pid_to_dest_folder = select_patients(patient_map)

    if not selected_pids:
        print("❌  No patients found matching criteria. Exiting.")
        return

    already_done, to_download, prev_results = check_resume(selected_pids, pid_to_dest_folder)

    if not to_download:
        print("🎉  All patients already downloaded!")
        return

    run_downloads(to_download, complete, pid_to_dest_folder, prev_results)


if __name__ == "__main__":
    main()
