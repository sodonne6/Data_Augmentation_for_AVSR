import re, subprocess, random, math
from pathlib import Path
import numpy as np

# ----------------- CONFIG -----------------
LRS3_ROOT = Path(r"G:\My Drive\LRS3\audio\trainval")

TCD_ROOT  = Path(r"G:\My Drive\TCD_TIMIT")
CAM = "straightcam"

VOLUNTEERS = [
    "01M","02M","03F","04M","05F","06M","07F","08F","09F","10M","11F","12M",
    "13F","14M","15F","16M","17F","18M","19M","20M","21M","22M","23M","24M",
    "25M","26M","27M","28M","29M","30F","31F","32F","33F","34M","35M","36F",
    "37F","38F","39M","40F","41M","42M","43F","44F","45F","46F","47M","48M",
    "49F","50F","51F","52M","53M","54M","55F","56M","57M","58F","59F"
]
LIPSPEAKERS = ["Lipspkr1","Lipspkr2","Lipspkr3"]

# sample sizes
N_LRS3 = 500                 #total random LRS3 wavs
N_PER_SPEAKER = 20           #per volunteer/lipspeaker per condition (cropped & norm)
SEED = 0

# which TCD folders to compare
TCD_BEFORE_NAME = "audio16k_cropped"
TCD_AFTER_NAME  = "audio16k_norm"   

# ----------------- HELPERS -----------------
random.seed(SEED)

MEAN_RE = re.compile(r"mean_volume:\s*([-\d\.]+)\s*dB")
MAX_RE  = re.compile(r"max_volume:\s*([-\d\.]+)\s*dB")

def volumedetect_db(path: Path):
    cmd = ["ffmpeg","-hide_banner","-i", str(path), "-af","volumedetect", "-f","null","NUL"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    txt = p.stderr
    m = MEAN_RE.search(txt)
    x = MAX_RE.search(txt)
    if not (m and x):
        return None
    return float(m.group(1)), float(x.group(1))

def summarize(name, arr):
    arr = np.array(arr, dtype=np.float64)
    return {
        "name": name,
        "n": int(arr.shape[0]),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

def print_summary(s, label):
    print(f"{label}: n={s['n']}  mean={s['mean']:.2f}  med={s['median']:.2f}  "
          f"p10={s['p10']:.2f}  p90={s['p90']:.2f}  min={s['min']:.2f}  max={s['max']:.2f}")

def sample_files(files, n):
    if len(files) == 0:
        return []
    if len(files) <= n:
        return files
    return random.sample(files, n)

def tcd_dir(speaker_type, speaker_id, folder_name):
    return TCD_ROOT / speaker_type / speaker_id / "Clips" / "processed" / folder_name / CAM

def collect_tcd_for_speaker(speaker_type, sid, folder_name, n):
    d = tcd_dir(speaker_type, sid, folder_name)
    if not d.exists():
        return []
    files = sorted(d.glob("*.wav"))
    return sample_files(files, n)

# ----------------- LRS3 -----------------
print("[INFO] Scanning LRS3 wav list...")
lrs3_files = list(LRS3_ROOT.rglob("*.wav"))
print(f"[INFO] LRS3 wavs found: {len(lrs3_files)}")
lrs3_pick = sample_files(lrs3_files, N_LRS3)

lrs3_mean = []
lrs3_max  = []
bad = 0
for p in lrs3_pick:
    r = volumedetect_db(p)
    if r is None:
        bad += 1
        continue
    mv, mx = r
    lrs3_mean.append(mv); lrs3_max.append(mx)

print(f"[INFO] LRS3 analyzed: {len(lrs3_mean)} (failed {bad})")

# ----------------- TCD (cropped + norm) -----------------
def collect_tcd_all(folder_name):
    all_mean = []
    all_max  = []
    per_speaker = {}
    missing_dirs = 0
    failed = 0

    # volunteers
    for sid in VOLUNTEERS:
        picks = collect_tcd_for_speaker("volunteers", sid, folder_name, N_PER_SPEAKER)
        if not picks:
            missing_dirs += 1
            continue
        mlist, xlist = [], []
        for p in picks:
            r = volumedetect_db(p)
            if r is None:
                failed += 1
                continue
            mv, mx = r
            all_mean.append(mv); all_max.append(mx)
            mlist.append(mv); xlist.append(mx)
        if mlist:
            per_speaker[f"volunteers/{sid}"] = (np.median(mlist), np.median(xlist), len(mlist))

    # lipspeakers
    for sid in LIPSPEAKERS:
        picks = collect_tcd_for_speaker("lipspeakers", sid, folder_name, N_PER_SPEAKER)
        if not picks:
            missing_dirs += 1
            continue
        mlist, xlist = [], []
        for p in picks:
            r = volumedetect_db(p)
            if r is None:
                failed += 1
                continue
            mv, mx = r
            all_mean.append(mv); all_max.append(mx)
            mlist.append(mv); xlist.append(mx)
        if mlist:
            per_speaker[f"lipspeakers/{sid}"] = (np.median(mlist), np.median(xlist), len(mlist))

    return all_mean, all_max, per_speaker, missing_dirs, failed

print("\n[INFO] Sampling TCD cropped...")
tcd_c_mean, tcd_c_max, per_c, miss_c, fail_c = collect_tcd_all(TCD_BEFORE_NAME)
print(f"[INFO] TCD cropped analyzed: {len(tcd_c_mean)}  missing_dirs={miss_c} failed={fail_c}")

print("\n[INFO] Sampling TCD norm...")
tcd_n_mean, tcd_n_max, per_n, miss_n, fail_n = collect_tcd_all(TCD_AFTER_NAME)
print(f"[INFO] TCD norm analyzed: {len(tcd_n_mean)}  missing_dirs={miss_n} failed={fail_n}")

# ----------------- Summaries -----------------
print("\n================ OVERALL (mean_volume dB) ================")
S_lrs3_m = summarize("LRS3 mean", lrs3_mean)
S_tcdc_m = summarize("TCD cropped mean", tcd_c_mean)
S_tcdn_m = summarize("TCD norm mean", tcd_n_mean)

print_summary(S_lrs3_m, "LRS3")
print_summary(S_tcdc_m, "TCD cropped")
print_summary(S_tcdn_m, "TCD norm")

print("\nGap vs LRS3 (mean_volume):")
print(f"  cropped - LRS3 (median): {S_tcdc_m['median'] - S_lrs3_m['median']:+.2f} dB")
print(f"  norm    - LRS3 (median): {S_tcdn_m['median'] - S_lrs3_m['median']:+.2f} dB")

print("\n================ OVERALL (max_volume dB) =================")
S_lrs3_x = summarize("LRS3 max", lrs3_max)
S_tcdc_x = summarize("TCD cropped max", tcd_c_max)
S_tcdn_x = summarize("TCD norm max", tcd_n_max)

print_summary(S_lrs3_x, "LRS3")
print_summary(S_tcdc_x, "TCD cropped")
print_summary(S_tcdn_x, "TCD norm")

print("\nGap vs LRS3 (max_volume):")
print(f"  cropped - LRS3 (median): {S_tcdc_x['median'] - S_lrs3_x['median']:+.2f} dB")
print(f"  norm    - LRS3 (median): {S_tcdn_x['median'] - S_lrs3_x['median']:+.2f} dB")

# ----------------- Per-speaker view (median mean_volume) -----------------
print("\n================ PER-SPEAKER (median mean_volume dB) ================")
keys = sorted(set(per_c.keys()) | set(per_n.keys()))
print("speaker\tcropped_med_mean\tnorm_med_mean\tdelta\tN")
for k in keys:
    c = per_c.get(k)
    n = per_n.get(k)
    if c and n:
        delta = n[0] - c[0]
        print(f"{k}\t{c[0]:.2f}\t{n[0]:.2f}\t{delta:+.2f}\t{n[2]}")
    elif c and not n:
        print(f"{k}\t{c[0]:.2f}\tNA\tNA\t{c[2]}")
    elif n and not c:
        print(f"{k}\tNA\t{n[0]:.2f}\tNA\t{n[2]}")
