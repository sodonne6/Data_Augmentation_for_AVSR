import argparse, csv, re, subprocess
from pathlib import Path
from typing import List, Tuple, Optional

RE_START = re.compile(r"silence_start:\s*([0-9.]+)")
RE_END   = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

SPEAKER_TYPE = ["lipspeakers", "volunteers"]

VOLUNTEER_NUM = [
    "01M",
    "02M","03F","04M","05F","06M","07F","08F","09F","10M","11F","12M",
    "13F","14M","15F","16M","17F","18M","19M","20M","21M","22M","23M","24M",
    "25M","26M","27M","28M","29M","30F","31F","32F","33F","34M","35M","36F",
    "37F","38F","39M","40F","41M","42M","43F","44F","45F","46F","47M","48M",
    "49F","50F","51F","52M","53M","54M","55F","56M","57M","58F","59F",
]

LIPSPKR_NUM = ["Lipspkr1","Lipspkr2","Lipspkr3"]
CAM_ANGLE = ["straightcam"]  # you can extend later if needed


def run(cmd: List[str]) -> Tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return p.returncode, p.stdout

def ffprobe_dur_any(path: Path) -> float:
    rc, out = run([
        "ffprobe","-v","error",
        "-show_entries","format=duration",
        "-of","default=nw=1:nk=1",
        str(path)
    ])
    if rc != 0 or not out.strip():
        return float("nan")
    return float(out.strip())

def silencedetect_log(wav: Path, noise_db: float, min_dur: float) -> str:
    rc, out = run([
        "ffmpeg","-hide_banner","-nostats","-i", str(wav),
        "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f","null","-"
    ])
    return out

def parse_silence_segments(log: str, dur: float) -> List[Tuple[float, float]]:
    starts = [float(m.group(1)) for m in RE_START.finditer(log)]
    ends   = [float(m.group(1)) for m in RE_END.finditer(log)]

    segs: List[Tuple[float,float]] = []
    j = 0
    current_start: Optional[float] = None

    for s in starts:
        if current_start is not None:
            segs.append((current_start, s))
        current_start = s

        while j < len(ends) and ends[j] < current_start:
            j += 1
        if j < len(ends):
            e = ends[j]
            if e >= current_start:
                segs.append((current_start, e))
                current_start = None
                j += 1

    # trailing open segment (ffmpeg often omits final silence_end)
    if current_start is not None and dur == dur:
        segs.append((current_start, dur))

    clean = []
    for s,e in segs:
        s = max(0.0, s)
        e = max(s, e)
        clean.append((s,e))
    clean.sort(key=lambda x: x[0])
    return clean

def compute_trim_from_segments(segs: List[Tuple[float,float]], dur: float, eps: float=0.03) -> Tuple[float,float,float,float]:
    lead = 0.0
    trail = 0.0
    speech_start = 0.0
    speech_end = dur

    if dur != dur:  # NaN
        return lead, trail, speech_start, speech_end

    if segs:
        s0,e0 = segs[0]
        if abs(s0 - 0.0) <= eps:
            lead = e0
            speech_start = e0

        sl,el = segs[-1]
        if abs(el - dur) <= eps:
            trail = dur - sl
            speech_end = sl

    speech_start = max(0.0, min(speech_start, dur))
    speech_end   = max(speech_start, min(speech_end, dur))
    return lead, trail, speech_start, speech_end

def trim_wav(in_wav: Path, out_wav: Path, start_s: float, end_s: float, overwrite: bool) -> None:
    dur = max(0.0, end_s - start_s)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    rc, out = run([
        "ffmpeg","-hide_banner","-loglevel","error",
        "-y" if overwrite else "-n",
        "-ss", f"{start_s:.6f}",
        "-i", str(in_wav),
        "-t", f"{dur:.6f}",
        "-acodec","pcm_s16le","-ar","16000","-ac","1",
        str(out_wav)
    ])
    if rc != 0:
        raise RuntimeError(out)

def trim_video(in_mp4: Path, out_mp4: Path, start_s: float, end_s: float, reencode: bool=True, overwrite: bool=True) -> None:
    dur = max(0.0, end_s - start_s)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Re-encode is safest for exact cuts. Copy can be fast but often cuts at keyframes.
    if reencode:
        cmd = [
            "ffmpeg","-hide_banner","-loglevel","error",
            "-y" if overwrite else "-n",
            "-ss", f"{start_s:.6f}",
            "-i", str(in_mp4),
            "-t", f"{dur:.6f}",
            "-an",
            "-c:v","libx264","-preset","veryfast","-crf","18",
            "-movflags","+faststart",
            str(out_mp4)
        ]
    else:
        cmd = [
            "ffmpeg","-hide_banner","-loglevel","error",
            "-y" if overwrite else "-n",
            "-ss", f"{start_s:.6f}",
            "-i", str(in_mp4),
            "-t", f"{dur:.6f}",
            "-an",
            "-c:v","copy",
            str(out_mp4)
        ]

    rc, out = run(cmd)
    if rc != 0:
        raise RuntimeError(out)

def process_one_speaker(
    in_wav_dir: Path,
    in_vid_dir: Path,
    out_wav_dir: Path,
    out_vid_dir: Path,
    out_csv: Path,
    noise_db: float,
    min_sil_dur: float,
    pad_start: float,
    pad_end: float,
    reencode_video: bool,
    overwrite: bool,
):
    wavs = sorted(in_wav_dir.glob("*.wav"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_wav_dir.mkdir(parents=True, exist_ok=True)
    out_vid_dir.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "wav","vid","wav_dur_s","vid_dur_s",
            "lead_sil_s","trail_sil_s",
            "speech_start_s","speech_end_s",
            "trim_start_s","trim_end_s","trim_dur_s",
            "num_silence_segments",
            "out_wav","out_vid","status"
        ])

        for wav in wavs:
            vid = in_vid_dir / (wav.stem + ".mp4")
            status = "ok"

            if not vid.exists():
                w.writerow([str(wav), str(vid), "", "", "", "", "", "", "", "", "", "", "", "", "missing_video"])
                continue

            wav_dur = ffprobe_dur_any(wav)
            vid_dur = ffprobe_dur_any(vid)

            log = silencedetect_log(wav, noise_db, min_sil_dur)
            segs = parse_silence_segments(log, wav_dur)
            lead, trail, speech_start, speech_end = compute_trim_from_segments(segs, wav_dur)

            trim_start = max(0.0, speech_start - pad_start)
            trim_end   = min(wav_dur, speech_end + pad_end)
            if trim_end < trim_start:
                trim_end = trim_start

            out_wav = out_wav_dir / wav.name
            out_vid = out_vid_dir / vid.name

            try:
                # no overwrite: if either exists, skip
                if (out_wav.exists() or out_vid.exists()) and not overwrite:
                    status = "skipped_exists_partial"
                else:
                    trim_wav(wav, out_wav, trim_start, trim_end, overwrite=overwrite)
                    trim_video(vid, out_vid, trim_start, trim_end, reencode=reencode_video, overwrite=overwrite)
            except Exception as e:
                status = f"error:{type(e).__name__}"

            w.writerow([
                str(wav), str(vid),
                f"{wav_dur:.6f}", f"{vid_dur:.6f}",
                f"{lead:.6f}", f"{trail:.6f}",
                f"{speech_start:.6f}", f"{speech_end:.6f}",
                f"{trim_start:.6f}", f"{trim_end:.6f}", f"{(trim_end-trim_start):.6f}",
                str(len(segs)),
                str(out_wav), str(out_vid), status
            ])

    print(f"[DONE] speaker processed: {in_wav_dir.parent.parent.parent}  (files={len(wavs)})")
    print(f"[DONE] wrote CSV: {out_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, type=Path, help="Root containing TCD_TIMIT")
    ap.add_argument("--dst_root", required=True, type=Path, help="Root for TCD_TIMIT_trimmed")

    ap.add_argument("--noise_db", type=float, default=-35.0)
    ap.add_argument("--min_sil_dur", type=float, default=0.20)
    ap.add_argument("--pad_start", type=float, default=0.11)
    ap.add_argument("--pad_end", type=float, default=0.30)
    ap.add_argument("--reencode_video", action="store_true")
    ap.add_argument("--only", choices=["volunteers", "lipspeakers", "both"], default="both")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for cam in CAM_ANGLE:
        # volunteers
        if args.only in ("volunteers", "both"):
            for vol in VOLUNTEER_NUM:
                in_wav = args.src_root / "volunteers" / vol / "Clips" / "processed" / "audio16k_norm_1" / cam
                in_vid = args.src_root / "volunteers" / vol / "Clips" / "processed" / "video25crop_alignmouth" / cam

                out_wav = args.dst_root / "volunteers" / vol / "Clips" / "processed" / "audio" / cam
                out_vid = args.dst_root / "volunteers" / vol / "Clips" / "processed" / "video" / cam
                out_csv = args.dst_root / "volunteers" / vol / "Clips" / "processed" / "trim_audit.csv"

                if not in_wav.exists():
                    print(f"[SKIP] missing audio dir: {in_wav}")
                    continue
                if not in_vid.exists():
                    print(f"[SKIP] missing video dir: {in_vid}")
                    continue

                process_one_speaker(
                    in_wav, in_vid, out_wav, out_vid, out_csv,
                    args.noise_db, args.min_sil_dur, args.pad_start, args.pad_end,
                    args.reencode_video, args.overwrite
                )

        # lipspeakers
        if args.only in ("lipspeakers", "both"):
            for lip in LIPSPKR_NUM:
                in_wav = args.src_root / "lipspeakers" / lip / "Clips" / "processed" / "audio16k_norm_1" / cam
                in_vid = args.src_root / "lipspeakers" / lip / "Clips" / "processed" / "video25crop_alignmouth" / cam

                out_wav = args.dst_root / "lipspeakers" / lip / "Clips" / "processed" / "audio" / cam
                out_vid = args.dst_root / "lipspeakers" / lip / "Clips" / "processed" / "video" / cam
                out_csv = args.dst_root / "lipspeakers" / lip / "Clips" / "processed" / "trim_audit.csv"

                if not in_wav.exists():
                    print(f"[SKIP] missing audio dir: {in_wav}")
                    continue
                if not in_vid.exists():
                    print(f"[SKIP] missing video dir: {in_vid}")
                    continue

                process_one_speaker(
                    in_wav, in_vid, out_wav, out_vid, out_csv,
                    args.noise_db, args.min_sil_dur, args.pad_start, args.pad_end,
                    args.reencode_video, args.overwrite
                )

    print("[ALL DONE] dataset pass finished.")

if __name__ == "__main__":
    main()
