import argparse
import csv
import math
import os
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm
import soundfile as sf


VOLUNTEERS = [
    "01M","02M","03F","04M","05F","06M","07F","08F","09F","10M","11F","12M",
    "13F","14M","15F","16M","17F","18M","19M","20M","21M","22M","23M","24M",
    "25M","26M","27M","28M","29M","30F","31F","32F","33F","34M","35M","36F",
    "37F","38F","39M","40F","41M","42M","43F","44F","45F","46F","47M","48M",
    "49F","50F","51F","52M","53M","54M","55F","56M","57M","58F","59F",
]
LIPSPEAKERS = ["Lipspkr1", "Lipspkr2", "Lipspkr3"]


def run_cmd(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def ffprobe_json(path: Path):
    # JSON gives us lots of metadata reliably.
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-print_format", "json",
        str(path)
    ]
    rc, out, err = run_cmd(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed for {path}\n{err}")
    import json
    return json.loads(out)


def summarize_video(path: Path):
    j = ffprobe_json(path)
    fmt = j.get("format", {})
    streams = j.get("streams", [])

    v0 = next((s for s in streams if s.get("codec_type") == "video"), None)
    a0 = next((s for s in streams if s.get("codec_type") == "audio"), None)

    def pick(s, keys):
        if not s:
            return {}
        return {k: s.get(k, None) for k in keys}

    v_keys = [
        "codec_name","codec_long_name","profile","level","pix_fmt",
        "width","height","coded_width","coded_height",
        "sample_aspect_ratio","display_aspect_ratio",
        "r_frame_rate","avg_frame_rate","time_base","nb_frames",
        "bit_rate","bits_per_raw_sample",
        "color_range","color_space","color_transfer","color_primaries",
        "field_order","has_b_frames","refs"
    ]
    a_keys = [
        "codec_name","codec_long_name","sample_rate","channels","channel_layout",
        "bit_rate","bits_per_sample"
    ]

    summary = {
        "path": str(path),
        "format_name": fmt.get("format_name"),
        "duration": fmt.get("duration"),
        "size": fmt.get("size"),
        "overall_bit_rate": fmt.get("bit_rate"),
        "video": pick(v0, v_keys),
        "audio": pick(a0, a_keys),
        "has_audio_stream": a0 is not None,
    }
    return summary


def print_video_comparison(lrs3_mp4: Path, tcd_mp4: Path):
    print("\n================ VIDEO FORMAT CHECK (LRS3 vs TCD) ================\n")
    print("LRS3 mp4:", lrs3_mp4)
    print("TCD  mp4:", tcd_mp4)

    l = summarize_video(lrs3_mp4)
    t = summarize_video(tcd_mp4)

    def pblock(title, d):
        print(f"\n--- {title} ---")
        print(f"format_name: {d.get('format_name')}")
        print(f"duration:    {d.get('duration')}")
        print(f"size:        {d.get('size')}")
        print(f"bit_rate:    {d.get('overall_bit_rate')}")
        print("video stream:")
        for k, v in d["video"].items():
            print(f"  {k:20s} {v}")
        if d["has_audio_stream"]:
            print("audio stream:")
            for k, v in d["audio"].items():
                print(f"  {k:20s} {v}")
        else:
            print("audio stream: (none)")

    pblock("LRS3", l)
    pblock("TCD", t)

    # highlight mismatches on the core invariants you care about for AV-HuBERT
    core = [
        ("video.width",        l["video"].get("width"),        t["video"].get("width")),
        ("video.height",       l["video"].get("height"),       t["video"].get("height")),
        ("video.pix_fmt",      l["video"].get("pix_fmt"),      t["video"].get("pix_fmt")),
        ("video.avg_frame_rate", l["video"].get("avg_frame_rate"), t["video"].get("avg_frame_rate")),
        ("video.r_frame_rate",   l["video"].get("r_frame_rate"),   t["video"].get("r_frame_rate")),
        ("video.codec_name",   l["video"].get("codec_name"),   t["video"].get("codec_name")),
        ("video.sample_aspect_ratio", l["video"].get("sample_aspect_ratio"), t["video"].get("sample_aspect_ratio")),
        ("video.display_aspect_ratio", l["video"].get("display_aspect_ratio"), t["video"].get("display_aspect_ratio")),
        ("video.color_space",  l["video"].get("color_space"),  t["video"].get("color_space")),
        ("video.color_transfer", l["video"].get("color_transfer"), t["video"].get("color_transfer")),
        ("video.color_primaries", l["video"].get("color_primaries"), t["video"].get("color_primaries")),
    ]

    mism = [(k,a,b) for (k,a,b) in core if (a is not None and b is not None and str(a) != str(b))]
    print("\n--- Core mismatch summary ---")
    if not mism:
        print("No core mismatches (width/height/fps/codec/SAR/DAR/pix_fmt/color).")
    else:
        for k,a,b in mism:
            print(f"{k}: LRS3={a}  vs  TCD={b}")

    print("\nNOTE: pix_fmt mismatch (e.g. yuv444p vs yuv420p) is *not fatal* if your decode path outputs 3-channel uint8 frames,")
    print("but it can change subtle pixel statistics. If you want them identical, transcode TCD to yuv444p (optional).")
    print("==================================================================\n")


def list_lrs3_wavs(root: Path):
    return list(root.rglob("*.wav"))


def robust_rms(x: np.ndarray):
    # x assumed float32/float64 in [-1,1]
    if x.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def normalize_one_wav(in_path: Path, out_path: Path, target_rms: float, peak_limit: float):
    # Read
    x, sr = sf.read(str(in_path), always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]  # mono
    x = x.astype(np.float32, copy=False)

    # Safety: if file is silent / near-silent, skip heavy boosting
    rms = robust_rms(x)
    if rms is None or rms < 1e-6:
        return {
            "status": "skip_silent",
            "sr": sr,
            "in_rms": rms if rms is not None else -1.0,
            "gain": 1.0,
            "peak_before": float(np.max(np.abs(x))) if x.size else 0.0,
            "peak_after": float(np.max(np.abs(x))) if x.size else 0.0,
        }

    # Desired gain to reach target RMS
    gain = target_rms / rms

    # Peak-aware cap (headroom)
    peak_before = float(np.max(np.abs(x)))
    if peak_before > 0:
        max_gain = peak_limit / peak_before
        if gain > max_gain:
            gain = max_gain

    y = x * gain
    peak_after = float(np.max(np.abs(y))) if y.size else 0.0

    # Ensure output format exactly: 16k, mono, PCM16
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), y, 16000, subtype="PCM_16")

    return {
        "status": "ok",
        "sr": sr,
        "in_rms": rms,
        "gain": float(gain),
        "peak_before": peak_before,
        "peak_after": peak_after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcd_root", required=True, type=Path, help="e.g. G:\\My Drive\\TCD_TIMIT")
    ap.add_argument("--lrs3_audio_root", required=True, type=Path, help="e.g. G:\\My Drive\\LRS3\\audio\\trainval")
    ap.add_argument("--cam", default="straightcam")
    ap.add_argument("--out_folder_name", default="audio16k_norm")
    ap.add_argument("--peak_limit", type=float, default=0.95)
    ap.add_argument("--lrs3_sample_n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--report_csv", required=True, type=Path)
    ap.add_argument("--check_lrs3_video", type=Path, default=None, help="One LRS3 mp4 path to compare")
    ap.add_argument("--check_tcd_video", type=Path, default=None, help="One TCD mp4 path to compare")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---------- VIDEO FORMAT CHECK (one pair) ----------
    if args.check_lrs3_video and args.check_tcd_video:
        print_video_comparison(args.check_lrs3_video, args.check_tcd_video)
    else:
        print("[INFO] Skipping video format check (no --check_lrs3_video/--check_tcd_video provided).")

    # ---------- Estimate target RMS from LRS3 ----------
    lrs3_wavs = list_lrs3_wavs(args.lrs3_audio_root)
    print(f"[INFO] Found {len(lrs3_wavs)} LRS3 wavs under: {args.lrs3_audio_root}")
    sample_n = min(args.lrs3_sample_n, len(lrs3_wavs))
    print(f"[INFO] Sampling {sample_n} LRS3 wavs to estimate target RMS (median)...")
    sample_paths = random.sample(lrs3_wavs, sample_n)

    rms_vals = []
    for p in tqdm(sample_paths, desc="LRS3 RMS scan", unit="file"):
        try:
            x, sr = sf.read(str(p), always_2d=False)
            if x.ndim > 1:
                x = x[:, 0]
            x = x.astype(np.float32, copy=False)
            rms = robust_rms(x)
            if rms is not None and rms > 1e-6:
                rms_vals.append(rms)
        except Exception:
            pass

    if not rms_vals:
        raise RuntimeError("No valid RMS values found from LRS3 sample.")
    target_rms = float(np.median(np.array(rms_vals)))
    print(f"[INFO] LRS3 RMS scan complete. Valid RMS samples: {len(rms_vals)}. Target RMS (median) = {target_rms:.6f}")
    print(f"[INFO] Using peak_limit = {args.peak_limit}")

    # ---------- Build list of TCD wavs ----------
    speakers = []
    for v in VOLUNTEERS:
        speakers.append(("volunteers", v))
    for l in LIPSPEAKERS:
        speakers.append(("lipspeakers", l))

    # For each speaker, we normalize from audio16k_cropped/<cam> to audio16k_norm/<cam>
    rows = []
    total_files = 0
    for spk_type, spk in speakers:
        in_dir = args.tcd_root / spk_type / spk / "Clips" / "processed" / "audio16k_cropped" / args.cam
        if in_dir.exists():
            wavs = list(in_dir.glob("*.wav"))
            total_files += len(wavs)

    print(f"[INFO] Total TCD wavs to consider (all speakers): {total_files}")

    # ---------- Normalize ----------
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "speaker_type","speaker","in_path","out_path","status",
            "sr_in","target_rms","in_rms","gain",
            "peak_before","peak_after"
        ])

        for spk_type, spk in speakers:
            in_dir = args.tcd_root / spk_type / spk / "Clips" / "processed" / "audio16k_cropped" / args.cam
            out_dir = args.tcd_root / spk_type / spk / "Clips" / "processed" / args.out_folder_name / args.cam

            if not in_dir.exists():
                print(f"[WARN] Missing input dir: {in_dir}")
                continue

            wavs = sorted(in_dir.glob("*.wav"))
            print(f"\n[INFO] {spk_type}/{spk}/{args.cam}: {len(wavs)} files")

            for in_wav in tqdm(wavs, desc=f"{spk_type}/{spk}", unit="file"):
                out_wav = out_dir / in_wav.name
                try:
                    stats = normalize_one_wav(in_wav, out_wav, target_rms, args.peak_limit)
                except Exception as e:
                    stats = {
                        "status": f"fail:{type(e).__name__}",
                        "sr": -1,
                        "in_rms": -1.0,
                        "gain": 1.0,
                        "peak_before": -1.0,
                        "peak_after": -1.0,
                    }

                w.writerow([
                    spk_type, spk, str(in_wav), str(out_wav),
                    stats["status"],
                    stats["sr"],
                    f"{target_rms:.8f}",
                    f"{stats['in_rms']:.8f}" if isinstance(stats["in_rms"], float) else stats["in_rms"],
                    f"{stats['gain']:.8f}",
                    f"{stats['peak_before']:.8f}",
                    f"{stats['peak_after']:.8f}",
                ])

    print(f"\n[DONE] Wrote report: {args.report_csv}")
    print("[DONE] TCD audio normalization complete.")


if __name__ == "__main__":
    main()
