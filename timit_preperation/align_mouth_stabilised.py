#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust mouth ROI cropper (TCD-TIMIT style) with:
- optional grayscale output
- optional stabilization (EMA on similarity transform + jump clamps)
- robust handling when decoded frame count != landmark count
  (flush on early video end; repeat last landmark if landmarks shorter)

This keeps OUTPUT VIDEO LENGTH == DECODED INPUT VIDEO LENGTH.
"""

import os
import sys
import math
import pickle
import shutil
import tempfile
import subprocess
from pathlib import Path
from collections import deque

import cv2
import numpy as np
from tqdm import tqdm


# -----------------------------
# Helpers
# -----------------------------

def validate_and_fill_landmarks(landmarks, pkl_path=None, max_bad_ratio=0.40):
    """
    Validate landmark frames and fill invalid ones with nearest valid frame.
    Rejects PKL files with >40% invalid frames.
    
    Args:
        landmarks: List of landmark arrays or None entries
        pkl_path: Path for error messages (optional)
        max_bad_ratio: Maximum ratio of invalid frames allowed (default 0.40)
    
    Returns:
        List of validated landmark arrays
    
    Raises:
        ValueError: If too many invalid frames or no valid frames found
    """
    if landmarks is None or len(landmarks) == 0:
        raise ValueError(f"No landmark frames found in {pkl_path if pkl_path else 'PKL'}")
    
    frames = list(landmarks)
    
    # Validate each frame: must be None or shape (68, 2)
    validated = []
    for i, lm in enumerate(frames):
        if lm is None:
            validated.append(None)
        else:
            arr = np.asarray(lm, dtype=np.float32)
            if arr.shape == (68, 2):
                validated.append(arr)
            else:
                validated.append(None)
    
    # Find valid frames
    valid_idx = [i for i, x in enumerate(validated) if x is not None]
    
    if not valid_idx:
        raise ValueError(f"No valid landmark frames found in {pkl_path if pkl_path else 'PKL'}")
    
    # Check bad frame ratio
    bad_count = len(validated) - len(valid_idx)
    bad_ratio = bad_count / len(validated)
    
    if bad_ratio > max_bad_ratio:
        raise ValueError(
            f"Too many invalid landmark frames in {pkl_path if pkl_path else 'PKL'}: "
            f"{bad_count}/{len(validated)} ({bad_ratio:.1%})"
        )
    
    # Fill sparse invalid frames with nearest valid frame
    if bad_count > 0:
        for i in range(len(validated)):
            if validated[i] is not None:
                continue
            # Find nearest valid frame
            nearest = min(valid_idx, key=lambda j: abs(j - i))
            validated[i] = validated[nearest].copy()
        
        if pkl_path:
            print(f"[WARN] Filled {bad_count} invalid landmark frame(s): {pkl_path.name if hasattr(pkl_path, 'name') else pkl_path}")
    
    return validated


def landmarks_interpolate(landmarks):
    """Fill None landmarks by linear interpolation; edge-fill at ends."""
    if landmarks is None or len(landmarks) == 0:
        return landmarks

    lms = list(landmarks)
    n = len(lms)

    good = [i for i, lm in enumerate(lms) if lm is not None]
    if not good:
        return lms

    # edge-fill
    first = good[0]
    last = good[-1]
    for i in range(0, first):
        lms[i] = lms[first]
    for i in range(last + 1, n):
        lms[i] = lms[last]

    # interpolate internal gaps
    for a, b in zip(good, good[1:]):
        if b == a + 1:
            continue
        la = lms[a].astype(np.float32)
        lb = lms[b].astype(np.float32)
        gap = b - a
        for k in range(1, gap):
            t = k / gap
            lms[a + k] = (1.0 - t) * la + t * lb

    return lms


def to_gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def estimate_similarity(pts_src, pts_dst):
    """
    Estimate similarity transform from pts_src -> pts_dst using OpenCV.
    Returns 2x3 affine matrix (similarity-ish).
    """
    pts_src = np.asarray(pts_src, dtype=np.float32)
    pts_dst = np.asarray(pts_dst, dtype=np.float32)

    # estimateAffinePartial2D gives rotation+scale+translation (no shear ideally)
    M, inliers = cv2.estimateAffinePartial2D(
        pts_src, pts_dst,
        method=cv2.LMEDS
    )
    if M is None:
        # fallback: identity
        M = np.array([[1, 0, 0],
                      [0, 1, 0]], dtype=np.float32)
    return M.astype(np.float32)


def affine_to_params(M):
    """
    Convert 2x3 affine (assumed similarity-ish) to (tx, ty, rot_deg, scale)
    """
    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    c, d, ty = M[1, 0], M[1, 1], M[1, 2]

    scale = math.sqrt(max(1e-12, a * a + c * c))
    rot = math.degrees(math.atan2(c, a))
    return tx, ty, rot, scale


def params_to_affine(tx, ty, rot_deg, scale):
    th = math.radians(rot_deg)
    a = scale * math.cos(th)
    c = scale * math.sin(th)
    b = -scale * math.sin(th)
    d = scale * math.cos(th)
    M = np.array([[a, b, tx],
                  [c, d, ty]], dtype=np.float32)
    return M


def clamp_jump(prev, cur, max_jump):
    if max_jump <= 0:
        return cur
    if abs(cur - prev) > max_jump:
        return prev
    return cur


def center_crop(img, cx, cy, w, h):
    """
    Crop (w,h) around (cx,cy). Pads with border-replicate if out of bounds.
    """
    H, W = img.shape[:2]
    x1 = int(round(cx - w / 2))
    y1 = int(round(cy - h / 2))
    x2 = x1 + w
    y2 = y1 + h

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - W)
    pad_b = max(0, y2 - H)

    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, borderType=cv2.BORDER_REPLICATE)
        x1 += pad_l
        y1 += pad_t
        # W/H change but we now safely slice

    return img[y1:y2, x1:x2]


def write_video_ffmpeg(frames, out_mp4, ffmpeg_bin="ffmpeg", fps=25, crf=20):
    """
    Write list/array of frames to mp4 via ffmpeg using temp png sequence.
    Frames can be:
      - grayscale: HxW uint8
      - color: HxWx3 uint8 (BGR ok)
    """
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # Normalize frames iterable
    if isinstance(frames, np.ndarray):
        if frames.dtype == object:
            frames = [np.asarray(x) for x in frames.tolist()]
        else:
            frames = list(frames)
    else:
        frames = list(frames)

    if len(frames) == 0:
        raise RuntimeError(f"No frames to write: {out_mp4}")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, fr in enumerate(frames):
            fr = np.asarray(fr)
            if fr.dtype != np.uint8:
                fr = fr.astype(np.uint8)

            # Ensure 2D or 3D
            if fr.ndim == 2:
                pass
            elif fr.ndim == 3 and fr.shape[2] == 3:
                pass
            else:
                raise RuntimeError(f"Bad frame shape at {i}: {fr.shape}")

            cv2.imwrite(str(td / f"{i:010d}.png"), fr)

        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(fps),
            "-i", str(td / "%010d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            str(out_mp4)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def crop_patch(
    video_path,
    landmarks_list,
    mean_face,
    crop_w=96,
    crop_h=96,
    start_idx=48,
    stop_idx=68,
    window_margin=12,
    gray=1,
    stab_alpha=0.15,
    mouth_alpha=0.20,
    max_t_jump=3.0,
    max_r_jump=3.0,
    max_s_jump=0.03,
):
    """
    Returns list of cropped frames (len == decoded input frame count).
    Robust to mismatched landmark length.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    # Validate and fill landmarks (no validation failure here - caller handles)
    # If landmarks_list is None, fallback to resize-only
    if landmarks_list is None or len(landmarks_list) == 0:
        out = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            if gray:
                fr = to_gray(fr)
            fr = cv2.resize(fr, (crop_w, crop_h), interpolation=cv2.INTER_AREA)
            out.append(fr)
        cap.release()
        return out
    
    # Landmarks provided - use nearest-frame filling (already validated by caller)
    lms = landmarks_list

    mean_face = np.asarray(mean_face, dtype=np.float32)

    # Stable points for similarity (nose/eyes)
    stable_ids = [33, 36, 39, 42, 45]
    mean_stable = mean_face[stable_ids]

    q_frames = deque()
    q_lms = deque()

    out = []

    # Transform EMA state
    prev_tx = prev_ty = prev_rot = prev_scale = None
    prev_mouth_cx = prev_mouth_cy = None

    # margin: keep as given but if the video is shorter we handle it in flush
    margin = max(1, int(window_margin))

    idx = 0
    last_M = None  # last affine used (after smoothing/clamp)

    def compute_M(smoothed_landmarks):
        nonlocal prev_tx, prev_ty, prev_rot, prev_scale, last_M

        src = np.asarray(smoothed_landmarks, dtype=np.float32)[stable_ids]
        M_raw = estimate_similarity(src, mean_stable)

        tx, ty, rot, scale = affine_to_params(M_raw)

        # Initialize EMA
        if prev_tx is None:
            prev_tx, prev_ty, prev_rot, prev_scale = tx, ty, rot, scale
        else:
            # Clamp jumps (least abrasive: don't accept wild spikes)
            tx = clamp_jump(prev_tx, tx, max_t_jump)
            ty = clamp_jump(prev_ty, ty, max_t_jump)
            rot = clamp_jump(prev_rot, rot, max_r_jump)
            scale = clamp_jump(prev_scale, scale, max_s_jump)

            # EMA
            a = float(stab_alpha)
            if a > 0:
                prev_tx = (1 - a) * prev_tx + a * tx
                prev_ty = (1 - a) * prev_ty + a * ty
                prev_rot = (1 - a) * prev_rot + a * rot
                prev_scale = (1 - a) * prev_scale + a * scale
            else:
                prev_tx, prev_ty, prev_rot, prev_scale = tx, ty, rot, scale

        last_M = params_to_affine(prev_tx, prev_ty, prev_rot, prev_scale)
        return last_M

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Choose landmark for this frame (repeat last if we ran out)
        if idx < len(lms):
            lm = lms[idx]
        else:
            lm = lms[-1]

        # Some pkls may store lists; normalize
        lm = np.asarray(lm, dtype=np.float32)

        q_frames.append(frame)
        q_lms.append(lm)

        if len(q_frames) < margin:
            idx += 1
            continue

        smoothed = np.mean(np.stack(list(q_lms), axis=0), axis=0)
        M = compute_M(smoothed)

        cur_frame = q_frames.popleft()
        cur_lm = q_lms.popleft()

        # Warp frame to mean face space
        warped = cv2.warpAffine(cur_frame, M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Transform landmarks to warped space
        ones = np.ones((cur_lm.shape[0], 1), dtype=np.float32)
        pts = np.hstack([cur_lm, ones])
        cur_lm_w = (M @ pts.T).T  # Nx2

        mouth = cur_lm_w[start_idx:stop_idx]
        cx, cy = float(np.mean(mouth[:, 0])), float(np.mean(mouth[:, 1]))

        # Mouth center EMA
        if prev_mouth_cx is None:
            prev_mouth_cx, prev_mouth_cy = cx, cy
        else:
            ma = float(mouth_alpha)
            if ma > 0:
                prev_mouth_cx = (1 - ma) * prev_mouth_cx + ma * cx
                prev_mouth_cy = (1 - ma) * prev_mouth_cy + ma * cy
            else:
                prev_mouth_cx, prev_mouth_cy = cx, cy

        patch = center_crop(warped, prev_mouth_cx, prev_mouth_cy, crop_w, crop_h)
        if gray:
            patch = to_gray(patch)

        out.append(patch)
        idx += 1

    # --- Flush tail frames so output length == decoded frames ---
    # If we never computed a transform (video shorter than margin), compute once from whatever we have.
    if last_M is None and len(q_lms) > 0:
        smoothed = np.mean(np.stack(list(q_lms), axis=0), axis=0)
        last_M = compute_M(smoothed)

    # If still none, identity
    if last_M is None:
        last_M = np.array([[1, 0, 0],
                           [0, 1, 0]], dtype=np.float32)

    # Use last mouth center; if none, estimate from last lm in queue
    if prev_mouth_cx is None and len(q_lms) > 0:
        lm0 = q_lms[-1]
        src = np.asarray(lm0, dtype=np.float32)[stable_ids]
        M0 = estimate_similarity(src, mean_stable)
        ones = np.ones((lm0.shape[0], 1), dtype=np.float32)
        pts = np.hstack([lm0, ones])
        lm_w = (M0 @ pts.T).T
        mouth = lm_w[start_idx:stop_idx]
        prev_mouth_cx = float(np.mean(mouth[:, 0]))
        prev_mouth_cy = float(np.mean(mouth[:, 1]))

    while len(q_frames) > 0:
        cur_frame = q_frames.popleft()
        cur_lm = q_lms.popleft() if len(q_lms) > 0 else lms[-1]

        warped = cv2.warpAffine(cur_frame, last_M, (256, 256), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        if prev_mouth_cx is None:
            # fallback: center of image
            cx, cy = 128.0, 128.0
        else:
            cx, cy = prev_mouth_cx, prev_mouth_cy

        patch = center_crop(warped, cx, cy, crop_w, crop_h)
        if gray:
            patch = to_gray(patch)
        out.append(patch)

    cap.release()
    return out


# -----------------------------
# CLI
# -----------------------------

def load_args():
    import argparse
    p = argparse.ArgumentParser()

    p.add_argument("--video-direc", required=True, type=str)
    p.add_argument("--landmark-direc", required=True, type=str)
    p.add_argument("--filename-path", required=True, type=str)
    p.add_argument("--save-direc", required=True, type=str)
    p.add_argument("--mean-face", required=True, type=str)

    p.add_argument("--crop-width", type=int, default=96)
    p.add_argument("--crop-height", type=int, default=96)
    p.add_argument("--start-idx", type=int, default=48)
    p.add_argument("--stop-idx", type=int, default=68)
    p.add_argument("--window-margin", type=int, default=12)

    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--nshard", type=int, default=1)

    p.add_argument("--ffmpeg", type=str, default="ffmpeg")

    # Output / stabilization knobs (as you were using)
    p.add_argument("--gray", type=int, default=1)
    p.add_argument("--stab-alpha", type=float, default=0.15)
    p.add_argument("--mouth-alpha", type=float, default=0.20)
    p.add_argument("--max-t-jump", type=float, default=3.0)
    p.add_argument("--max-r-jump", type=float, default=3.0)
    p.add_argument("--max-s-jump", type=float, default=0.03)

    return p.parse_args()


def main():
    args = load_args()

    video_dir = Path(args.video_direc)
    lmk_dir = Path(args.landmark_direc)
    save_dir = Path(args.save_direc)
    save_dir.mkdir(parents=True, exist_ok=True)

    mean_face = np.load(args.mean_face)

    # Read file ids (stems)
    fids = [ln.strip() for ln in Path(args.filename_path).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]

    # Sharding
    fids = fids[args.rank::args.nshard]

    for fid in tqdm(fids, desc=f"rank{args.rank}", ncols=80):
        vid = video_dir / f"{fid}.mp4"
        lmk = lmk_dir / f"{fid}.pkl"
        out_mp4 = save_dir / f"{fid}.mp4"

        if out_mp4.exists():
            continue

        if not vid.exists():
            print(f"[skip] missing video: {vid}")
            continue
        if not lmk.exists():
            print(f"[skip] missing landmark: {lmk}")
            continue

        try:
            with open(lmk, "rb") as f:
                landmarks = pickle.load(f)
            # Validate and fill landmarks with 40% threshold
            landmarks = validate_and_fill_landmarks(landmarks, pkl_path=lmk, max_bad_ratio=0.40)
        except ValueError as e:
            print(f"[skip] {e}")
            continue
        except Exception as e:
            print(f"[skip] cannot read pkl {lmk}: {e}")
            continue

        seq = crop_patch(
            vid,
            landmarks,
            mean_face,
            crop_w=args.crop_width,
            crop_h=args.crop_height,
            start_idx=args.start_idx,
            stop_idx=args.stop_idx,
            window_margin=args.window_margin,
            gray=args.gray,
            stab_alpha=args.stab_alpha,
            mouth_alpha=args.mouth_alpha,
            max_t_jump=args.max_t_jump,
            max_r_jump=args.max_r_jump,
            max_s_jump=args.max_s_jump,
        )

        # Last-resort fallback: resize-only (keeps length)
        if seq is None or len(seq) == 0:
            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                print(f"[skip] cannot open video for fallback: {vid}")
                continue
            seq = []
            while True:
                ret, fr = cap.read()
                if not ret:
                    break
                if args.gray:
                    fr = to_gray(fr)
                fr = cv2.resize(fr, (args.crop_width, args.crop_height), interpolation=cv2.INTER_AREA)
                seq.append(fr)
            cap.release()

        try:
            write_video_ffmpeg(seq, out_mp4, ffmpeg_bin=args.ffmpeg, fps=25, crf=20)
        except Exception as e:
            print(f"[fail] write failed for {fid}: {e}")
            # don't crash whole job

    print("Done.")


if __name__ == "__main__":
    main()
