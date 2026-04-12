#!/usr/bin/env python3
"""
LRS3 Mouth Cropping Script

Uses align_mouth_stabilise.py to create 96x96 stabilized mouth crops from blurred full-face videos.

Input:
  - Blurred video directory (e.g., E:\lrs3_rj\lrs3\Visemes_G_H_I_J_K_word\{speaker}\{stem}.mp4)
  - Landmark directory (e.g., E:\lrs3_rj\lrs3\landmark\trainval\{speaker}\{stem}.pkl)

Output:
  - Cropped video directory with same structure:
    E:\lrs3_rj\lrs3\Visemes_G_H_I_J_K_word_roi\{speaker}\{stem}.mp4

Usage:
    python lrs3_mouth_crop.py \
        --blur-video-dir E:\lrs3_rj\lrs3\Visemes_G_H_I_J_K_word \
        --landmark-dir E:\lrs3_rj\lrs3\landmark\trainval \
        --output-base-dir E:\lrs3_rj\lrs3\Visemes_G_H_I_J_K_word_roi

"""
from pathlib import Path
import argparse
import subprocess
import sys
import json
from datetime import datetime


# Config
ALIGN_SCRIPT = Path(r"C:\Users\irish\Computer_Electronic_Engineering_Year5\AVSR_project\av_hubert\av_hubert\avhubert\preparation\align_mouth_stabilised.py")
MEAN_FACE_NPY = Path(r"C:\Users\irish\Computer_Electronic_Engineering_Year5\AVSR_project\assets\mean_face\20words_mean_face.npy")

# Mouth crop parameters
CROP_W, CROP_H = 96, 96
WINDOW_MARGIN = 12
START_IDX, STOP_IDX = 48, 68

# Stabilization parameters
GRAY = 0
STAB_ALPHA = 0.15
MOUTH_ALPHA = 0.20
MAX_T_JUMP = 3.0
MAX_R_JUMP = 3.0
MAX_S_JUMP = 0.03
FFMPEG = "ffmpeg"

# ===================================


def validate_inputs(blur_video_dir, landmark_dir, output_base_dir):
    """Validate that required files and directories exist."""
    blur_video_dir = Path(blur_video_dir)
    landmark_dir = Path(landmark_dir)
    output_base_dir = Path(output_base_dir)
    
    if not blur_video_dir.exists():
        raise FileNotFoundError(f"Blur video directory not found: {blur_video_dir}")
    if not landmark_dir.exists():
        raise FileNotFoundError(f"Landmark directory not found: {landmark_dir}")
    if not ALIGN_SCRIPT.exists():
        raise FileNotFoundError(f"align_mouth_stabilised.py not found: {ALIGN_SCRIPT}")
    if not MEAN_FACE_NPY.exists():
        raise FileNotFoundError(f"Mean face npy not found: {MEAN_FACE_NPY}")
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    return blur_video_dir, landmark_dir, output_base_dir


def discover_speakers(blur_video_dir):
    """Discover speaker directories in blur video directory."""
    speakers = []
    for item in sorted(blur_video_dir.iterdir()):
        if item.is_dir():
            speakers.append(item.name)
    return speakers


def find_matching_clips(video_dir, landmark_dir):
    """
    Find all video/landmark pairs in speaker directories.
    Returns: [(video_path, landmark_path, stem), ...]
    """
    triplets = []
    
    # Get all .mp4 files in video_dir
    videos = {p.stem: p for p in video_dir.glob("*.mp4")}
    
    # Get all .pkl files in landmark_dir
    landmarks = {p.stem: p for p in landmark_dir.glob("*.pkl")}
    
    # Find common stems
    common_stems = sorted(set(videos.keys()) & set(landmarks.keys()))
    
    for stem in common_stems:
        triplets.append((videos[stem], landmarks[stem], stem))
    
    return triplets


def get_speaker_completion_status(speaker_code, blur_video_dir, landmark_dir, output_base_dir):
    """
    Determine whether a speaker is fully complete based on expected stems.

    A speaker is complete if every expected stem (video+landmark pair) has an
    output mp4 in output_base_dir/speaker_code.
    """
    spk_video_dir = blur_video_dir / speaker_code
    spk_landmark_dir = landmark_dir / speaker_code
    spk_out_dir = output_base_dir / speaker_code

    if not spk_video_dir.exists() or not spk_landmark_dir.exists():
        return {
            "speaker": speaker_code,
            "exists": False,
            "expected": 0,
            "done": 0,
            "complete": False,
        }

    triplets = find_matching_clips(spk_video_dir, spk_landmark_dir)
    expected_stems = {stem for _, _, stem in triplets}
    done_stems = {p.stem for p in spk_out_dir.glob("*.mp4")} if spk_out_dir.exists() else set()

    complete = bool(expected_stems) and expected_stems.issubset(done_stems)
    return {
        "speaker": speaker_code,
        "exists": True,
        "expected": len(expected_stems),
        "done": len(expected_stems & done_stems),
        "complete": complete,
    }


def safe_run(cmd):
    """
    Run subprocess safely with UTF-8 decoding.
    Returns: (return_code, stdout_text)
    """
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", errors="replace")
    return p.returncode, out


def process_speaker(
    speaker_code,
    blur_video_dir,
    landmark_dir,
    output_base_dir,
    clean_output=False,
):
    """
    Process all clips for one speaker.
    
    Args:
        speaker_code: Speaker directory name (e.g., "00j9bKdiOjk")
        blur_video_dir: Path to blurred video directory root
        landmark_dir: Path to landmark directory root
        output_base_dir: Path to output directory root
        clean_output: If True, remove existing speaker mp4 outputs before processing
    """
    
    # Build paths for this speaker
    spk_video_dir = blur_video_dir / speaker_code
    spk_landmark_dir = landmark_dir / speaker_code
    spk_out_dir = output_base_dir / speaker_code
    
    # Validate
    if not spk_video_dir.exists():
        print(f"[SKIP] Speaker video dir not found: {spk_video_dir}")
        return False
    if not spk_landmark_dir.exists():
        print(f"[SKIP] Speaker landmark dir not found: {spk_landmark_dir}")
        return False
    
    # Find matching clips
    triplets = find_matching_clips(spk_video_dir, spk_landmark_dir)
    if not triplets:
        print(f"[{speaker_code}] No video/landmark pairs found")
        return False
    
    print(f"[{speaker_code}] Found {len(triplets)} video/landmark pairs")
    
    # Create output directory
    spk_out_dir.mkdir(parents=True, exist_ok=True)

    if clean_output:
        for p in spk_out_dir.glob("*.mp4"):
            p.unlink()
        print(f"[{speaker_code}] Cleared existing output mp4 files before redo")
    
    # Create temporary filelist
    stems = [stem for _, _, stem in triplets]
    filelist_path = spk_out_dir / f"_filelist_{speaker_code}.txt"
    filelist_path.write_text("\n".join(stems) + "\n", encoding="utf-8")
    
    # Build command for align_mouth_stabilised.py
    cmd = [
        sys.executable,
        str(ALIGN_SCRIPT),
        "--video-direc", str(spk_video_dir),
        "--landmark-direc", str(spk_landmark_dir),
        "--filename-path", str(filelist_path),
        "--save-direc", str(spk_out_dir),
        "--mean-face", str(MEAN_FACE_NPY),
        "--crop-width", str(CROP_W),
        "--crop-height", str(CROP_H),
        "--start-idx", str(START_IDX),
        "--stop-idx", str(STOP_IDX),
        "--window-margin", str(WINDOW_MARGIN),
        "--ffmpeg", FFMPEG,
        "--rank", "0",
        "--nshard", "1",
        "--gray", str(GRAY),
        "--stab-alpha", str(STAB_ALPHA),
        "--mouth-alpha", str(MOUTH_ALPHA),
        "--max-t-jump", str(MAX_T_JUMP),
        "--max-r-jump", str(MAX_R_JUMP),
        "--max-s-jump", str(MAX_S_JUMP),
    ]
    
    print(f"[{speaker_code}] Running mouth crop processing...")
    rc, out = safe_run(cmd)
    
    if out.strip():
        print(out.strip())
    
    if rc != 0:
        print(f"[{speaker_code}] FAILED (return code {rc})")
        return False
    else:
        print(f"[{speaker_code}] SUCCESS")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Crop LRS3 blurred videos to 96x96 mouth regions using landmarks."
    )
    parser.add_argument(
        "--blur-video-dir",
        required=True,
        help="Path to blurred video directory (e.g., E:\\lrs3_rj\\lrs3\\Visemes_G_H_I_J_K_word)",
    )
    parser.add_argument(
        "--landmark-dir",
        required=True,
        help="Path to landmark directory (e.g., E:\\lrs3_rj\\lrs3\\landmark\\trainval)",
    )
    parser.add_argument(
        "--output-base-dir",
        required=True,
        help="Base output directory (e.g., E:\\lrs3_rj\\lrs3\\Visemes_G_H_I_J_K_word_roi)",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help=(
            "Resume mode: redo the last completed speaker folder, then continue with later speakers, "
            "skipping already completed folders."
        ),
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    print("="*70)
    print("LRS3 Mouth Crop Processing")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    try:
        blur_video_dir, landmark_dir, output_base_dir = validate_inputs(
            args.blur_video_dir,
            args.landmark_dir,
            args.output_base_dir,
        )
    except FileNotFoundError as e:
        print(f"Validation error: {e}")
        return 1
    
    print(f"\nInput (blurred videos): {blur_video_dir}")
    print(f"Input (landmarks):      {landmark_dir}")
    print(f"Output:                 {output_base_dir}")
    print(f"Resume mode:            {args.resume}")
    print()
    
    # Discover speakers
    speakers = discover_speakers(blur_video_dir)
    if not speakers:
        print("No speakers found in blur video directory")
        return 1
    
    print(f"Found {len(speakers)} speakers\n")
    
    # Decide which speakers to process
    redo_speaker = None
    if args.resume:
        status_list = [
            get_speaker_completion_status(s, blur_video_dir, landmark_dir, output_base_dir)
            for s in speakers
        ]
        completed = [s["speaker"] for s in status_list if s["complete"]]

        if completed:
            redo_speaker = completed[-1]
            redo_idx = speakers.index(redo_speaker)
            process_speakers = []
            for idx, speaker in enumerate(speakers):
                if idx < redo_idx:
                    continue
                status = next(x for x in status_list if x["speaker"] == speaker)
                if idx == redo_idx:
                    process_speakers.append(speaker)
                elif status["complete"]:
                    continue
                else:
                    process_speakers.append(speaker)

            print(f"Resume plan: redoing last completed speaker '{redo_speaker}', then continuing forward.")
            print(f"Speakers selected for this run: {len(process_speakers)} / {len(speakers)}")
        else:
            # Nothing complete yet; start from the beginning
            process_speakers = speakers
            print("Resume mode requested, but no completed speaker folders were detected. Starting from first speaker.")
    else:
        process_speakers = speakers

    # Process each speaker
    success_count = 0
    fail_count = 0
    
    for speaker_code in process_speakers:
        print(f"\n{'='*70}")
        print(f"Processing speaker: {speaker_code}")
        print(f"{'='*70}")
        
        ok = process_speaker(
            speaker_code,
            blur_video_dir,
            landmark_dir,
            output_base_dir,
            clean_output=(args.resume and speaker_code == redo_speaker),
        )
        
        if ok:
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "="*70)
    print("Mouth Crop Processing Complete")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
