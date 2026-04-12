#!/usr/bin/env python3
"""
MFA Alignment Script for LRS3 Dataset
=====================================

Aligns all .wav files in trainval directory using MFA 3.3.8 with english_mfa model.
Outputs TextGrid files alongside audio with same stem name.

Usage:
    python align_lrs3_with_mfa.py

Requirements:
    - MFA 3.3.8 installed in 'aligner' conda environment
    - .wav files in trainval/{speaker_code}/ directory
    - .lab files prepared (run Cell 3 of smart_blur_notebook_lrs3.ipynb first)
"""

import subprocess
import sys
from pathlib import Path
import shutil

# Config
TRAINVAL_ROOT = Path(r"E:\lrs3_rj\lrs3\trainval")

# MFA settings
CONDA_ENV = "aligner"
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_MPS = False  

# Helper functions

def run_command(cmd, description=""):
    """Run a shell command and report status."""
    if description:
        print(f"\n{'='*70}")
        print(f"  {description}")
        print(f"{'='*70}\n")
    
    print(f"Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode}")
        return False
    
    print("\nOK: command completed successfully")
    return True

def check_wav_and_lab_files(speaker_dir: Path) -> int:
    """Count how many .wav/.lab pairs exist in the speaker directory."""
    wav_files = set(p.stem for p in speaker_dir.glob("*.wav"))
    lab_files = set(p.stem for p in speaker_dir.glob("*.lab"))
    pairs = wav_files & lab_files
    return len(pairs)

def validate_setup():
    """Check that required directories and files exist."""
    print("\n" + "="*70)
    print("  SETUP VALIDATION")
    print("="*70 + "\n")
    
    # Check trainval root
    if not TRAINVAL_ROOT.exists():
        print(f"[ERROR] TRAINVAL_ROOT not found: {TRAINVAL_ROOT}")
        return False
    print(f"OK: TRAINVAL_ROOT exists: {TRAINVAL_ROOT}")
    
    # Count speakers and clips
    speaker_dirs = sorted([d for d in TRAINVAL_ROOT.iterdir() if d.is_dir()])
    if not speaker_dirs:
        print(f"[ERROR] No speaker directories found in {TRAINVAL_ROOT}")
        return False
    print(f"OK: found {len(speaker_dirs)} speaker directories")
    
    # Check for .wav/.lab pairs
    total_pairs = 0
    for spk_dir in speaker_dirs[:3]:  # Show first 3
        pairs = check_wav_and_lab_files(spk_dir)
        total_pairs += pairs
        print(f"  - {spk_dir.name}: {pairs} .wav/.lab pairs")
    
    # Count remaining
    for spk_dir in speaker_dirs[3:]:
        total_pairs += check_wav_and_lab_files(spk_dir)
    
    print(f"\nTotal .wav/.lab pairs ready for alignment: {total_pairs}")
    
    if total_pairs == 0:
        print("\n[ERROR] No .wav/.lab pairs found")
        print("  Make sure you've run Cell 3 (prepare .lab files) in the notebook first.")
        return False
    
    return True

# ========== MAIN MFA ALIGNMENT ==========

def main():
    print("\n" + "="*70)
    print("  LRS3 MFA ALIGNMENT - Montreal Forced Aligner 3.3.8")
    print("="*70)
    
    # Step 1: Validate setup
    if not validate_setup():
        print("\n[FAILED] Setup validation failed. Please check the errors above.")
        sys.exit(1)
    
    # Step 2: Run MFA alignment on entire trainval directory
    print("\n" + "="*70)
    print("  RUNNING MFA ALIGNMENT")
    print("="*70)
    print(f"\n  Input directory:     {TRAINVAL_ROOT}")
    print(f"  Acoustic model:      {MFA_ACOUSTIC_MODEL}")
    print(f"  Conda environment:   {CONDA_ENV}")
    print(f"  MPS (GPU on Mac):    {MFA_MPS}\n")
    
    # Build the MFA command.
    
    mps_flag = "--mps" if MFA_MPS else ""
    
    # Keep the built-in english_mfa setup explicit.
    
    cmd = (
        f"conda run -n {CONDA_ENV} "
        f"mfa align "
        f"--clean "
        f"--verbose "
        f"{mps_flag} "
        f"\"{TRAINVAL_ROOT}\" "
        f"english_mfa "
        f"english_mfa "
        f"\"{TRAINVAL_ROOT}\""
    )
    
    success = run_command(cmd, "Running MFA alignment on all speakers...")
    
    if not success:
        print("\n[ERROR] MFA alignment failed!")
        print("Check environment setup, input files, and MFA model availability.")
        sys.exit(1)
    
    # Step 3: Verify TextGrid output
    print("\n" + "="*70)
    print("  VERIFYING OUTPUT")
    print("="*70 + "\n")
    
    textgrid_count = 0
    speaker_dirs = sorted([d for d in TRAINVAL_ROOT.iterdir() if d.is_dir()])
    
    for spk_dir in speaker_dirs:
        tg_files = list(spk_dir.glob("*.TextGrid"))
        if tg_files:
            textgrid_count += len(tg_files)
            if len(speaker_dirs) <= 5:  # Only show if few speakers
                print(f"  {spk_dir.name}: {len(tg_files)} TextGrid files")
    
    if textgrid_count == 0:
        print("[WARNING] No TextGrid files found in output!")
        print("  Check the MFA output above for errors.")
        sys.exit(1)
    
    print(f"\nOK: total TextGrid files created: {textgrid_count}")
    print("OK: TextGrids are saved alongside .wav files in each speaker folder")
    
    # Step 4: Success message
    print("\n" + "="*70)
    print("  MFA ALIGNMENT COMPLETE")
    print("="*70)
    print("\nTextGrids are ready for use in the smart blur notebook:")
    print("  Cell 10 will now find and use these TextGrids for phoneme-level blurring.")

if __name__ == "__main__":
    main()
