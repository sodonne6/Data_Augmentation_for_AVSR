# LRS3 Preparation

This folder contains the LRS3 preprocessing workflow used before augmentation or model training.

## Files in this folder

- [lrs3_preprocessing.ipynb](lrs3_preprocessing.ipynb): clean LRS3 preprocessing notebook.
- [lrs3_mouth_crop.py](lrs3_mouth_crop.py): batch wrapper for stabilized 96x96 mouth ROI generation.
- [align_lrs3_with_mfa.py](align_lrs3_with_mfa.py): Montreal Forced Aligner runner for TextGrid generation.
- [compare_wer_ttest.py](compare_wer_ttest.py): evaluation comparison helper with significance testing.

## Notebook pipeline

1. Configure root paths, split, and optional speaker subset.
2. Optional media conversion to 25 fps and 16 kHz audio.
3. Extract landmarks on selected split.
4. Crop stabilized mouth ROI clips from landmark-video pairs.

## Alignment and blur readiness

For phone-level blur workflows, the common flow is:

1. Generate lab text files (if needed).
2. Run [align_lrs3_with_mfa.py](align_lrs3_with_mfa.py) to produce TextGrid alignments.
3. Use alignments in augmentation notebooks for phone-span blurring.

## Mouth crop script behavior

[lrs3_mouth_crop.py](lrs3_mouth_crop.py) does the following:

- validates video and landmark directory structure
- discovers speaker folders automatically
- pairs clips by common file stem
- executes stabilized mouth alignment/cropping with fixed parameters
- supports resume logic for long-running jobs

## Figures

![Landmark point visualization](../figures/pkl_landmark_points.png)

![Audio trim and sync concept](../figures/audio_trim.png)

![Dataset split concept](../figures/avsr_dataset_split_option1.drawio.png)

## Scope boundaries

- This folder is LRS3-specific.
- Augmentation method notebooks live in [augmentation](../augmentation).
- TCD-TIMIT-specific prep lives in [timit_preperation](../timit_preperation).
