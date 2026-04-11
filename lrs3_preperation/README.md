# LRS3 Preparation

This folder contains the LRS3 preprocessing workflow used before augmentation or model training.

## Files in this folder

- [lrs3_preprocessing.ipynb](lrs3_preprocessing.ipynb): clean LRS3 preprocessing notebook.
- [lrs3_mouth_crop.py](lrs3_mouth_crop.py): stabilized 96x96 mouth ROI batch wrapper.
- [align_lrs3_with_mfa.py](align_lrs3_with_mfa.py): Montreal Forced Aligner runner for TextGrid generation.
- [compare_wer_ttest.py](compare_wer_ttest.py): evaluation comparison helper with significance testing.

## External requirements and expected layout

- `ffmpeg` and `ffprobe` available in your active environment (or set explicit paths in notebook cells).
- MFA models used by the alignment script: `english_mfa` acoustic and `english_mfa` dictionary.

Expected LRS3 structure:

- Video split root: `.../lrs3/<split>/<speaker>/<utt>.mp4`
- Landmark root: `.../lrs3/landmark/<split>/<speaker>/<utt>.pkl`
- TextGrid output after MFA: next to each `.wav`/`.lab` pair in your alignment input folder

## Notebook flow

The notebook starts by configuring the split and optional speaker subset, then runs ROI cropping on the already-prepared media.

The split structure matters here, because the preprocessing uses the dataset organization to decide which speaker folders to walk. The diagram below is the visual reminder of how the dataset is partitioned before alignment and cropping happen.

After the input structure is settled, the notebook crops stabilized mouth ROI clips from the matching video-landmark pairs. The landmark files are treated as existing inputs for LRS3 preprocessing.

## Alignment and blur readiness

If you need phone-level blur later, the alignment step is what makes that possible. The workflow is: generate lab text files if needed, run [align_lrs3_with_mfa.py](align_lrs3_with_mfa.py) to produce TextGrid alignments, and then feed those alignments into the augmentation notebooks for phone-span blurring.

For this preprocessing stage, no extra audio conversion or fps conversion is required.

## Mouth crop script behavior

[lrs3_mouth_crop.py](lrs3_mouth_crop.py) validates the directory structure, discovers speaker folders automatically, pairs clips by common file stem, runs stabilized mouth alignment/cropping with fixed parameters, and supports resume logic for long-running jobs.

The landmark layout below is the visual reference for the crop stage.

![Landmark layout for mouth cropping](../figures/pkl_landmark_points.png)


