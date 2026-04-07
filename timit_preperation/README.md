# TCD-TIMIT Preparation

This folder contains the TCD-TIMIT preprocessing pipeline used to produce AV-HuBERT-ready inputs.

## Files in this folder

- [AVH_Data_Preprocessing.ipynb](AVH_Data_Preprocessing.ipynb): main clean preprocessing notebook.
- [align_mouth_stabilised.py](align_mouth_stabilised.py): robust mouth ROI cropper with stabilization and landmark gap handling.
- [normalise_timit_audio.py](normalise_timit_audio.py): RMS-targeted audio normalization against LRS3 loudness references.
- [compare_lrs3_tcd_loudness.py](compare_lrs3_tcd_loudness.py): loudness comparison utility between LRS3 and TCD-TIMIT.

## Notebook pipeline

The notebook is intentionally linear. Run top to bottom.

1. Configure dataset root, speaker scope, camera view, and tool paths.
2. Convert media:
: video to 25 fps and audio to 16 kHz mono.
3. Build per-speaker clip list for landmark extraction.
4. Run landmark extraction.
5. Generate stabilized 96x96 mouth ROI clips.

## Crop and stabilization behavior

The crop helper script is built for real-world noisy landmarks and includes:

- invalid landmark frame fill/repair
- interpolation across missing frames
- transform smoothing and jump clamping
- frame-count-safe output handling

This is why the script is preferred over simpler crop loops when speaker/video quality varies.

## Figures

![Audio trim and sync concept](../figures/audio_trim.png)

![Landmark point visualization](../figures/pkl_landmark_points.png)

![Batching and prep flow](../figures/batching_fix.png)

## Expected outputs

For each speaker and camera view, the workflow produces:

- converted full-face video clips at 25 fps
- converted audio at 16 kHz mono
- landmark files per clip
- aligned mouth ROI videos

## Scope boundaries

- This folder is TCD-TIMIT-specific.
- Augmentation workflows belong in [augmentation](../augmentation).
- LRS3-specific preparation belongs in [lrs3_preperation](../lrs3_preperation).
