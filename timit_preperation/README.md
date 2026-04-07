# TCD-TIMIT Preparation

This folder contains the TCD-TIMIT preprocessing pipeline used to produce AV-HuBERT-ready inputs.

## Files in this folder

- [AVH_Data_Preprocessing.ipynb](AVH_Data_Preprocessing.ipynb): main preprocessing notebook.
- [align_mouth_stabilised.py](align_mouth_stabilised.py): stabilized mouth ROI cropper.
- [normalise_timit_audio.py](normalise_timit_audio.py): audio normalization helper.
- [compare_lrs3_tcd_loudness.py](compare_lrs3_tcd_loudness.py): loudness comparison utility.

## Notebook flow

The notebook is intentionally linear. It starts by setting the dataset root, speaker scope, camera view, and tool paths, then converts the source media into 25 fps video and 16 kHz mono audio. The conversion step gives the rest of the pipeline a consistent input format.

Once the media is standardized, the notebook builds the per-speaker clip list used for landmark extraction. The landmark layout below is the one the cropper relies on when it stabilizes the mouth region.

![Landmark layout for mouth cropping](../figures/pkl_landmark_points.png)

After landmarks are available, the notebook runs the crop stage to produce stabilized 96x96 mouth ROI clips. The crop helper is designed for noisy or incomplete landmark sequences and includes invalid-frame repair, interpolation, smoothing, jump clamping, and frame-count-safe output handling.

The audio trimming/alignment concept is part of the same preparation story: once media is standardized, the audio side needs to stay in sync with the visual output so the final clips remain usable for AV-HuBERT training.

![Audio trim and sync concept](../figures/audio_trim.png)

## Expected outputs

For each speaker and camera view, the workflow produces converted full-face video clips at 25 fps, converted audio at 16 kHz mono, landmark files per clip, and aligned mouth ROI videos.

## Scope boundaries

- This folder is TCD-TIMIT-specific.
- Augmentation workflows belong in [augmentation](../augmentation).
- LRS3-specific preparation belongs in [lrs3_preperation](../lrs3_preperation).
