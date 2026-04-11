# TCD-TIMIT Preparation

This folder contains the TCD-TIMIT preprocessing pipeline used to produce AV-HuBERT-ready inputs.

## Files in this folder

- [AVH_Data_Preprocessing.ipynb](AVH_Data_Preprocessing.ipynb): main preprocessing notebook.
- [align_mouth_stabilised.py](align_mouth_stabilised.py): stabilized mouth ROI cropper.
- [normalise_timit_audio.py](normalise_timit_audio.py): audio normalization helper.
- [trim_tcd_timit_dataset.py](trim_tcd_timit_dataset.py): detects leading/trailing silence from audio using a configurable dB threshold, then applies the same trim window to audio and video.
- [compare_lrs3_tcd_loudness.py](compare_lrs3_tcd_loudness.py): loudness comparison utility.

## Notebook flow

The notebook is intentionally linear. It starts by setting the dataset root, speaker scope, camera view, and tool paths, then converts the source media into 25 fps video and 16 kHz mono audio. The conversion step gives the rest of the pipeline a consistent input format.

The audio normalization stage is handled by [normalise_timit_audio.py](normalise_timit_audio.py). It scans a random sample of LRS3 audio, computes a target RMS level from the sample median, and then applies gain to each TCD clip so loudness is matched to that target. The script enforces a peak limit to prevent clipping, writes normalized files as 16 kHz mono PCM16, and saves a CSV report with per-file RMS, gain, and peak-before/peak-after values.

Once the media is standardized, the notebook builds the per-speaker clip list used for landmark extraction. The landmark layout below is the one the cropper relies on when it stabilizes the mouth region.

![Landmark layout for mouth cropping](../figures/pkl_landmark_points.png)

After landmarks are available, the notebook runs the crop stage to produce stabilized 96x96 mouth ROI clips. The crop helper is designed for noisy or incomplete landmark sequences and includes invalid-frame repair, interpolation, smoothing, jump clamping, and frame-count-safe output handling.

Audio trimming and alignment are part of the same preparation flow: once media is standardized, leading and trailing silence is detected in the audio, trimmed to keep only the speech segment, and the corresponding video clip is trimmed by the same amount to preserve A/V synchronization.

Use [trim_tcd_timit_dataset.py](trim_tcd_timit_dataset.py) when you want threshold-controlled silence trimming that keeps audio and video synchronized. The key controls are `--noise_db` (silence threshold in dB) and `--min_sil_dur` (minimum silence duration in seconds).

```powershell
python trim_tcd_timit_dataset.py --src_root "E:\TCD_TIMIT" --dst_root "E:\TCD_TIMIT_trimmed" --noise_db -35 --min_sil_dur 0.20 --pad_start 0.11 --pad_end 0.30 --only both --reencode_video
```

![Audio trim and sync concept](../figures/audio_trim.png)

## Expected outputs

For each speaker and camera view, the workflow produces converted full-face video clips at 25 fps, converted audio at 16 kHz mono, landmark files per clip, and aligned mouth ROI videos.


