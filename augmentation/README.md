# Augmentation

This folder contains the augmentation experiments that are applied after the base dataset has already been prepared.

## Files in this folder

- [interpolation_lead_lag.ipynb](interpolation_lead_lag.ipynb): TCD-TIMIT interpolation workflow.
- [interpolation_lead_lag_lrs3.ipynb](interpolation_lead_lag_lrs3.ipynb): LRS3 interpolation workflow.
- [smart_blur_notebook.ipynb](smart_blur_notebook.ipynb): TCD-TIMIT smart blur workflow.
- [smart_blur_notebook_lrs3.ipynb](smart_blur_notebook_lrs3.ipynb): LRS3 smart blur workflow.

## Interpolation workflow

The interpolation notebooks create temporally shifted or densified variants of already-prepared clips. The flow is straightforward: start with a 25 fps clip, interpolate it to 50 fps, then downsample back to 25 fps using the mid-frame phase so the inserted frames are what remains in the final output.

<img src="../figures/interpolate_frames.png" alt="Frame interpolation concept" width="900">

After the frame stream is rebuilt, the audio is trimmed and realigned to the new timeline so that the final clip stays synchronized. The timing diagram below is the reminder for why the lead and lag variants need separate alignment handling.

![Lead/lag timing](../figures/lead_lag_fixed.png)

The result is a set of aligned outputs for each source clip, typically including the base stream plus lead and lag variants.

## Smart blur workflow

Smart blur works at a more semantic level. The notebook selects target viseme groups, chooses whether the blur span is defined at word level or phone level, loads the relevant alignment data, and then blurs only the chosen temporal spans.

The mouth-region illustration is the visual reference for what part of the face is being protected or suppressed during the blur step.

![Mouth region targeting](../figures/ch4_fig_mouth_regions_final.png)

This makes the blur method useful when you want to reduce access to specific articulatory cues while leaving the rest of the clip intact.

## Practical run order

1. Run one of the interpolation notebooks if you need lead/lag variants.
2. Inspect a few outputs to make sure timing and frame quality look right.
3. Run the smart blur notebook for the dataset you are working on.
4. Use the generated clips in training or evaluation pipelines.

## MFA TextGrid generation from .lab files

If you want phone-level spans in augmentation workflows, generate TextGrid alignments with Montreal Forced Aligner (MFA) first.

### Required corpus layout

For MFA to align correctly, each utterance must have:

- a `.wav` file
- a `.lab` transcript file
- the same basename
- both files in the same folder

Example:

```text
trainval/
	00j9bKdiOjk/
		50001.wav
		50001.lab
		50002.wav
		50002.lab
```

### 1. Create the aligner conda environment

From `AVSR_project/augmentation`:

```powershell
conda env create -f aligner_env.yml
conda activate aligner
```

Environment file: [augmentation/aligner_env.yml](aligner_env.yml)

### 2. Download MFA models (one-time)

```powershell
mfa model download acoustic english_mfa
mfa model download dictionary english_us_arpa
```

### 3. Run MFA alignment (generic command)

Use your corpus directory as both input and output so `.TextGrid` files are written next to each `.wav`/`.lab` pair:

```powershell
mfa align --clean --verbose "E:\lrs3_rj\lrs3\trainval" english_us_arpa english_mfa "E:\lrs3_rj\lrs3\trainval"
```

This produces files like:

```text
trainval/00j9bKdiOjk/50001.TextGrid
```

### 4. Use TextGrid files in augmentation notebooks

After alignment, phone-level smart blur can read these TextGrid files directly from the same speaker folders.


