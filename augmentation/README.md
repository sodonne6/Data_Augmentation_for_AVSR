# Augmentation

This folder contains the augmentation experiments that are applied after the base dataset has already been prepared.

## Files in this folder

- [interpolation_lead_lag.ipynb](interpolation_lead_lag.ipynb): TCD-TIMIT interpolation workflow.
- [interpolation_lead_lag_lrs3.ipynb](interpolation_lead_lag_lrs3.ipynb): LRS3 interpolation workflow.
- [smart_blur_notebook.ipynb](smart_blur_notebook.ipynb): TCD-TIMIT smart blur workflow.
- [smart_blur_notebook_lrs3.ipynb](smart_blur_notebook_lrs3.ipynb): LRS3 smart blur workflow.

## Interpolation workflow

The interpolation notebooks create temporally shifted or densified variants of already-prepared clips. The flow is straightforward: start with a 25 fps clip, interpolate it to 50 fps, then downsample back to 25 fps using the mid-frame phase so the inserted frames are what remains in the final output.

![Frame interpolation concept](../figures/interpolate_frames.png)

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

## Scope boundaries

- Keep dataset conversion and landmark generation in [timit_preperation](../timit_preperation) and [lrs3_preperation](../lrs3_preperation).
- Keep this folder focused on augmentation methods only.
