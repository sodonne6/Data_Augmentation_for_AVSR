# Data Augmentation for AVSR

This repository contains the preprocessing, alignment, and augmentation workflows used for Audio Visual Speech Recognition experiments built around AV-HuBERT.

The repo is centered on three working areas:

- [timit_preperation](timit_preperation) for TCD-TIMIT preprocessing
- [lrs3_preperation](lrs3_preperation) for LRS3 preprocessing
- [augmentation](augmentation) for interpolation and smart blur experiments

## How the project flows

The usual path is: prepare a dataset, generate landmarks, crop a stable mouth ROI, then run augmentation on the prepared clips.

For the preprocessing stages, the media is first converted into model-friendly formats and then turned into mouth-centric inputs. The landmark step is what connects the full-face videos to the crop stage, and the crop stage is what produces the final AV-HuBERT-ready clips. The landmark visualization below shows the kind of point layout the cropper works from.

![Landmark layout used for cropping](figures/pkl_landmark_points.png)

Once the base clips are ready, the augmentation notebooks introduce controlled temporal perturbations. Interpolation-based augmentation creates additional intermediate frames, while smart blur focuses on viseme or phoneme regions inside the clip.

![Frame interpolation concept](figures/interpolate_frames.png)

![Mouth region targeting](../figures/ch4_fig_mouth_regions_final.png)

## Where to start

If you are working on TCD-TIMIT, start in [timit_preperation/README.md](timit_preperation/README.md). If you are working on LRS3, start in [lrs3_preperation/README.md](lrs3_preperation/README.md). If you already have prepared clips and want to experiment with augmentation, go straight to [augmentation/README.md](augmentation/README.md).

## Typical workflow order

1. Run the relevant preprocessing notebook.
2. Check that the landmark and crop outputs look correct.
3. Run the augmentation notebooks on the prepared clips.
4. Compare training or evaluation results.

The repository also contains supporting scripts, notes, and diagrams outside these main workflow folders, but the three folders above are the intended entry points.


