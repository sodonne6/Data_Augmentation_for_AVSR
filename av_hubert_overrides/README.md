# AV-HuBERT Overrides

This folder contains local AV-HuBERT behavior overrides used to stabilize data loading and batching when mixing base and augmented samples.

## Batching fix summary

The batching fix follows the same overall idea as AV-HuBERT's built-in augmentation behavior: a sample request first decides whether the base example is used or whether an augmentation path should be taken, and the final sampling distribution is kept explicit so the epoch composition stays stable.

![Batching probability flow](../figures/batching_fix.png)

The first diagram shows the probability split. A training request is made, the pipeline decides whether augmentation is applied, and if augmentation is selected, one augmentation source is chosen. That gives the intended distribution of base sample 50%, augmentation source 1 sample 25%, and augmentation source 2 sample 25%.

The second diagram shows how the data items are paired so that the base utterance and its augmented variants remain aligned by identity rather than colliding in the loader.

![Base to augmented pairing](../figures/two_aug_batching.drawio.png)

That pairing rule keeps the loader consistent across epochs and avoids accidental over- or under-sampling of augmented items.

## Files in this folder

- [hubert_dataset.py](hubert_dataset.py): dataset loading override for batching/indexing behavior.
- [align_mouth_stabilised.py](align_mouth_stabilised.py): preprocessing override related to stabilized mouth crops.

