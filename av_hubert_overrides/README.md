# AV-HuBERT Overrides

This folder contains local AV-HuBERT behavior overrides used to stabilize data loading and batching when mixing base and augmented samples.

## Batching fix summary

The batching issue came from how augmentation sampling interacted with dataset indexing.

1. A training request first decides whether augmentation is applied.
2. If augmentation is selected, one augmentation source is chosen.
3. Effective probabilities become:
: base sample 50%, augmentation source 1 sample 25%, augmentation source 2 sample 25%.

This makes the intended sampling distribution explicit and prevents accidental over/under-sampling of augmented data.

![Batching probability flow](../figures/batching_2_aug.png)

## Pairing/index fix

The second part of the fix ensures that base and augmented entries remain aligned by key identity (for example the same utterance stem), while augmented variants use a deterministic suffix pattern.

This avoids collisions and keeps loader behavior consistent across epochs.

![Base to augmented pairing](../figures/two_aug_batching.drawio.png)

## Files in this folder

- [hubert_dataset.py](hubert_dataset.py): dataset loading override for batching/indexing behavior.
- [align_mouth_stabilised.py](align_mouth_stabilised.py): preprocessing override related to stabilized mouth crops.

