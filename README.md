# Data Augmentation for AVSR model

This repo outlines the code and notes outlining the research completed exploring unique ways to augment training data to improve model performance in Audio Visual Speech Recognition Models. The model chosen for this experiment was AV-Hubert.

## Dataset Used

TCD Timit - 3 Lipspeakers and 49 Volunteers

## Model Used

Av-Hubert from Meta

## Libraries

## Augmentation Techniques

### Inbetweening 

This technique entails upsampling the raw video from 25fps to 50fps by inserting an artifial frame inbetween each real frame using ffmpeg. The upsampled video is then downsampled back to 25fps only taking the artificial frames. The audio is then aligned by shifting the audio the same amount as the video (1/50).

