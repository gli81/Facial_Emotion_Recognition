# Predicting Human Emotions  
Using convolutional neural networks to recognize human facial emotions and determine the relative importance of the upper, lower, right, and left half of one’s face in emotion recognition.  

## How to Navigate this Repository: 

## Overview:
Facial emotional recognition (FER) is an important and developing field in deep learning, with a wide range of applications, including informing human-computer interaction, mental health awareness, and more. This study aimed to further explore FER using ResNet50 on the FER2013 dataset. In particular, it sought to understand whether FER could be conducted on partial images, by using selective occlusion of the upper, lower, right, and left halves of facial images. The findings indicate that a baseline ResNet50 model performs optimally on full facial images, and that performance significantly deteriorates when attempting to predict emotions on masked images. However, when masked images are incorporated into the training set in fine-tuned models, the performance on the respective masked image (e.g., performance on upper-masked images for a baseline model further fine-tuned on upper-masked images) is comparable, if not marginally superior. The findings of this study indicate that fine-tuning using image augmentation via occlusion may be a promising avenue for the advancement of the field of FER. Additionally, the study’s strengths and limitations are outlined, along with potential future directions for the research area. 


## Methodology:
As mentioned, we employed a ResNet50 and trained it on the FER2013 dataset to obtain a baseline model. This model was then fine tuned on four types of masked images (upper, lower, right, and left). Each model (including the baseline) was evaluated on its facial emotion recognition performance on five types of images: full facial images, upper-masked images, lower-masked images, right-masked images, and left-masked images. The metrics used to evaluate this performance were: overall accuracy, macro F1-score, weighted average F1-scores, and weighted average AUC score

The following figure gives an overview of the work flow of this project. 

<img width="1466" alt="image" src="https://github.com/AaryaDesai1/Facial_Emotion_Recognition/assets/143753050/c5750b05-11f8-4f7d-93c8-8dc82fd4a1a0">

The code for each step can be found in by navigating to the following links: 
1. [Training the baseline model]()
2. [Finetuning]()
3. 

## Results






