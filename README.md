# Predicting Human Emotions  
Using convolutional neural networks to recognize human facial emotions and determine the relative importance of the upper, lower, right, and left half of one’s face in emotion recognition.  

## Overview:
Facial emotional recognition (FER) is an important and developing field in deep learning, with a wide range of applications, including informing human-computer interaction, mental health awareness, and more. This study aimed to further explore FER using ResNet50 on the FER2013 dataset. In particular, it sought to understand whether FER could be conducted on partial images, by using selective occlusion of the upper, lower, right, and left halves of facial images. The findings indicate that a baseline ResNet50 model performs optimally on full facial images, and that performance significantly deteriorates when attempting to predict emotions on masked images. However, when masked images are incorporated into the training set in fine-tuned models, the performance on the respective masked image (e.g., performance on upper-masked images for a baseline model further fine-tuned on upper-masked images) is comparable, if not marginally superior. The findings of this study indicate that fine-tuning using image augmentation via occlusion may be a promising avenue for the advancement of the field of FER. Additionally, the study’s strengths and limitations are outlined, along with potential future directions for the research area. 


## Methodology:
<img width="1728" alt="Screenshot 2024-04-21 at 9 12 59 PM" src="https://github.com/AaryaDesai1/Facial_Emotion_Recognition/assets/143753050/9a9022bd-6b0b-4869-bef3-f58fbc01d0ca">


