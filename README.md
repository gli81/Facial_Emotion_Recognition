# Predicting Human Emotions  
Using convolutional neural networks to recognize human facial emotions and determine the relative importance of the upper, lower, right, and left half of one’s face in emotion recognition.  

## How to Navigate this Repository: 

This repository, originally forked from a Duke University Machine Learning class project by team Iguana, focuses on using ResNet50 for facial emotion recognition on full and partial images from the FER2013 dataset. The contents include:
```
|
| - classifier (PyTorch)
    | - data: training dataset
    | - saved_models: trained ResNet models saved as .pth files
    | - custom_transforms.py: defines custom transformation for preprocessing the data
    | - dataset.py: defines custom torch.utils.data.Dataset class to transform our images to tensors torch can easily sample from.
    | - evaluate_.py: defines various evaluation metrics
    | - finetune_*_masked.ipynb: partial face models training
    | - hyperparameters: defines hyperparameters for training
    | - loaders.py: defines torch.utils.data.DataLoader to form batches and load data samples
    | - training.py: defines the whole training process
| - web
    |
    | - frontend: frontend code (React)
    | - inference
        | - saved_models: trained ResNet models saved as .pth files
        | - src: API for model inference (Python Flask)
    | - interface: API that handles webpage user actions (Go)
```

## How to use

To run the frontend

```shell
cd web/frontend
npm i
npm run dev
```

To run the interface
```shell
cd web/interface
go run main.go
```

To run the inference API
```shell
cd web
flask --app inference.src run
```

## Abstract:
Facial emotional recognition (FER) is an important and developing field in deep learning, with a wide range of applications, including informing human-computer interaction, mental health awareness, and more. This study aimed to further explore FER using ResNet50 on the FER2013 dataset. In particular, it sought to understand whether FER could be conducted on partial images, by using selective occlusion of the upper, lower, right, and left halves of facial images. The findings indicate that a baseline ResNet50 model performs optimally on full facial images, and that performance significantly deteriorates when attempting to predict emotions on masked images. However, when masked images are incorporated into the training set in fine-tuned models, the performance on the respective masked image (e.g., performance on upper-masked images for a baseline model further fine-tuned on upper-masked images) is comparable, if not marginally superior. The findings of this study indicate that fine-tuning using image augmentation via occlusion may be a promising avenue for the advancement of the field of FER. Additionally, the study’s strengths and limitations are outlined, along with potential future directions for the research area. 


## Methodology:
As mentioned, we employed a ResNet50 and trained it on the FER2013 dataset to obtain a baseline model. This model was then fine tuned on four types of masked images (upper, lower, right, and left). Each model (including the baseline) was evaluated on its facial emotion recognition performance on five types of images: full facial images, upper-masked images, lower-masked images, right-masked images, and left-masked images. The metrics used to evaluate this performance were: overall accuracy, macro F1-score, weighted average F1-scores, and weighted average AUC score

The following figure gives an overview of the work flow of this project. 

<img width="1466" alt="image" src="https://github.com/AaryaDesai1/Facial_Emotion_Recognition/assets/143753050/c5750b05-11f8-4f7d-93c8-8dc82fd4a1a0">

## Results
The results of training the baseline and subsequent fine-tuned models (the baseline model was additionally trained on full face, upper-, lower, right-, or left-masked images) and evaluating their performance on full facial, upper-, lower-, right-, and left-masked images are presented. All models demonstrated the highest performance on full facial images; therefore, the best performances on masked images are also highlighted in the table. The metrics employed were overall accuracy, macro F1 scores, weighted average F1 score (W.A. F1 score), and weighted average AUC score (W.A. AUC).
<img width="665" alt="image" src="https://github.com/AaryaDesai1/Facial_Emotion_Recognition/assets/143753050/51fcd9f1-8f52-4f75-aaad-c3c33dc07160">

## Conclusion: 
Overall, the study demonstrates that while full-face images yield the best FER performance, there is substantial merit in exploring and improving techniques for FER from partially occluded faces. This is contingent on the implementation of appropriate training adjustments and model enhancement. For a more extensive discussion of the results, strengths, and limitations, please refer to the [final report](https://github.com/AaryaDesai1/Facial_Emotion_Recognition/blob/main/04_docs/Final_Report_-_ML_Team_Iguana_.pdf)




