# Cats vs Dogs - Convolutional Neural Network

This project implements a Convolutional Neural Network (CNN) to perform binary image classification of the **cats vs. dogs** dataset from Kaggle using TensorFlow / Keras. The goal is to demonstrate a full experiment pipeline: data loading, augmentation, model building, training and evaluation. Furthermore, by applying Transfer Learning, we can compare our results with those obtained from a pre-trained network such as MobileNetV2, which is available in the Keras Applications module.

## Objectives
- Build a simple CNN classifier for cats and dogs images.
- Use `ImageDataGenerator` to load and augment images from directory structure.
- Train and evaluate the model (binary classification).
- Compare different models using Transfer Learning.

## Requirements
Minimum recommended environment:
- Python 3.8+  
- TensorFlow 2.x  
- numpy  
- matplotlib  
- pillow (PIL)  
- jupyterlab or notebook

## Results
1. The scan_corrupted_image function has found a few corrupted files, so we can now take a look at a labelled subset of the image dataset:
<img width="281" height="54" alt="image" src="https://github.com/user-attachments/assets/03c9d544-60cc-47fa-b6e4-d36a75a30b3b" />
<br>
<img width="559" height="130" alt="image" src="https://github.com/user-attachments/assets/bd09c586-3552-419e-b6b4-ed2cf04722d6" />

2. This is the architecture for our simple CNN. 
<img width="527" height="336" alt="image" src="https://github.com/user-attachments/assets/e8928205-b7e4-4c73-8585-1d2d142febe8" />

3. After training our network for 10 epochs we obtained a very high value for the accuracy, so let's hope the model hasn't overfitted.
<img width="512" height="323" alt="image" src="https://github.com/user-attachments/assets/8ce4da00-ab55-4f15-b609-d3ed9b74ba47" />

4. Finally, here we have our results on the test set. Surprisingly, the model performs very well and has not overfitted the test data!
<img width="172" height="15" alt="image" src="https://github.com/user-attachments/assets/76ce8406-1160-442f-b3cd-73c0d6a50fde" />

## Transfer Learning
With transfer learning we can repurpose a model which has already been trained on a much larger amount of data. This is key when working in projects with limited data and for time and cost reduction. 
In this example we have imported the MobileNetV2 model.
<img width="607" height="426" alt="image" src="https://github.com/user-attachments/assets/bdbb7163-29d5-4c5b-b4b9-c4a92938549a" />
<br>
With just 5 epochs of training it has already achieved a higher accuracy on the training set, and after testing, has also improved the accuracy on the test set.
<br>
<img width="174" height="15" alt="image" src="https://github.com/user-attachments/assets/b0bde5e7-2b2c-4390-a1b6-bf76dfd83d6f" />
