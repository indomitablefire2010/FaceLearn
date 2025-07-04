# FaceLearn

# Enhanced Face Recognition using Transfer Learning (VGG16 and ResNet50)

This project focuses on improving facial recognition performance using transfer learning techniques with VGG16 and ResNet50. It tackles challenges such as limited dataset size, variations in lighting, pose, and facial expressions through effective use of data augmentation and fine-tuning of pre-trained models.

Final Year Minor Project  
Indian Institute of Information Technology, Bhagalpur

## Table of Contents

1. Project Overview  
2. Objectives  
3. Dataset and Preprocessing  
4. Model Architecture  
5. Methodology  
6. Tools and Technologies  
7. Results  
8. Challenges and Solutions  
9. Future Scope  
10. Contributor

## 1. Project Overview

Facial recognition systems are widely used in security, authentication, and human-computer interaction. However, most models struggle with generalization when trained on small datasets. This project enhances face recognition accuracy by leveraging transfer learning with VGG16 and ResNet50, along with data augmentation techniques to improve model generalization.

## 2. Objectives

- Develop a robust facial recognition model using VGG16 and ResNet50.
- Apply transfer learning and fine-tuning to adapt pre-trained models to our dataset.
- Use data augmentation to address overfitting and improve model performance.
- Achieve high accuracy and generalization across variations in pose, lighting, and expression.

## 3. Dataset and Preprocessing

- Collected a labeled facial image dataset with variations in lighting, angle, and expression.
- Preprocessing steps included:
  - Resizing images to 224x224 pixels
  - Normalizing pixel values
  - One-hot encoding the labels

## 4. Model Architecture

VGG16-based model:
- Used pre-trained VGG16 without top classification layers
- Added custom dense layers for classification
- Early layers frozen to retain ImageNet features
- Final layers fine-tuned for task-specific learning

ResNet50-based model:
- Used pre-trained ResNet50 similarly as a feature extractor
- Residual connections allowed better gradient flow
- Final dense layers added and fine-tuned for classification

## 5. Methodology

- Transfer Learning:
  - Loaded pre-trained VGG16 and ResNet50 models
  - Removed fully connected layers and added custom ones
  - Frozen initial layers; fine-tuned later layers
- Data Augmentation:
  - Applied rotation, zoom, flip, shear, and shift
  - Helped increase dataset variability and reduce overfitting
- Training:
  - Used early stopping and learning rate scheduling
  - Evaluated with training-validation split
- Evaluation Metrics:
  - Accuracy, precision, recall, F1-score

## 6. Tools and Technologies

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib and Seaborn  
- OpenCV / PIL  
- Scikit-learn  
- Google Colab for training with GPU support

## 7. Results

VGG16:
- Maximum training accuracy: 81.72 percent
- Validation accuracy: 78.23 percent
- Training loss: 0.5234
- Validation loss: 0.6417

ResNet50:
- Maximum training accuracy: 97.56 percent
- Validation accuracy: 96.78 percent
- Training loss: 0.0932
- Validation loss: 0.1056

Data augmentation significantly improved generalization, especially for underrepresented classes.

## 8. Challenges and Solutions

- Overfitting:
  - Addressed with data augmentation and freezing early layers
- Imbalanced dataset:
  - Handled using class weights and oversampling
- Training stability:
  - Used learning rate tuning and early stopping

## 9. Future Scope

- Expand dataset diversity to improve robustness
- Experiment with newer architectures like Vision Transformers (ViT)
- Apply hyperparameter optimization techniques
- Deploy on edge devices for real-time applications
- Extend to other domains like medical imaging and biometrics

## 10. Contributor

- Nainvi Singh (2101251CS)  
  B.Tech, Computer Science and Engineering  
  Indian Institute of Information Technology, Bhagalpur
