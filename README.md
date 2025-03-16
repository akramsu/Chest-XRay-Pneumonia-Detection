# Chest X-Ray Pneumonia Detection with SVM

## Overview
This project implements a model to classify chest X-ray images into two categories: **Normal** and **Pneumonia**. Using the **Chest X-Ray Images (Pneumonia)** dataset, the model utilizes a **Support Vector Machine (SVM)** classifier to predict pneumonia based on features extracted using **Principal Component Analysis (PCA)** for dimensionality reduction.

## Dataset
The dataset used for this project is the **Chest X-Ray Images (Pneumonia)** dataset, which consists of chest X-ray images categorized into **Normal** and **Pneumonia** classes. It can be found on Kaggle:
- Dataset link: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Dataset Organization
The dataset is divided into three sets:
- **Train**: Contains images for training the model.
- **Validation**: Used to validate the model during training.
- **Test**: Used to evaluate the model's performance after training.

The project processes and analyzes images using Python libraries such as OpenCV, scikit-learn, and PCA.

## Key Steps
1. **Data Preprocessing**: The images are resized and converted to grayscale.
2. **Dimensionality Reduction**: PCA is applied to reduce the features and retain the most significant components.
3. **Model Training**: An SVM classifier with an RBF kernel is trained using the PCA-transformed data.
4. **Model Evaluation**: The model's performance is evaluated using accuracy, classification report, and confusion matrix.
5. **Cross-Validation**: 5-fold cross-validation is used to assess the model's generalizability.

## Installation

To run this project, you will need Python and several dependencies. You can install them using `pip`:

1. Clone the repository:
   ```bash
   git clone https://github.com/akramsu/Chest-XRay-Pneumonia-Detection.git
   cd Chest-XRay-Pneumonia-Detection
