# Coffee Plantation Image Classification: Ripe vs. Unripe

This project performs a comparative analysis of Machine Learning (Random Forest and SVM) and Deep Learning (CNN) models to classify images of coffee plantations into "Ripe" and "Unripe" categories.

## Project Overview
* **Objective:** To distinguish between ripe and unripe coffee cherries using various computational models.
* **Models Compared:** Random Forest, Support Vector Machine (SVM), and a Convolutional Neural Network (CNN).
* **Data Handling:** The project addresses class imbalances using resampling techniques and class weights.

## Dataset Details
* **Source:** [Drone-based Agricultural Dataset for Crop Yield Estimation](https://huggingface.co/datasets/KaraAgroAI/Drone-based-Agricultural-Dataset-for-Crop-Yield-Estimation) by KaraAgroAI.
* **Structure:** The dataset contains images stored in "Aerial" (5 batches) and "side" (10 batches) folders.
* **Classes:** The primary classes are "Ripe" and "Unripe"; images labeled as "Spoiled" or "Ripening" were removed to prevent model skew.
* **Splitting:** The data is split into Training (70%), Validation (15%), and Test (15%) sets.

## Methodology

### 1. Machine Learning (Random Forest & SVM)
* **Feature Extraction:** The project extracts three types of features:
    * **Color:** Mean and standard deviation of each channel.
    * **Texture:** Local Binary Pattern (LBP).
    * **Shape:** Histogram of Oriented Gradients (HOG).
* **Balancing:** `RandomUnderSampler` is applied to the training data to handle class imbalance.
* **Training:** Both Random Forest and SVM are initialized with `class_weight='balanced'`.

### 2. Deep Learning (CNN)
* **Preprocessing:** Images are resized to 128x128 pixels and normalized to a [0, 1] range.
* **Architecture:** The model utilizes a Sequential architecture with four convolutional blocks, MaxPooling, Dropout layers, and a final sigmoid activation for binary classification.
* **Augmentation:** `ImageDataGenerator` is used for data augmentation, applying rotation, shifting, and zooming to improve generalization.

## Evaluation
* **Metrics:** Models are evaluated based on Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
* **Comparison:** A final summary compares the test set performance of the Random Forest, SVM, and CNN models side-by-side.

## Requirements
Key libraries used in this notebook include:
* `tensorflow` / `keras`
* `sklearn`
* `imblearn`
* `opencv-python` (`cv2`)
* `skimage`
* `numpy`, `pandas`, `matplotlib`, `seaborn`