# Banana Leaf Disease Detection

## Overview

A computer vision project that classifies banana leaf images into seven categories using deep learning.

## Disease Classes

- Banana Black Sigatoka Disease
- Banana Bract Mosaic Virus Disease
- Banana Healthy Leaf
- Banana Insect Pest Disease
- Banana Moko Disease
- Banana Panama Disease
- Banana Yellow Sigatoka Disease

## Dataset

- Total images: 408
- Training images: 283
- Validation images: 60
- Test images: 65
- Image dimensions: 512 × 512 pixels

The dataset is split into training, validation and testing subsets. The generated dataset folders are excluded from this repository.

## Models Implemented

### 1. Custom CNN
A convolutional neural network trained from scratch.

### 2. MobileNetV2
A transfer-learning model using pretrained ImageNet features.

## Test Results

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 66.15% | 81.54% |
| Macro Precision | 66.95% | 79.18% |
| Macro Recall | 67.49% | 76.28% |
| Macro F1-score | 65.95% | 75.61% |

MobileNetV2 was selected as the final model based on its better test-set performance.

## Project Files

- `dataset_analysis.py` — dataset analysis and image quality checks
- `prepare_dataset.py` — creates training, validation and test splits
- `check_split.py` — verifies split counts
- `cnn_model.py` — Custom CNN implementation
- `evaluate_cnn.py` — Custom CNN evaluation
- `mobilenet_model.py` — MobileNetV2 implementation
- `evaluate_mobilenet.py` — MobileNetV2 evaluation
- `model_comparison.py` — compares evaluation metrics
- `predict.py` — predicts a disease class for an input image

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place the original dataset in the expected directory, then run:

```bash
python dataset_analysis.py
python prepare_dataset.py
python check_split.py
```

Train the models using their respective scripts. Run the evaluation scripts after the corresponding model files have been saved.

## Limitations

Performance depends on the dataset and image quality. The model showed weaker recall for Yellow Sigatoka Disease, and some Black Sigatoka images were misclassified. More diverse images and further validation are needed before real-world deployment.

## Technologies

Python, TensorFlow/Keras, NumPy, Matplotlib, scikit-learn and Seaborn.
