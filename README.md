# Plant Disease Predictor (CNN)

## Overview

This project implements a Convolutional Neural Network (CNN) for multi-label classification of plant leaf diseases using PyTorch. It is based on the [Plant Pathology 2020 - FGVC7 Kaggle competition](https://www.kaggle.com/c/plant-pathology-2020-fgvc7). The notebook demonstrates the workflow from data preprocessing, model building, training, and inference, making it a practical starting point for image-based plant disease classification tasks.

## Features

- Multi-label classification for four categories: `healthy`, `multiple_diseases`, `rust`, and `scab`.
- Image preprocessing using torchvision.
- Custom PyTorch `Dataset` classes for flexible data loading.
- CNN architecture with three convolutional layers and fully connected layers.
- Training loop with Adam optimizer and BCELoss.
- Inference and prediction on test data.
- Model saving for deployment.

## Dataset

- **Source:** [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)
- The dataset consists of annotated leaf images with disease labels.

## How to Use

1. **Clone the Repository**
    ```bash
    git clone https://github.com/MeghP89/Plant_Disease_Predictor_CNN.git
    cd Plant_Disease_Predictor_CNN
    ```

2. **Prepare the Data**
   - Download the dataset from Kaggle and place it in the `input` directory as per notebook instructions.

3. **Run the Notebook**
   - Open `Plant_Disease_Predictor.ipynb` in Jupyter or Colab.
   - Follow the notebook cells to preprocess data, build, train, and evaluate the model.

4. **Export the Model**
   - The trained model will be saved as `Crop_Disease_Classifier.pth`.

## Dependencies

- Python 3.11+
- PyTorch
- torchvision
- pandas
- numpy
- PIL (Pillow)
- kagglehub (for Kaggle integration)

Install dependencies using pip:
```bash
pip install torch torchvision pandas numpy pillow kagglehub
```

## File Structure

- `Plant_Disease_Predictor.ipynb` - Main notebook with complete workflow.
- `README.md` - Project documentation.
- [Other files as needed for data, outputs, etc.]

## Model Architecture

```
Conv2d(3, 16) → ReLU → MaxPool2d
Conv2d(16, 32) → ReLU → MaxPool2d
Conv2d(32, 64) → ReLU → MaxPool2d
Flatten
Linear(64*28*28, 128) → ReLU
Linear(128, 4) → Sigmoid
```

## Training

- Batch size: 4
- Epochs: 7
- Optimizer: Adam
- Loss: Binary Cross Entropy (BCELoss)

## Inference

- Test images are processed and passed through the trained model.
- Predictions are thresholded at 0.5 for each class.

## License

This repository is for educational and research purposes.

## References

- [Kaggle - Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fgvc7)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
