# FoodVision:

This project, named "FoodVision" (also known as "Culinary Classifier"), demonstrates the process of building and training a convolutional neural network (CNN) for image classification using PyTorch. The goal is to classify images of pizza, steak, and sushi.

## Project Overview

The notebook walks through the following steps:

1.  **Data Acquisition:** Downloading and preparing a subset of the Food101 dataset containing images of pizza, steak, and sushi.
2.  **Data Preparation:** Exploring and visualizing the dataset, including transforming images into tensors using `torchvision.transforms`.
3.  **Data Loading:** Creating PyTorch `Dataset` and `DataLoader` objects to efficiently load and batch the image data. This includes exploring both the built-in `ImageFolder` and a custom `Dataset` implementation.
4.  **Data Augmentation:** Implementing data augmentation techniques (`TrivialAugmentWide`) to increase the diversity of the training data and improve model generalization.
5.  **Model Definition:** Creating a simple CNN model architecture called "TinyVGG" using PyTorch's `nn.Module`.
6.  **Training and Evaluation:** Setting up training and testing loops, defining loss functions (`CrossEntropyLoss`) and optimizers (`Adam`), and training the TinyVGG model with and without data augmentation.
7.  **Model Comparison:** Comparing the performance of the models trained with and without data augmentation by plotting their loss and accuracy curves.
8.  **Prediction on Custom Image:** Demonstrating how to load a custom image and use the trained model to make a prediction.

## Tech Stack and Flow

*   **Implemented Convolutional Neural Network (CNN):** Specifically, a TinyVGG architecture for classifying images into three classes: pizza, steak, and sushi.
*   **PyTorch and torchvision:** Used as the primary libraries for building, training, and evaluating the deep learning models, handling datasets, and applying image transformations.
*   **DataLoader:** Employed for efficient batching and loading of image data during training and evaluation.
*   **torch.nn:** Used to define the layers and architecture of the TinyVGG model.
*   **Adam Optimizer:** Utilized for optimizing the model's parameters during training.
*   **CrossEntropyLoss:** Used as the loss function for the multi-class classification task.
*   **Matplotlib:** Used for visualizing images, training curves (loss and accuracy), and prediction results.
*   **Pandas:** Used for organizing and comparing the training and evaluation results of different models.
*   **torchinfo:** Used to display a summary of the model architecture and the shape of tensors at each layer.
*   **Python:** The entire project is implemented in Python.

## How to Run the Notebook

1.  Ensure you have a Google Colab environment or a local Python environment with the necessary libraries installed (PyTorch, torchvision, matplotlib, pandas, torchinfo, requests, zipfile, pathlib, PIL).
2.  Run the code cells sequentially in the notebook.
3.  The notebook will automatically download the dataset and proceed with training and evaluation.

## Results

The notebook compares the performance of the TinyVGG model trained with and without data augmentation. The plots of the loss and accuracy curves show the impact of data augmentation on the model's ability to generalize to unseen data.

## Future Work

*   Experiment with different CNN architectures (e.g., ResNet, VGG).
*   Train on the full Food101 dataset.
*   Implement more advanced data augmentation techniques.
*   Explore transfer learning by using pre-trained models.
*   Build a user interface for classifying custom images.
