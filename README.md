# fashion-mnist-pytorch
Implemented a convolutional neural network using PyTorch to classify different images from the Fashion-MNIST dataset. 

Achieved **98%** test accuracy through model tuning and hyperparameter experimentation.


# Requirements
Install all dependencies with: **pip install -r requirements.txt**


# Dataset
This project uses the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) dataset, which is a 10 category collection of 70,000 grayscale clothing images (28x28 pixels).

Manually downloading the dataset is not needed, as it is automatically downloaded by torchvision.datasets.FashionMNIST when src/data.py is run for the first time. 
