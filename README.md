# fashion-mnist-pytorch

Implemented a convolutional neural network using PyTorch to classify different images from the Fashion-MNIST dataset. 

Achieved **88.57%** test accuracy through architecture adjustments and hyperparameter tuning.


## Requirements
Install all dependencies with: 
```bash  
pip install -r requirements.txt
```

## Dataset
This project uses the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist), which is a 10 category collection of 70,000 grayscale clothing images (28x28 pixels).

Manually downloading the dataset is not needed, as it is automatically downloaded by torchvision.datasets.FashionMNIST when src/data.py is run for the first time. 


## Project Structure 
```text
.
├── src
│   ├── data.py      # dataset loading
│   ├── demo.py      # gradio demo
│   ├── model.py     # CNN architecture
│   └── train.py     # training loop and logging
├── requirements.txt
└── README.md
```


## Demo(Hosted on Hugging Face)
[Try it here](https://huggingface.co/spaces/azeng123/fashionMNISTdemo)

