# flower_classification

### Introduction
This project is aimed to build a flower classificator utilising convolutional neural network trained over 102 classes of diffrent flowers. 

<p align="center">
<img src="https://github.com/ptolmachev/flower_classification/blob/master/img/Flowers.png"/>
</p>
The original dataset can be downloaded here: 

[Flower dataset](https://www.floydhub.com/ptolmachev/datasets/flower_data)
The dataset was kindly provided within [Udacity](https://www.udacity.com/) Pytorch Scholarship Challenge.

### Requirements
In this project I've used **pytorch** python package and GPUs to speed up training. The GPUs are readily available on floydhub (see later for more information) 

### How to use the code
The code is organized in the following fashionL:

The actual code is stored in the folder 'src', which includes:
- `run_classificator.py`- the main file which contains the function for training. 
- `validate.py` - contains the function to get the accuracy on validation dataset
- `utils.py` - contains functions _prepare_model_, which build the specified neural network, _save_ - for saving the obtained weights of neural network, and the function _load_ to embed the saved weights into the network.
- `predict_rundom.py` - takes the random image from the train dataset and predicts the flower depicted in the image.

### How to use floydhub 

### Code walk through and results

#### Augmenting dataset

####  Transfer-learning

#### Conclusins
