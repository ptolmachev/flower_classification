# flower_classification

### Introduction

<p align="center">
<img src="https://github.com/ptolmachev/flower_classification/blob/master/img/Flowers.png"/>
</p>

This project was aimed to build a flower classificator on a convolutional neural network trained over 102 classes of flowers. 

The original dataset of flower images can be downloaded here: 

[Flower dataset](https://www.floydhub.com/ptolmachev/datasets/flower_data)

The dataset was kindly provided within [Udacity](https://www.udacity.com/) Pytorch Scholarship Challenge.

In this project I've used **pytorch** python package and GPUs to speed up the training. I've also used a pretrained network Resnet152, the last fully connected layer of which was repurposed for the new task. 

### Structure of the project
The project is organized in the following fashion:

The actual code is stored in the folder 'src', which includes:
- `run_classificator.py`- the main file which contains the training function. 
- `validate.py` - contains the function to get the accuracy on the validation dataset
- `utils.py` - contains functions _prepare_model_, which builds the specified neural network, _save_ - saves the obtained weights of the neural network, and the function _load_ , which embeds the saved weights into the network.
- `predict_rundom.py` - takes the random image from the train dataset and predicts the flower depicted in the image.

The folder 'aux' contains an auxilliary information about classes.
- `cat_to_name.json` - contains the mapping of class number to actual name of the flower.
- `cat_to_names.json` - also contains mapping from class to names: 1) actual name of the class and 2) the google request to obtain relevant images.

### How to use the code
To use the code above, download the full directory 'flower_classification' to your local machine.

Download the dataset ([Flower dataset](https://www.floydhub.com/ptolmachev/datasets/flower_data)) to the folder named 'flower_data' located in the 'flower_classification'.

Also download the weights of a trained model if you want to use the network to predict the class of the presented image:
[model weights (Resnet152)](https://www.floydhub.com/ptolmachev/projects/flower_classification/24/files/models/model_95.0.pth) and place it into the 'models' directory alongside with 'src', 'aux' and 'flower_data' folders.

The model is now ready to use!

### How to use floydhub 
In my own experience, the GPU computes 12 epochs in just one hour, while the same number of epochs on CPU will take around 15 hours. 

To speed up the training of the CNN, I've used the cloud GPU available on the [floydhub](https://www.floydhub.com/).

The service is very straigtforward to use: there is always good documentation and plenty of hints, so you will not get lost! 

At first, you need to create your project. To run your job, install the floydhub client on you machine, upload the dataset to their website and run the following command in your terminal:

`floyd run --gpu --env pytorch-1.0 --data username/datasets/flower_data/i:flower_data 'python run_classificator.py'`

where 'username' is your login name, and 'i' is the folder where the dataset is located.

## Code walkthrough and the results

#### Augmenting dataset
To augment the dataset I've downloaded the relevant images from the Internet using 'google_images_download' API.
```python
from google_images_download import google_images_download as gid   #importing the library
...
response = gid.googleimagesdownload()   #class instantiation
for i in range(NUM_OF_LABELS):

    folder_name = str(i + 1)
    request = cat_to_names[folder_name][1]
    
    #creating list of arguments
    arguments = {"keywords" : request, 
                 "limit" : 100, 
                 "print_urls" : False, 
                 "output_directory" : './downloads/', 
                 "image_directory" : folder_name
                } 
    paths = response.download(arguments)   #passing the arguments to the function
    print(paths)   #printing absolute paths of the downloaded images
```
Make sure to pick off the outliers from the images (broken files or irrelevant images which got there just by accident). 

#### Standardization of images
To make the images have the same size and pixel variation, one may use transfors module from pytorch:
```python
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms= transforms.Compose([transforms.Resize(size = 224),transforms.CenterCrop(224),
                                     transforms.RandomRotation(20),transforms.ToTensor(),
                                     transforms.Normalize(mean,std)])

```

####  Transfer-learning
Training your own model from the scratch may not be the wisest option, as there are plenty of networks available trained on various datasets. Intuitively, the low-level features like edge- and other simple shapes detectors are similar for the different models even if the clasificators were trained for the different purposes. 

In fact, in this project, I've trained only the last layer of the pretrained classificator, and even that gives quite good results!

Here is the example how to use the pretrained model, readily available via torchvision and pytorch: 

```python

import torch
from torch import nn as nn
import torchvision.models as models

def prepare_model():
    model = models.resnet152(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    model._modules['fc'] = nn.Linear(2048, 102)
    model._modules['fc'].requires_grad = True
    return model
```


#### Training the model
The fucntion which does the training is presented below:

```python
def train(n_epochs, data_dir, batch_size, num_workers, data_transforms):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # defining the datasets for training and validation with the specified set of transformations
    image_datasets_train = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    image_datasets_valid = datasets.ImageFolder(root=valid_dir, transform=data_transforms)
    
    #defining the generator objects to get the batches of standardized images from
    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=batch_size,shuffle=True,
        num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=batch_size,
        num_workers=num_workers)
    
    # get the device to train your model at (pgu or cpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # building your neural network
    model = prepare_model()
    # send the network to specified device (gpu if it's available)
    model = model.to(device)
    
    # defining the cost function
    criterion = nn.CrossEntropyLoss()
    
    # spesifying the optimization method
    optimizer = optim.Adam(model._modules['fc'].parameters(), lr=0.001)
    
    # set your model into the training regime
    model.train()
    
    # training loop
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        for data, target in trainloader:
            # get batch of images
            data = data.to(device)
            
            # get the correct labels for the images
            target = target.to(device)
            
            # clean the buffer from previous gradients
            optimizer.zero_grad()
            
            # get the predicted lables
            output = model(data)
            
            #compute the objective function
            loss = criterion(output, target)
            
            # compute the gradients with respect to the network weights
            loss.backward()
            
            # make an update of weights
            optimizer.step()
            
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss/len(trainloader.dataset)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch+1,train_loss))
        
        # set your model in evaluation regime (disables dropout, the gradients are not computed)
        model.eval()
        test_loss, accuracy = validation(model, validloader, criterion)
        model.train()
        print('Test Loss: {:.6f} \t Accuracy: {}'.format(test_loss, accuracy))
        
        # save your model if it's better then the one before!
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print('Saving the model')
            file_save_to = "models/model_" + str(100*np.round(best_accuracy,2)) + ".pth"
            save_model(model.cpu(), file_save_to)
            model = model.to(device)
```

#### Results and ideas for improvement
The resulting model was trained for 1.5 hour on the dataset and achieved **95% accuracy** on the validation dataset: it is pretty accurate, but, surely, one can always do better!

- To improve the accuracy, one can apply the gradient update not only to the last fully-connected layer of the network but to the several layers preceding the final output.

- If you got plenty of time and free GPU you may design your own network and train it from scratch for ~30-50 epochs to get the network specifically tailored for your need.

- Getting more clean data will also result in better performance!
