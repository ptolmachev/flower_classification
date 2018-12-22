# Imports here
import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models
import json
import torch.nn as nn #from torch import nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch import nn as nn
from PIL import Image
import os

def prepare_model():
    # model = models.vgg19_bn(pretrained=True)
    model = models.resnet152(pretrained = True)
    i = 0
    # model.classifier._modules['6'] = nn.Linear(4096, 102)
    model._modules['fc'] = nn.Linear(2048, 102)
    for param in model.parameters():
        param.requires_grad = False
        i += 1
    model._modules['fc'].requires_grad = True

    # layers = [param for param in model.classifier.parameters()]
    # layers[4].require_grad = True
    # layers[5].require_grad = True
    return model

def load_model(load_from):
    model = prepare_model()
    info = torch.load(load_from, map_location=lambda storage, loc: storage)
    state_dict = info['state_dict']
    model.load_state_dict(state_dict)
    return model

def validation(model, validloader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loss = 0.0
    correct = 0
    total = 0
    i = 0
    for data, target in validloader:
        data = data.to(device)
        target = target.to(device)
        output = model(data).detach()
        loss = criterion(output, target).detach()
        output = output.cpu().detach().numpy()
        prediction = np.argmax(output, axis=1)
        target = target.cpu().detach().numpy()
        correct += np.sum((prediction == target) * 1.0)
        total += len(target)
        test_loss += loss.item() * data.size(0)
        i+=1
        print("Batch {} \t Accuracy: {}".format(i, float(correct) / float(total)))
    test_loss = test_loss / len(validloader.dataset)
    accuracy = float(correct) / float(total)
    return test_loss, accuracy



#test
data_dir = '../flower_data'
valid_dir = data_dir + '/valid'

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms = transforms.Compose([transforms.Resize(size = 224),transforms.CenterCrop(224),
                                          transforms.ToTensor(),transforms.Normalize(mean,std)])

image_datasets_valid = datasets.ImageFolder(root=valid_dir, transform=data_transforms)

validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=100,num_workers=2)

with open('cat_to_names.json', 'r') as f:
    cat_to_name = json.load(f)

classes = [x[1][0] for x in cat_to_name.items()]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = prepare_model()
model = load_model("../models/model_93.0.pth")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
model.eval()
print(validation(model, validloader, criterion))