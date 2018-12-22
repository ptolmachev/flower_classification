# Imports here
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.optim as optim
from torch import nn as nn
import os
from utils import prepare_model, save_model, load_model


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

def train(n_epochs, data_dir, batch_size, num_workers, data_transforms):

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    if not os.path.exists('models'):
        os.makedirs('models')

    image_datasets_train = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    image_datasets_valid = datasets.ImageFolder(root=valid_dir, transform=data_transforms)

    trainloader = torch.utils.data.DataLoader(image_datasets_train, batch_size=batch_size,shuffle=True,
        num_workers=num_workers)
    validloader = torch.utils.data.DataLoader(image_datasets_valid, batch_size=batch_size,num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = prepare_model()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # resnet162
    optimizer = optim.Adam(model._modules['fc'].parameters(), lr=0.001)
    # vgg19_bn
    # optimizer = optim.Adam(list(model.classifier._modules['6'].parameters()) + list(model.classifier._modules['5'].parameters()), lr=0.001)

    model.train()
    best_accuracy = 0.0
    for epoch in range(n_epochs):
        # monitor training loss
        train_loss = 0.0
        for data, target in trainloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        train_loss = train_loss/len(trainloader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1,train_loss))
        model.eval()
        test_loss, accuracy = validation(model, validloader, criterion)
        model.train()
        print('Test Loss: {:.6f} \t Accuracy: {}'.format(test_loss, accuracy))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            print('Saving the model')
            file_save_to = "models/model_" + str(100*np.round(best_accuracy,2)) + ".pth"
            save_model(model.cpu(), file_save_to)
            model = model.to(device)


n_epochs = 10
data_dir = "../flower_data_augmented"
batch_size = 100
num_workers = 8

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transforms= transforms.Compose([transforms.Resize(size = 224),transforms.CenterCrop(224),
                                     transforms.RandomRotation(20),transforms.ToTensor(),
                                     transforms.Normalize(mean,std)])

train(n_epochs, data_dir, batch_size, num_workers, data_transforms)