import torch
from torch import nn as nn
import torchvision.models as models

def prepare_model():
    # model = models.vgg19_bn(pretrained=True)
    model = models.resnet152(pretrained = True)
    i = 0
    for param in model.parameters():
        param.requires_grad = False
        i += 1
    model._modules['fc'] = nn.Linear(2048, 102)
    model._modules['fc'].requires_grad = True
    # model.classifier._modules['6'] = nn.Linear(4096, 102)
    # layers = [param for param in model.classifier.parameters()]
    # layers[5].require_grad = True
    return model

def save_model(model, save_to):
    info = dict()
    info['state_dict'] = model.state_dict()
    torch.save(info, open(save_to, 'wb+'))

def load_model(load_from):
    model = prepare_model()
    info = torch.load(load_from,map_location='cpu')
    state_dict = info['state_dict']
    model.load_state_dict(state_dict, strict = False)
    return model
