from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from torchvision import transforms
import torch
from utils import load_model, prepare_model
from torch.nn import functional as F

def process_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([transforms.Resize(size = 224),transforms.CenterCrop(224),
                                          transforms.ToTensor(),transforms.Normalize(mean,std)])
    return data_transforms(image)

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    plt.show()
    return ax

def predict(image, model, cat_to_name, topk=5):
    img_tensor = torch.from_numpy(np.asarray(image)).unsqueeze(0)
    model.eval()
    logits = model(img_tensor)
    probs = F.softmax(logits, dim = 1).detach().numpy()
    probs = np.squeeze(probs)

    nums = range(len(probs))
    #these three lines needed to fix incorrect mapping issue:
    nums = sorted(cat_to_name, key=lambda tup: tup[0])
    nums = sorted(nums)
    nums = [int(x) for x in nums]

    probs_and_classes = list(zip(probs,nums))
    probs_and_classes = sorted(probs_and_classes, key=lambda tup: tup[0])[::-1]
    probs = [x[0] for x in probs_and_classes]
    classes = [cat_to_name[str(x[1])] for x in probs_and_classes]
    return probs[:topk], classes[:topk]


model = load_model("../models/model_95.0.pth")
model.eval()
cat_to_name = json.load(open("../aux/cat_to_name.json"))
path = '../flower_data_augmented/train/{}/'.format(np.random.randint(1,102))
rand_img_path = path + str(os.listdir(path)[np.random.randint(len(os.listdir(path)))])
rand_pil_img = Image.open(rand_img_path)
image = process_image(rand_pil_img)
actual_class = cat_to_name[rand_img_path.split('/')[3]]
imshow(image)
print(predict(image, model, cat_to_name, topk=3))
print("Actual class: {}".format(actual_class))