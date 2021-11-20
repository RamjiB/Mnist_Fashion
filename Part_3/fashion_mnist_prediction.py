import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets, models, transforms
import math
import torch.optim as optim

import numpy as np

from model import resnet18



def prediction(image, model_path):
    # Define transforms.
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = resnet18()
    net.load_state_dict(torch.load(model_path, map_location = device)['state_dict'])
    optimizer = optim.Adam(net.parameters())
    net.to(device)
    net.eval()
    image = data_transforms(image.astype(np.float)).reshape((1,1,28,28)).to(device, dtype=torch.float)
    
    # Zero the gradient
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        output = net(image)
        _, pred = torch.max(output, 1)

    return pred.numpy()[0]
    