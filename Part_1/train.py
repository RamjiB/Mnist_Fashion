import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import torch.optim as optim

from model import resnet18
import time
import sys, os
import math
import numpy as np


def run_model(net, loader, criterion, optimizer, train = True):
    running_loss = 0
    running_accuracy = 0
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    net.to(device)
    # Set mode
    if train:
        net.train()
    else:
        net.eval()


    for i, (X, y) in enumerate(loader):
        # Pass to gpu or cpu
        y = y.type(torch.LongTensor)
        X, y = X.to(device, dtype=torch.float), y.to(device)

        # Zero the gradient
        optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            output = net(X)
            _, pred = torch.max(output, 1)
            loss = criterion(output, y)

        # If on train backpropagate
        if train:
            loss.backward()
            optimizer.step()

        # Calculate stats
        running_loss += loss.item()
        running_accuracy += torch.sum(pred == y.detach())
    return running_loss / len(loader), running_accuracy.double() / len(loader.dataset)

class Dataset():
    def __init__(self, images, labels, data_transforms):
        self.transforms = data_transforms
        # load all image files
        self.images = images
        self.labels = labels
        
    def __getitem__(self, idx):
        image = self.images[idx]
        label = torch.tensor(self.labels[idx])
#         y = np.zeros((10))
#         y[label] = 1
        image = self.transforms(image.astype(np.float))
#         print(image.shape,label)
        return image, label

    def __len__(self):
        return len(self.images)

def save_model(model_state, filename):
        """ Save model """
        # TODO: add it as checkpoint
        torch.save(model_state,filename)


def train(x_train, y_train, x_test, y_test, epochs=50, model_folder_path = 'models/'):

    if (len(x_train.shape) != 3) or (len(x_test.shape) != 3) or (len(y_train.shape) != 1) or (len(y_test.shape) != 1):
        print('Exception: Input image or Label of train/test is not as per requirement')
    else:
        # Define transforms.
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        try:
            os.makedirs(model_folder_path)
        except Exception as e:
            print(e)

        net = resnet18()
        # Train Dataset loader for pytorch model
        train_dataset = Dataset(x_train, y_train, data_transforms)
        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

        test_dataset = Dataset(x_test, y_test, data_transforms)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4)
        criterion = torch.nn.CrossEntropyLoss()
        # Define optimizer
        optimizer = optim.Adam(net.parameters())
        num_epochs = epochs
        patience = 3
        best_loss = 1e4

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        for e in range(num_epochs):
            start = time.time()
            train_loss, train_acc = run_model(net, train_data_loader,
                                        criterion, optimizer)
            val_loss, val_acc = run_model(net, test_data_loader,
                                        criterion, optimizer, False)
            end = time.time()

            # print stats
            stats = """Epoch: {}\t train loss: {:.3f}, train acc: {:.3f}\t
                    val loss: {:.3f}, val acc: {:.3f}\t
                    time: {:.1f}s""".format(e+1, train_loss, train_acc, val_loss,
                                            val_acc, end - start)
            print(stats)
            if val_loss < best_loss:
                best_loss = val_loss
                patience = patience
                save_model({
                    'arch': 'Resnet-18',
                    'state_dict': net.state_dict()
                }, '{}-epoch-{}.pth'.format(model_folder_path + '/Resnet-18', e))
            else:
                patience -= 1
                if patience == 0:
                    print('Run out of patience!')
                    break


    