import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import logging
import sys
import time
import os
import copy

# Custom
import model
import attacks
from cifar_input import *

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = './data'

# Test on clean datasest

val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=128, shuffle=False, 
                                    num_workers=0, pad=2, image_size=32, flip_rate=0.5)

net = model.ResNetCIFAR(50).to(device)
net.load_state_dict(torch.load("r_model_norm.pt"))

running_loss = 0.0
running_corrects = 0

net.eval()

for val_inputs, val_labels in val.trainloader:

    val_inputs = val_inputs.to(device)
    val_labels = val_labels.to(device)

    #optimizer.zero_grad()

    with torch.set_grad_enabled(False):

        # forward
        # track history if only in train
        outputs = net(val_inputs) 

        _, preds = torch.max(outputs, 1)

        loss = F.cross_entropy(outputs, val_labels)

    running_loss += loss.item() * val_inputs.size(0)
    running_corrects += torch.sum(preds == val_labels.data)

epoch_loss = running_loss / len(val.trainset)
epoch_acc = running_corrects.cpu().numpy() / len(val.trainset)

print('Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch_loss, epoch_acc))


# Test adversarial attack on standard model

val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=128, shuffle=False, 
                                    num_workers=0, pad=2, image_size=32, flip_rate=0.5)

net = model.ResNetCIFAR(50).to(device)
net.load_state_dict(torch.load("r_model_norm.pt"))

ATK_EPS = 0.25
ATK_ITERS = 7
ATK_ALPHA = 0.05

running_loss = 0.0
running_corrects = 0

net.eval()

for val_inputs, val_labels in val.trainloader:

    val_inputs = val_inputs.to(device)
    val_labels = val_labels.to(device)

    # Generate adversarial training examples
    
    attack = attacks.PGDAttack(net, ATK_EPS, ATK_ITERS, ATK_ALPHA, rand=True)
    val_inputs = attack.perturb_l2_v2(val_inputs, val_labels)

    #optimizer.zero_grad()

    with torch.set_grad_enabled(False):

        # forward
        # track history if only in train
        outputs = net(val_inputs) 

        _, preds = torch.max(outputs, 1)

        loss = F.cross_entropy(outputs, val_labels)

    running_loss += loss.item() * val_inputs.size(0)
    running_corrects += torch.sum(preds == val_labels.data)

epoch_loss = running_loss / len(val.trainset)
epoch_acc = running_corrects.cpu().numpy() / len(val.trainset)

print('Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch_loss, epoch_acc))
