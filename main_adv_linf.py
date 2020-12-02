import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#import matplotlib.pyplot as plt
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
from train_util_linf import *
from cifar_input import *

# create logger
logger = logging.getLogger('ECE590_03 Final Project')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception


device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = './data'

train = CIFAR10_Augmented(data_path, train=True, download=True, batch_size=128, shuffle=True, 
                                            num_workers=0, pad=2, image_size=32, flip_rate=0.5)
val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=128, shuffle=False, 
                                    num_workers=0, pad=2, image_size=32, flip_rate=0.5)

## Adversarial training

net = model.ResNetCIFAR(50).to(device)

## Checkpoint name for this model

model_checkpoint = "adv_model_50_linf.pt"

## Basic training params

num_epochs = 20
initial_lr = 0.1
momentum = 0.9
weight_decay = 5e-4
summary_steps = 5

ATK_EPS = 8 / 255
ATK_ITERS = 7
ATK_ALPHA = 2 / 255

do_advtrain = True
do_advtrain_val = False

optimizer = torch.optim.SGD(net.parameters(), initial_lr, momentum, weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1)

logging.basicConfig(filename='adv_train_50_linf.log', level=logging.INFO)
logging.info('Started')

train_model(net, optimizer, scheduler, train, val, device, num_epochs, summary_steps, ATK_EPS, ATK_ITERS, ATK_ALPHA, model_checkpoint, 
    do_advtrain, do_advtrain_val)

logging.info('Finished')