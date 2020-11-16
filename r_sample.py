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
import pickle

# Custom
import model
import attacks
from train_util import *
from cifar_input import *

# create logger
logger = logging.getLogger('ECE590_03 Final Project Creating Robust Samples')
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

# logging.basicConfig(filename='create_rsamples.log', level=logging.INFO)
# logging.info('Started')

device = "cuda" if torch.cuda.is_available() else "cpu"

net = model.ResNetCIFAR(50).to(device)
net.load_state_dict(torch.load("adv_model_50.pt"))

model = Dr_model(net).cuda()

data_path = './data'

def get_norm_batch(x, p):
	batch_size = x.size(0)
	return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

data = CIFAR10_Raw(data_path, train=True, download=True, batch_size=128, shuffle=False)

def r_sample(model, max_epoch, step_size, start, end):

    logging.basicConfig(filename='create_rsamples' + str(start) + '_' + str(end) + '.log', level=logging.INFO)
    logging.info('Started')

    Dr = []

    for j, (inputs, labels) in enumerate(data.trainset):
        
        if j >= start and j < end:
        
            sample_label = labels

            while sample_label == labels:
                rand_ind = np.random.randint(0, 50000)
                #print(rand_ind)
                s_inputs, s_labels = data.trainset[rand_ind]
                sample_label = s_labels

            inputs = inputs.reshape(1,3,32,32).cuda()
            labels = torch.tensor(labels).cuda()

            s_inputs = s_inputs.reshape(1,3,32,32).cuda()
            s_labels = torch.tensor(s_labels).cuda()

            s_inputs.requires_grad_()
            
            outputs = model(inputs) 
            outputs = outputs.clone().detach()
            
            for i in range(max_epoch):
                s_inputs = s_inputs.clone().detach().requires_grad_(True)
                s_outputs = model(s_inputs)
                diff = s_outputs - outputs
                l_diff = len(diff.shape) - 1
                loss = torch.norm(diff.view(diff.shape[0], -1), dim=1).view(-1, *([1]*l_diff))

                loss.backward()

                l_s = len(s_inputs.shape) - 1
                grad = s_inputs.grad.data
                g_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, *([1]*l_s))
                scaled_g = grad / (g_norm + 1e-10)
    #                 if i % 50 == 0:
    #                     print(normalized_grad)
                s_inputs = s_inputs.clone().detach() - step_size * scaled_g
            
                s_inputs = torch.clamp(s_inputs.clone().detach(), 0 ,1)
                
            Dr.append((s_inputs.clone().detach(), labels.clone().detach()))

            if j % 100 == 0:
                logging.info("Completed {} samples".format(j))
                
            #print("final loss:", loss)
            
            with open("Dr_" + str(start) + "_" + str(end) + ".txt", "wb") as f:   #Pickling
                pickle.dump(Dr, f)
    
#Dr = r_sample(model, 50000, 0.0005)
r_sample(model, 1000, 0.1, 0, 1000)

logging.info('Finished')