import attacks

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import random
import sys
import time
import os
import copy

import logging

def train_model(net, optimizer, scheduler, train, val, device, num_epochs, summary_steps, ATK_EPS, ATK_ITERS, ATK_ALPHA, model_dir, 
    do_advtrain, do_advtrain_val):

    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    for i in range(num_epochs):

        net.train()

        running_loss = 0.0
        running_corrects = 0
        ii = 0 

        for inputs, labels in train.trainloader:

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            #inputs = inputs + np.random.uniform(-epsilon, epsilon, inputs.size)
            #inputs = np.clip(inputs, 0, 255) # ensure valid pixel range

            # Generate adversarial training examples
            if do_advtrain:
                attack = attacks.PGDAttack(net, ATK_EPS, ATK_ITERS, ATK_ALPHA, rand=True)
                inputs = attack.perturb_l2_v2(inputs, labels)

            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                # forward
                # track history if only in train
                outputs = net(inputs) 

                _, preds = torch.max(outputs, 1)

                loss = F.cross_entropy(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if ii % 50 == 0:
                logging.info("{}, running loss: {:.4f}, running corrects: {:.4f}".format(ii, running_loss, running_corrects))
            
            ii += 1

        scheduler.step()

        
        logging.info("{} epoch train completed".format(i))
        logging.info("====================================")

        if i % summary_steps == 0:

            epoch_loss = running_loss / len(train.trainset)
            epoch_acc = running_corrects.cpu().numpy() / len(train.trainset)

            logging.info('{} Train Loss: {:.4f} Trian Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

            running_loss = 0.0
            running_corrects = 0

            net.eval()

            for val_inputs, val_labels in val.trainloader:

                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                # Generate adversarial training examples
                if do_advtrain_val:
                    attack = attacks.PGDAttack(net, ATK_EPS, ATK_ITERS, ATK_ALPHA, rand=True)
                    val_inputs = attack.perturb_l2_v2(val_inputs, labels)

                val_inputs = val_inputs.cuda()
                val_labels = val_labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(False):

                    # forward
                    # track history if only in train
                    outputs = net(val_inputs) 

                    _, preds = torch.max(outputs, 1)

                    loss = F.cross_entropy(outputs, labels)

                running_loss += loss.item() * val_inputs.size(0)
                running_corrects += torch.sum(preds == val_labels.data)

            epoch_loss = running_loss / len(val.trainset)
            epoch_acc = running_corrects.cpu().numpy() / len(val.trainset)

            logging.info('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict()) 

    torch.save(best_model_wts, model_dir)

        #logging.info()

    time_elapsed = time.time() - since
    logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
