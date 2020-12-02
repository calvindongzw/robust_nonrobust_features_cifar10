import torch as ch
from torchvision import transforms
import os
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvF
import pickle

from cifar_input import *
from train_util_l2 import *
import model

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = './data'

class TensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)
        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)

import pickle
with open("./data_r/Dr_norm.txt", "rb") as f:   
   data_raw = pickle.load(f)
# with open("./data_r/Dr_clip.txt", "rb") as f:   
#    data_raw = pickle.load(f)

train_data = torch.zeros((len(data_raw),3,32,32))
for i in range(len(data_raw)):
    train_data[i] = data_raw[i][0]

train_labels = torch.zeros((len(data_raw),1))
for i in range(len(data_raw)):
    train_labels[i] = data_raw[i][1]
    
train_data = torch.clamp(train_data, 0, 1)
    
train_transform = transforms.Compose([
						transforms.ToTensor()
						])
    
train_set = TensorDataset(train_data, train_labels.long().view(-1), transform=train_transform) 
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)

data_path = './data'

val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=128, shuffle=False, 
                                    num_workers=0, pad=2, image_size=32, flip_rate=0.5)

## Adversarial training

net = model.ResNetCIFAR(50).to(device)

## Checkpoint name for this model

model_checkpoint = "r_model_norm.pt"
#model_checkpoint = "r_model_clip.pt"

## Basic training params

num_epochs = 100
initial_lr = 0.1
momentum = 0.9
weight_decay = 5e-4
summary_steps = 10

optimizer = torch.optim.SGD(net.parameters(), initial_lr, momentum, weight_decay)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs*0.5), int(num_epochs*0.75)], gamma=0.1)

since = time.time()

best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

for i in range(num_epochs):

    net.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.long().to(device)

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

    scheduler.step()


    print("{} epoch train completed".format(i))
    #logging.info("====================================")

    if i % summary_steps == 0:

        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.cpu().numpy() / len(train_data)

        print('{} Train Loss: {:.4f} Trian Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

        running_loss = 0.0
        running_corrects = 0

        net.eval()

        for val_inputs, val_labels in val.trainloader:

            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            optimizer.zero_grad()

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

        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict()) 

torch.save(best_model_wts, model_checkpoint)

    #logging.info()

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))










