import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import pandas
from scipy.io import loadmat 
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time 
import math
import random
from scipy.fftpack import dct, idct # import for discrete cosine transform
from torchsummary import summary 

from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
matPath = "../data/DatasColor_29.mat"
data = scipy.io.loadmat(matPath) 

# def showMatFile(matPath):
    # Load the mat file
    # data = scipy.io.loadmat(matPath)    
    # Show images
    # for i in range(0, 10):
        # img = data['DATA'][0][0][0][i]
        # plt.imshow(img)
        # plt.show()
#showMatFile(matPath)
 
class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, mat_path, transform=None, train=True, fold=1):
        self.mat_data = scipy.io.loadmat(mat_path)
        self.fold = fold
        self.train = train
        self.transform = transform
        self.train_indices = self.mat_data['DATA'][0][2][fold-1, :299] - 1
        self.test_indices = self.mat_data['DATA'][0][2][fold-1, 299:374] - 1
        self.y_train = self.mat_data['DATA'][0][1][0, self.train_indices]
        self.y_test = self.mat_data['DATA'][0][1][0, self.test_indices]
        self.num_classes = len(np.unique(self.y_train))
        self.images = self.mat_data["DATA"][0][0][0]   # contains 374 images with images being of size (312,417,3)
        
    def __len__(self):
        if self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)
        
    def __getitem__(self, idx):
        if self.train:
            img = Image.fromarray(self.images[self.train_indices[idx]])
            label = self.y_train[idx] - 1  # shift the labels to start from 0
        else:
            img = Image.fromarray(self.images[self.test_indices[idx]])
            label = self.y_test[idx] - 1   # shift the labels to start from 0
            
        if self.transform:
            img = self.transform(img)
        
        return img, label



# load dataset
data = scipy.io.loadmat(matPath)
DIV = data['DATA'][0][2]   # Division between training and test set
DIM1 = 299  # Number of training patterns
DIM2 = 374 # Number of patterns
NF = 5 
yE = data['DATA'][0][1]  # Labels of the patterns 
Images = data["DATA"][0][0][0]   # Images

# # Neural network parameters
miniBatchSize = 30 
num_classes = 3


train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=227, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=227),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MyDataset(mat_path=matPath, transform=train_transforms, train=True, fold=1)
valid_dataset = MyDataset(mat_path=matPath, transform=val_transforms, train=False, fold=1)

train_data_loader = DataLoader(train_dataset, batch_size=miniBatchSize, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=miniBatchSize, shuffle=False)

train_data_size = DIV[0, :DIM1].shape[0]
valid_data_size = DIV[0, DIM1:DIM2].shape[0]

# Transer learning model
alexnet = models.alexnet(pretrained=True)
alexnet

# Freeze model parameters : Only use last layer and not update whole params (faster)
for param in alexnet.parameters():
    param.requires_grad = False
# Change the final layer of AlexNet Model for Transfer Learning
alexnet.classifier[6] = nn.Linear(4096, num_classes)
alexnet.classifier.add_module("7", nn.LogSoftmax(dim = 1))
alexnet
summary(alexnet, (3, 224, 224))

# Define Optimizer and Loss Function
loss_func = nn.NLLLoss()
optimizer = optim.Adam(alexnet.parameters())
optimizer

def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_acc = 0.0

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs) 
               
            # print(labels.long())
            # print(outputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels.long())
            
            # Backpropagate the gradients
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels.long())

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)

                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
                
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
            
    return model, history


num_epochs = 5
trained_model, history = train_and_validate(alexnet, loss_func, optimizer, num_epochs)

torch.save(history, 'datasColor_29_training_history.pt') 
