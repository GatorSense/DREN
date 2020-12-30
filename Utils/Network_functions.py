# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:26:08 2019

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import matplotlib.animation as animation
from math import floor, log
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models

## Local external libraries
from Utils.Histogram_Model import HistRes
# from barbar import Bar
from Utils.Embedding_Model import Embedding_model

import pdb


# Function used to scale KL loss to same as classification
def floor_power_of_10(n):
    try:
        exp = log(n, 10)
        exp = floor(exp)
    except:
        exp = 0
    return 10 ** exp


def train_model(model, dataloaders, criterion, optimizer, device,
                saved_bins=None, saved_widths=None, histogram=True,
                num_epochs=25, scheduler=None, dim_reduced=True,
                weight=.5, class_names=None):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    train_error_class_history = []
    train_error_embed_history = []
    val_error_history = []
    val_error_class_history = []
    val_error_embed_history = []
    # frames = [] # for storing the generated images
    dict_embeddings = []
    dict_labels = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode 
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            running_loss_class = 0.0
            running_loss_embed = 0.0

            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                index = index.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs, embedding, model_feats = model(inputs)
                    loss_class = criterion['class'](outputs, labels)
                    loss_embedding = criterion['embed'](model_feats,embedding)
                    
                    #Check scale on features, KL is usually smaller
                    #If just embedding loss, don't need scale. 
                    # if not(weight==1):
                    if(loss_embedding.item() != 0):
                        scale = loss_class.item() // loss_embedding.item()
                        scale = floor_power_of_10(scale)
                    else:
                         scale = 0
                    loss = (1-weight)*loss_class  + (weight)*loss_embedding*scale
    
                    _, preds = torch.max(outputs, 1)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_class += loss_class.item() * inputs.size(0) 
                running_loss_embed += loss_embedding.item() * inputs.size(0) * scale
                running_corrects += torch.sum(preds.data == labels.data)
        
            epoch_loss = running_loss / (len(dataloaders[phase].sampler))
            epoch_loss_class = running_loss_class / (len(dataloaders[phase].sampler))
            epoch_loss_embed = running_loss_embed / (len(dataloaders[phase].sampler))
            epoch_acc = running_corrects.double().cpu().numpy() / (len(dataloaders[phase].sampler))
            
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                train_error_history.append(epoch_loss)
                train_error_class_history.append(epoch_loss_class)
                train_error_embed_history.append(epoch_loss_embed)
                train_acc_history.append(epoch_acc)
                if(histogram):
                    if dim_reduced:
                        #save bins and widths
                        saved_bins[epoch+1,:] = model.features.histogram_layer[-1].centers.detach().cpu().numpy()
                        saved_widths[epoch+1,:] = model.features.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
                    else:
                        # save bins and widths
                        saved_bins[epoch + 1, :] = model.features.histogram_layer.centers.detach().cpu().numpy()
                        saved_widths[epoch + 1, :] = model.features.histogram_layer.widths.reshape(
                            -1).detach().cpu().numpy()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                #          if phase == 'val' and epoch_loss < best_loss: #Save model that achieves lowest loss
                best_epoch = epoch
                best_acc = epoch_acc
                # best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_error_history.append(epoch_loss)
                val_error_class_history.append(epoch_loss_class)
                val_error_embed_history.append(epoch_loss_embed)
                val_acc_history.append(epoch_acc)

                # Get embedding for training, validation, and test
                if model.embed_dim == 2 or model.embed_dim == 3:
                    temp_embeddings, temp_labels = Get_Embeddings(dataloaders,
                                                                  model, device, class_names,
                                                                  model.embed_dim)
                    # ani = animation.FuncAnimation(vid_fig,Get_Embeddings(dataloaders,
                    #                               model,device,class_names,
                    #                               model.embed_dim))
                    dict_embeddings.append(temp_embeddings)
                    dict_labels.append(temp_labels)


            print()
            print('{} Total Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
            print()
            print('{} Classification Loss: {:.4f}'.format(phase, epoch_loss_class))
            print()
            print('{} Embedding Loss: {:.4f}'.format(phase, epoch_loss_embed))
            print()
    
    # if frames: #If not empty
    #     vid_fig = plt.figure(figsize=(14,6))
    #     ani = animation.ArtistAnimation(vid_fig, frames, interval=50, blit=True,
    #                                 repeat_delay=1000) 
    #     plt.show()       

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return losses as dictionary
    train_loss = {'total': train_error_history, 'class_loss': train_error_class_history,
                 'embed_loss': train_error_embed_history}
    
    val_loss = {'total': val_error_history, 'class_loss': val_error_class_history,
                 'embed_loss': val_error_embed_history}
 
    #Return training and validation information
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_loss,'train_acc_track': train_acc_history, 
                  'train_error_track': train_loss,'best_epoch': best_epoch, 
                  'saved_bins': saved_bins, 'saved_widths': saved_widths,
                  'embedding_epochs': dict_embeddings,
                  'embedding_labels': dict_labels}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def Get_Embeddings(dataloaders,model,device,class_names,embed_dim):
   
    print('Generate embedding visual...')

    model.eval() #Figure out how to incoporate this in training loop
  
    # fig =  plt.figure(figsize=(14,6))
    # count = 0
    dict_embeddings = {'train': None, 'val': None, 'test': None}
    dict_labels = {'train': None, 'val': None, 'test': None}
    # Each epoch has a training and validation phase
    for phase in ['train', 'val', 'test']:
        GT_vals = np.array(0)
        embedding = []
        with torch.no_grad():
            for idx, (inputs, labels, index) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device).cpu().numpy()
        
                GT_vals = np.concatenate((GT_vals, labels),axis = None)
                
                # Get embedding for each batch
                _, temp_embedding, _ = model(inputs)
                embedding.append(temp_embedding.detach().cpu().numpy())
        
        #Convert embedding list to numpy array
        embedding = np.concatenate(embedding,axis=0) 
        GT_vals = GT_vals[1:]
        
        dict_embeddings[phase] = embedding
        dict_labels[phase] = GT_vals
   
    return dict_embeddings,dict_labels
     
def test_model(dataloader,model,criterion,device,weight):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    running_loss = 0.0
    running_loss_class = 0.0
    running_loss_embed = 0.0
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels, index) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            # forward
            outputs, embedding, model_feats = model(inputs)
            loss_class = criterion['class'](outputs, labels)
            loss_embedding = criterion['embed'](model_feats,embedding)
            #Check scale on features, KL is usually smaller
            if (loss_embedding.item() != 0):
                scale = loss_class.item() // loss_embedding.item()
                scale = floor_power_of_10(scale)
            else:
                scale = 0
            loss = (1-weight)*loss_class + (weight)*loss_embedding*scale
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
        
            running_corrects += torch.sum(preds == labels.data)
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_class += loss_class.item() * inputs.size(0)
            running_loss_embed += loss_embedding.item() * inputs.size(0)
            
    epoch_loss = running_loss / (len(dataloader.sampler))
    epoch_loss_class = running_loss_class / (len(dataloader.sampler))
    epoch_loss_embed = running_loss_embed / (len(dataloader.sampler))
    test_acc = running_corrects.double() / (len(dataloader.sampler))
    
    test_loss = {'total': epoch_loss, 'class_loss': epoch_loss_class,
                 'embed_loss': epoch_loss_embed}
    
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index[1:],
                 'test_acc': np.round(test_acc.cpu().numpy()*100,2),
                 'test_loss': test_loss}
    
    return test_dict


def initialize_model(model_name, num_classes, in_channels, out_channels,
                     feature_extract=False, histogram=True, histogram_layer=None,
                     parallel=True, use_pretrained=True, add_bn=True, scale=5,
                     feat_map_size=4, embed_dim=2):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if(histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific.
        model_ft = HistRes(histogram_layer,parallel=parallel,
                           model_name=model_name,add_bn=add_bn,scale=scale,pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft.backbone, feature_extract)
        
        #Reduce number of conv channels from input channels to input channels/number of bins*feat_map size (2x2)
        reduced_dim = int((out_channels/feat_map_size)/(histogram_layer.numBins))
        if (in_channels==reduced_dim): #If input channels equals reduced/increase, don't apply 1x1 convolution
            model_ft.histogram_layer = histogram_layer
        else:
            conv_reduce = nn.Conv2d(in_channels,reduced_dim,(1,1))
            model_ft.histogram_layer = nn.Sequential(conv_reduce,histogram_layer)
        if(parallel):
            num_ftrs = model_ft.fc.in_features*2
        else:
            num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    # Baseline model
    else:
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        
    #Take model and return embedding model
    model_ft = Embedding_model(model_ft,input_feat_size=num_ftrs,
                               embed_dim=embed_dim)
    
    return model_ft, input_size


