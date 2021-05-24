# -*- coding: utf-8 -*-
"""
Main script for divergence regulated encoder network (DREN)
demo.py
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os
import pdb

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim

## Local external libraries
from Utils.Network_functions import initialize_model, train_model, test_model
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Save_Results import save_results
from Demo_Parameters import Network_parameters
from Prepare_Data import Prepare_DataLoaders
from Texture_information import Class_names
from Utils.TSNE_Loss import TSNE_Loss
import pdb

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Name of dataset
Dataset = Network_parameters['Dataset']  # 'KTH_TIPS'
print('ytorch version ', torch.__version__)

# Model(s) to be used
model_name = Network_parameters['Model_names'][Dataset]  # 'resnet18'

# Number of classes in dataset
num_classes = Network_parameters['num_classes'][Dataset]

# Number of runs and/or splits for dataset
numRuns = Network_parameters['Splits'][Dataset]

# Number of bins and input convolution feature maps after channel-wise pooling
numBins = Network_parameters['numBins']
num_feature_maps = Network_parameters['out_channels'][model_name]

# Local area of feature map after histogram layer
feat_map_size = Network_parameters['feat_map_size']

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Location to store trained models
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, Network_parameters['folder'])

# Class names for embeddings
class_names = Class_names[Dataset]

print('Starting Experiments...')
for divergence_method in Network_parameters['divergence_method']:
    for current_dim in Network_parameters['embed_dim']:
        for alpha in Network_parameters['alpha']:
            for current_weight in Network_parameters['weights']:
                for split in range(0, numRuns):
                    print(current_dim)
                    # Keep track of the bins and widths as these values are updated each
                    # epoch
                    saved_bins = np.zeros((Network_parameters['num_epochs'] + 1,
                                           numBins * int(num_feature_maps / (feat_map_size * numBins))))
                    saved_widths = np.zeros((Network_parameters['num_epochs'] + 1,
                                             numBins * int(num_feature_maps / (feat_map_size * numBins))))

                    histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                                     Network_parameters['kernel_size'][model_name],
                                                     num_bins=numBins, stride=Network_parameters['stride'],
                                                     normalize_count=Network_parameters['normalize_count'],
                                                     normalize_bins=Network_parameters['normalize_bins'])

                    # Initialize the histogram model for this run
                    model_ft, input_size = initialize_model(model_name, num_classes,
                                                            Network_parameters['in_channels'][model_name],
                                                            num_feature_maps,
                                                            feature_extract=Network_parameters['feature_extraction'],
                                                            histogram=Network_parameters['histogram'],
                                                            histogram_layer=histogram_layer,
                                                            parallel=Network_parameters['parallel'],
                                                            use_pretrained=Network_parameters['use_pretrained'],
                                                            add_bn=Network_parameters['add_bn'],
                                                            scale=Network_parameters['scale'],
                                                            feat_map_size=feat_map_size,
                                                            embed_dim=current_dim)

                    # Send the model to GPU if available, use multiple if available
                    if torch.cuda.device_count() > 1:
                        print("Using", torch.cuda.device_count(), "GPUs!")
                        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                        model_ft = nn.DataParallel(model_ft)
                    model_ft = model_ft.to(device)

                    # Print number of trainable parameters
                    num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
                    print("Number of parameters: %d" % (num_params))
                    print("Initializing Datasets and Dataloaders...")

                    # Create training and validation dataloaders
                    dataloaders_dict = Prepare_DataLoaders(Network_parameters, split, input_size=input_size)

                    # Save the initial values for bins and widths of histogram layer
                    # Set optimizer for model
                    if (Network_parameters['histogram']):
                        reduced_dim = int((num_feature_maps / feat_map_size) / (numBins))
                        if (Network_parameters['in_channels'][model_name] == reduced_dim):
                            dim_reduced = False
                            saved_bins[0, :] = model_ft.features.histogram_layer[
                                -1].centers.detach().cpu().numpy()
                            saved_widths[0, :] = model_ft.features.histogram_layer[-1].widths.reshape(
                                -1).detach().cpu().numpy()
                        else:
                            dim_reduced = True
                            saved_bins[0, :] = model_ft.features.histogram_layer[
                                -1].centers.detach().cpu().numpy()
                            saved_widths[0, :] = model_ft.features.histogram_layer[-1].widths.reshape(
                                -1).detach().cpu().numpy()
                            # When running on hpg, model_ft.features...
                    else:
                        saved_bins = None
                        saved_widths = None
                        dim_reduced = None

                    # Use same learning rate for whole model
                    # optimizer_ft = optim.Adam(model_ft.parameters(), lr=Network_parameters['pt_lr'])

                    # Setup the loss fxn
                    class_criterion = nn.CrossEntropyLoss()
                 
                    embedding_criterion = TSNE_Loss(reduction='mean', device=device, dof=Network_parameters['dof'], alpha=alpha,
                                                    loss_metric=divergence_method)
                    criterion = {'class': class_criterion, 'embed': embedding_criterion}
                 
                    scheduler = None
                    
                
                    params = list(model_ft.parameters()) + list(embedding_criterion.parameters())
                    optimizer_ft = optim.Adam(params, lr=Network_parameters['pt_lr'])

                    # Train and evaluate
                    train_dict = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device,
                                             saved_bins=saved_bins, saved_widths=saved_widths,
                                             histogram=Network_parameters['histogram'],
                                             num_epochs=Network_parameters['num_epochs'],
                                             scheduler=scheduler,
                                             dim_reduced=dim_reduced,
                                             weight=current_weight, class_names=class_names)
                    test_dict = test_model(dataloaders_dict['test'], model_ft, criterion,
                                           device, current_weight)

                    # Save results
                    if (Network_parameters['save_results']):
                        save_results(train_dict, test_dict, split, Network_parameters,
                                     num_params, current_weight, current_dim, divergence_method, alpha)
                        del train_dict, test_dict
                        torch.cuda.empty_cache()

                    if (Network_parameters['histogram']):
                        print('**********Run ' + str(split + 1) + ' For '
                              + Network_parameters['hist_model'] + ' Weight = ' +
                              str(current_weight) + ' Finished**********')
                    else:
                        print('**********Run ' + str(split + 1) + ' For GAP_' +
                              model_name + ' Weight = ' + str(current_weight)
                              + ' Finished**********')