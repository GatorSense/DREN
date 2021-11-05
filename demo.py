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
import argparse

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim

## Local external libraries
from Utils.Network_functions import initialize_model, train_model, test_model
from Utils.RBFHistogramPooling import HistogramLayer
from Utils.Save_Results import save_results
from Demo_Parameters import Parameters
from Prepare_Data import Prepare_DataLoaders
from Texture_information import Class_names
from Utils.TSNE_Loss import TSNE_Loss
import pdb


def main(Params):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Name of dataset
    Dataset = Params['Dataset']  # 'KTH_TIPS'
    # print('ytorch version ', torch.__version__)
    
    # Model(s) to be used
    model_name = Params['Model_name']
    
    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]
    
    # Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]
    
    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']
    num_feature_maps = Params['out_channels'][model_name]
    
    # Local area of feature map after histogram layer
    feat_map_size = Params['feat_map_size']
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Location to store trained models
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, Params['folder'])
    
    # Class names for embeddings
    class_names = Class_names[Dataset]
    
    print('Starting Experiments...')
    for divergence_method in Params['divergence_method']:
        for current_dim in Params['embed_dim']:
            for alpha in Params['alpha']:
                for current_weight in Params['weights']:
                    for split in range(0, numRuns):
                        print(current_dim)
                        # Keep track of the bins and widths as these values are updated each
                        # epoch
                        saved_bins = np.zeros((Params['num_epochs'] + 1,
                                               numBins * int(num_feature_maps / (feat_map_size * numBins))))
                        saved_widths = np.zeros((Params['num_epochs'] + 1,
                                                 numBins * int(num_feature_maps / (feat_map_size * numBins))))
    
                        histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                                         Params['kernel_size'][model_name],
                                                         num_bins=numBins, stride=Params['stride'],
                                                         normalize_count=Params['normalize_count'],
                                                         normalize_bins=Params['normalize_bins'])
    
                        # Initialize the histogram model for this run
                        model_ft, input_size = initialize_model(model_name, num_classes,
                                                                Params['in_channels'][model_name],
                                                                num_feature_maps,
                                                                feature_extract=Params['feature_extraction'],
                                                                histogram=Params['histogram'],
                                                                histogram_layer=histogram_layer,
                                                                parallel=Params['parallel'],
                                                                use_pretrained=Params['use_pretrained'],
                                                                add_bn=Params['add_bn'],
                                                                scale=Params['scale'],
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
                        dataloaders_dict = Prepare_DataLoaders(Params, split, input_size=input_size)
    
                        # Save the initial values for bins and widths of histogram layer
                        # Set optimizer for model
                        if (Params['histogram']):
                            reduced_dim = int((num_feature_maps / feat_map_size) / (numBins))
                            if (Params['in_channels'][model_name] == reduced_dim):
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
    
                        # Setup the loss fxn
                        class_criterion = nn.CrossEntropyLoss()
                     
                        embedding_criterion = TSNE_Loss(reduction='mean', device=device, dof=Params['dof'], alpha=alpha,
                                                        loss_metric=divergence_method)
                        criterion = {'class': class_criterion, 'embed': embedding_criterion}
                     
                        scheduler = None
                        
                    
                        params = list(model_ft.parameters()) + list(embedding_criterion.parameters())
                        optimizer_ft = optim.Adam(params, lr=Params['lr'])
    
                        # Train and evaluate
                        train_dict = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device,
                                                 saved_bins=saved_bins, saved_widths=saved_widths,
                                                 histogram=Params['histogram'],
                                                 num_epochs=Params['num_epochs'],
                                                 scheduler=scheduler,
                                                 dim_reduced=dim_reduced,
                                                 weight=current_weight, class_names=class_names)
                        test_dict = test_model(dataloaders_dict['test'], model_ft, criterion,
                                               device, current_weight)
    
                        # Save results
                        if (Params['save_results']):
                            save_results(train_dict, test_dict, split, Params,
                                         num_params, current_weight, current_dim, divergence_method, alpha)
                            del train_dict, test_dict
                            torch.cuda.empty_cache()
    
                        if (Params['histogram']):
                            print('**********Run ' + str(split + 1) + ' For '
                                  + Params['hist_model'] + ' Weight = ' +
                                  str(current_weight) + ' Finished**********')
                        else:
                            print('**********Run ' + str(split + 1) + ' For GAP_' +
                                  model_name + ' Weight = ' + str(current_weight)
                                  + ' Finished**********')

def parse_args():
    parser = argparse.ArgumentParser(description='Run DREN experiments for dataset')
    parser.add_argument('--save_results', type=int, default=True,
                        help='Save results of experiments(default: True')
    parser.add_argument('--folder', type=str, default='Saved_Models/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='efficientnet',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', type=bool, default=False,
                        help='Flag to use histogram model or baseline global average pooling (GAP)')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection:  1: DTD, 2: GTOS-mobile, 3: MINC_2500, 4: KTH_TIPS')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', type=bool, default=True,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', type=bool, default=True,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('--weights', type=list, default=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1],
                        help=' Set weights for objective term: value(s) should be between 0 and 1. (default: [.25, .5, .75, 1]')
    parser.add_argument('--embed_dim', type=list, default=[2,3,4,8,16],
                        help=' Embedding dimension of encoder. (default: [2,3,4,8,16]')
    parser.add_argument('--divergence_method', type=list, default=['Renyi', 'EMD'], 
                        help='Divergence approach to use (default: [Renyi, EMD])')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--dof', type=int, default=1,
                        help='Degrees of freedome for t-distribution (default: 1)')
    parser.add_argument('--alpha', type=list, default=[.5,1],
                        help='Alpha for Renyi divergence (default: [.5,1])')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='Resize the image before center crop. (default: 256)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    params = Parameters(args)
    main(params)