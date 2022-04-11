# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:08:12 2021
Compute validation/test metrics for each model (supplemental files)
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
import pdb
import argparse


## PyTorch dependencies
import torch

## Local external libraries
from Demo_Parameters import Parameters
from Utils.Plot_Accuracy import plot_metrics_multiple_runs



def Generate_Dir_Name(Params):
    
    if (Params['histogram']):
        if (Params['parallel']):
            dir_name = (Params['folder'] + '/' + Params['mode']
                        + '/' + Params['Dataset'] + '/Hist_Models/')
        else:
            dir_name = (Params['folder'] + '/' + Params['mode']
                        + '/' + Params['Dataset'] + '/')
    # Baseline model
    else:
        dir_name = (Params['folder'] + '/' + Params['mode']
                    + '/' + Params['Dataset'] + '/Base_Models/')
        
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        
    return dir_name


          
def main(Params):
  
    count = 0
    NumRuns = Params['Splits'][Params['Dataset']]
    for divergence_method in Params['divergence_method']:
        for alpha in Params['alpha']:
            dim_count = 0
            # fig, ax = plt.subplots(len(Params['embed_dim']))
            for dimension in Params['embed_dim']:
                # fig, ax = plt.subplots(nrows=1)
                weight_count = 0
                dim_dict = {}
                for weight in Params['weights']:
    
                    # If no encoder and weight is non zero, skip generating results
                    if (dimension is None) and not(weight==0):
                        pass
                    else:
                        count += 1
                        
                        if (Params['histogram']):
                            if (Params['parallel']):
                                weight_dir = (Params['folder'] + '/' + Params['mode']
                                            + '/' + Params['Dataset'] + '/{}_'.format(Params['Model_name'])
                                            + Params['hist_model'] + '/' + divergence_method + '/' + 'alpha_' + str(
                                            alpha) + '/Weight_' + str(weight)
                                            + '/Embed_' + str(dimension) + 'D/Parallel/')
                            else:
                                weight_dir = (Params['folder'] + '/' + Params['mode']
                                            + '/' + Params['Dataset'] + '/{}'.format(Params['Model_name']) +
                                            + Params['hist_model'] + '/' + divergence_method + '/' + 'alpha_' + str(
                                            alpha) + '/Weight_' +
                                            str(weight) + '/Embed_' + str(dimension)
                                            + 'D' + '/Inline/')
                        # Baseline model
                        else:
                            weight_dir = (Params['folder'] + '/' + Params['mode']
                                        + '/' + Params['Dataset'] + '/GAP_' +
                                        Params['Model_name']
                                        + '/' + divergence_method + '/' + 'alpha_' + str(alpha) + '/Weight_' + str(weight) + '/Embed_' +
                                        str(dimension) + 'D/')
                            
                            
                        train_embed_loss = []
                        train_class_loss = []
                        train_total_loss = []
                        train_acc = []
                        val_embed_loss = []
                        val_class_loss = []
                        val_total_loss = []
                        val_acc = []
                        for split in range(0, NumRuns):
        
                            sub_dir = weight_dir + 'Run_' + str(split + 1) + '/'
                            # Load training and testing files (Python)
                            train_pkl_file = open(sub_dir + 'train_dict.pkl', 'rb')
                            train_dict = pickle.load(train_pkl_file)
                            train_pkl_file.close()
        
                            # Remove pickle files
                            del train_pkl_file
                            
                            #Get training and validation errors/accuracy
                            temp_train_error = train_dict['train_error_track']
                            temp_train_acc  = train_dict['train_acc_track']
                            temp_val_error = train_dict['val_error_track']
                            temp_val_acc = train_dict['val_acc_track']
                            
                            #Loss: 'embed_loss', 'class_loss', and 'total'
                            #Append results 
                            train_embed_loss.append(np.array(temp_train_error['embed_loss']))
                            train_class_loss.append(np.array(temp_train_error['class_loss']))
                            train_total_loss.append(np.array(temp_train_error['total']))
                            train_acc.append(np.array(temp_train_acc))
                            val_embed_loss.append(np.array(temp_val_error['embed_loss']))
                            val_class_loss.append(np.array(temp_val_error['class_loss']))
                            val_total_loss.append(np.array(temp_val_error['total']))
                            val_acc.append(np.array(temp_val_acc))
                            
                    #Put results into dictionary  
                    train_info = {'class_loss': np.array(train_class_loss),
                                  'embed_loss': np.array(train_embed_loss),
                                  'total_loss': np.array(train_total_loss),
                                  'accuracy': np.array(train_acc)}
                    
                    val_info = {'class_loss': np.array(val_class_loss),
                                  'embed_loss': np.array(val_embed_loss),
                                  'total_loss': np.array(val_total_loss),
                                  'accuracy': np.array(val_acc)}
        
                    dim_dict[weight] = {'train': train_info, 'val': val_info}
        
                    weight_count += 1
                
                #Plot learning curves (mean and std) for each dimensionality
                fig_dir = Generate_Dir_Name(Params)
                plot_metrics_multiple_runs(dim_dict,'D = {}'.format(dimension),
                                           dimension,fig_dir=fig_dir)
                
                #Increment dimension
                dim_count += 1

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    parser = argparse.ArgumentParser(description='Run DREN experiments for dataset')
    parser.add_argument('--save_results', type=int, default=True,
                        help='Save results of experiments(default: True')
    parser.add_argument('--folder', type=str, default='kth_augmented/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='densenet121',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=True, type=boolean_string,
                        help='Flag to use histogram model or baseline global average pooling (GAP)')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection:  1: DTD, 2: GTOS-mobile, 3: MINC_2500, 4: KTH_TIPS')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', type=bool, default=True,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', type=bool, default=True,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('-w','--weights', nargs='*', help='<Required> Set flag', required=False,
                        default=[0,.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], type=float)
    parser.add_argument('-l','--embed_dim', nargs='*', help='<Required> Set flag', required=False,
                        default=[2, 3, 4, 8, 16], type=int)
    parser.add_argument('--divergence_method', type=list, default=['Renyi'], 
                        help='Divergence approach to use (default: [Renyi, EMD])')
    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=256,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--dof', type=int, default=1,
                        help='Degrees of freedome for t-distribution (default: 1)')
    parser.add_argument('--alpha', type=list, default=[1],
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
   
    Params = Parameters(args)
    models = ['resnet50']

    model_count = 0
    for model in models:
        setattr(args, 'model', model) 
        params = Parameters(args)
        main(params)
        model_count += 1
        print('Finished Model {} of {}'.format(model_count,len(models)))
        









    
    