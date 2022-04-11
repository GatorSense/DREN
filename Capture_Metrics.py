# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:08:12 2021
Compute validation/test metrics for each model in spreadsheet
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

def load_metrics(sub_dir, metrics, phase = 'val'):
    
    #Load metrics
    temp_file = open(sub_dir + '{}_metrics.pkl'.format(phase), 'rb')
    temp_metrics = pickle.load(temp_file)
    temp_file.close()
    
    #Return max value for each metric (unless loss or inference time)
    metric_vals = np.zeros(len(metrics))
    
    count = 0
    for metric in metrics.keys():
        metric_vals[count] = temp_metrics[metric]
        count += 1
    
    return metric_vals
 
def add_to_excel(table,writer,weights,dim_names,model_name):
    
    DF = pd.DataFrame(table,index=weights,columns=dim_names)
    DF.colummns = dim_names
    DF.index = weights
    DF.to_excel(writer,sheet_name='{}'.format(model_name))
           
#Compute desired metrics and save to excel spreadsheet
def Get_Metrics(metrics,model_names,args):
    
    #Set paramters for experiments
    Params = Parameters(args)    

    weights = Params['weights']
    dim_names = Params['embed_dim']
    
    #Generate dim names with std
    new_dim_names = []
    
    for dim in dim_names:
        new_dim_names.append('{}D'.format(dim))
        new_dim_names.append('STD')

    fig_dir = Generate_Dir_Name(Params)
    test_writer = pd.ExcelWriter(fig_dir+'Overall_Accuracy.xlsx', engine='xlsxwriter')
    
    # Initialize the histogram model for this run
    model_count = 0
    for model in model_names:
        # setattr(args, 'model', model)
        # temp_params = Parameters(args)
        
        #Get metrics for validation and test
        test_table = metrics[:,:,model_count]
        
        print('Finished Model {}'.format(model))
            
        #Add metrics to spreadsheet
        add_to_excel(test_table,test_writer,weights,new_dim_names,model)
        model_count += 1
    
    #Compute average and std across folds
    #Save spreadsheets
    test_writer.save()
    

def main(Params):
  
    metrics = np.zeros((len(Params['weights']),len(Params['embed_dim']*2)))
    count = 0
    for divergence_method in Params['divergence_method']:
        for alpha in Params['alpha']:
            dim_count = 0
            for dimension in Params['embed_dim']:
                weight_count = 0
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
                            
                        #Read metrics
                        try:
                            accuracy = np.loadtxt('{}List_Accuracy.txt'.format(weight_dir))
                        except: #Results not completed
                            accuracy = [0,0]
                        
                        #Get mean and std
                        mean_acc = np.mean(accuracy)
                        std_acc = np.std(accuracy)
                    
                        metrics[weight_count,2*dim_count:2*dim_count+2] = mean_acc, std_acc
        
                    weight_count += 1
                dim_count += 1
                        # if dimension > 17:
                        #     break

    return metrics

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parse_args():
    parser = argparse.ArgumentParser(description='Run DREN experiments for dataset')
    parser.add_argument('--save_results', type=int, default=True,
                        help='Save results of experiments(default: True')
    parser.add_argument('--folder', type=str, default='Saved_Models/',
                        help='Location to save models')
    parser.add_argument('--model', type=str, default='densenet121',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=True, type=boolean_string,
                        help='Flag to use histogram model or baseline global average pooling (GAP)')
    parser.add_argument('--data_selection', type=int, default=4,
                        help='Dataset selection:  1: DTD, 2: GTOS-mobile, 3: MINC_2500, 4: KTH_TIPS')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--feature_extraction', type=bool, default=True,
                        help='Flag for feature extraction. False, train whole model. True, only update fully connected and histogram layers parameters (default: True)')
    parser.add_argument('--use_pretrained', type=bool, default=True,
                        help='Flag to use pretrained model from ImageNet or train from scratch (default: True)')
    parser.add_argument('-w','--weights', nargs='*', help='<Required> Set flag', required=False,
                        default=[0,.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], type=float)
    # parser.add_argument('--weights', type=list, default=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
    #                     help=' Set weights for objective term: value(s) should be between 0 and 1. (default: [.25, .5, .75, 1]')
    # parser.add_argument('-embed_dim', '--list', help='Embedding dimension', 
    # type=lambda s: [int(item) for item in s.split(',')], default=[2,3,4,8,16])
    parser.add_argument('-l','--embed_dim', nargs='*', help='<Required> Set flag', required=False,
                        default=[2, 3, 4, 8, 16], type=int)
    # parser.add_argument('-l','--embed_dim', nargs='*', help='<Required> Set flag', required=False,
    #                     default=[None], type=int)
    # parser.add_argument('--embed_dim', type=list, default=[2,3,4,8,16],
    #                     help=' Embedding dimension of encoder. (default: [2,3,4,8,16]')
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
    # params = Parameters(args)
    # main(params)
    
    #Embedding dims (no encoder, no histogram)
    # [1280, 2048, 2048, 1024, 400]
    
    #Embedding dims (no encoder, with histogram)
    # [2560, 4096, 4096, 2048, 784]
    Params = Parameters(args)
    models = ['efficientnet', 'resnet50_wide', 'resnet50_next','densenet121', 'regnet']
    model_metrics = np.zeros((len(Params['weights']),len(Params['embed_dim']*2),
                              len(models)))
    
    model_count = 0
    for model in models:
        setattr(args, 'model', model) 
        params = Parameters(args)
        model_metrics[:,:,model_count] = main(params)
        model_count += 1
        print('Finished Model {} of {}'.format(model_count,len(models)))
        
    #Generate spreadsheet
    Get_Metrics(model_metrics,models,args)










    
    