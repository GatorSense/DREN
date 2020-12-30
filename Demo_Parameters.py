# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
import os
import sys

######## ONLY CHANGE PARAMETERS BELOW ########
#Flag for if results are to be saved out
#Set to True to save results out and False to not save results
save_results = True

#Location to store trained models
#Always add slash (/) after folder name
folder = 'Saved_Models/'

#Flag to use histogram model or baseline global average pooling (GAP)
# Set to True to use histogram layer and False to use GAP model
histogram = True

#Select dataset. Set to number of desired texture dataset
# For KTH, currently training on 2 samples, validating on 1 sample, and testing
# on 1 sample
data_selection = 4
Dataset_names = { 1: 'DTD', 2: 'GTOS-mobile', 3: 'MINC_2500', 4: 'KTH_TIPS'}

#Number of bins for histogram layer. Recommended values are 4, 8 and 16.
#Set number of bins to powers of 2 (e.g., 2, 4, 8, etc.)
#For HistRes_B models using ResNet18 and ResNet50, do not set number of bins
#higher than 128 and 512 respectively. Note: a 1x1xK convolution is used to
#downsample feature maps before binning process. If the bin values are set
#higher than 128 or 512 for HistRes_B models using ResNet18 or ResNet50
#respectively, than an error will occur due to attempting to reduce the number of
#features maps to values less than one
numBins = 16


#Flag for feature extraction. False, train whole model. True, only update
#fully connected and histogram layers parameters (default: False)
#Flag to use pretrained model from ImageNet or train from scratch (default: True)
#Flag to add BN to convolutional features (default:True)
#Location/Scale at which to apply histogram layer (default: 5 (at the end))
feature_extraction = True
use_pretrained = True
add_bn = True
scale = 5

#Set learning rate for new and pretrained (pt) layers
#Recommended values are to have the new layers at
#a learning rate 10 times larger than the pt learning rate.
#e.g., new_lr = .001 and pt_lr = .01
pt_lr = .01
new_lr = .01

#Set weights for objective term: value(s) should be between 0 and 1.
#Loss = (1-weight)*classification loss + (weight)*embedding loss
weights = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]


#Embedding dimension of encoder
embed_dim = [2,3]

#Type of divergence measure to use when calculating TSNE loss
#This will be the method to determine the embedding loss between the low
#and high dimensional features. Available options are Renyi's, Mutual Info,
# and Wasserstein (Earth Movers Distance)
divergence_method = ['Renyi']

#Degrees of freedom for T-distribution for TSNE
#If embed_dim is 2 or 3, recommended value is 1
# If embed_dim is greater than 3, will need more degrees of freedom
dof = 1

#alpha for Renyi's divergence
alpha = [0,.5,1.5,2]

#Parameters of Histogram Layer
#For no padding, set 0. If padding is desired,
#enter amount of zero padding to add to each side of image
#(did not use padding in paper, recommended value is 0 for padding)
padding = 0

#Apply rotation to test set (did not use in paper)
#Set rotation to True to add rotation, False if no rotation (used in paper)
#Recommend values are between 0 and 25 degrees
#Can use to test robustness of model to rotation transformation
rotation = False
degrees = 25

#Reduce dimensionality based on number of output feature maps from GAP layer
#Used to compute number of features from histogram layer
out_channels = {"resnet50": 2048, "resnet18": 512}

#Set whether to have the histogram layer inline or parallel (default: parallel)
#Set whether to use sum (unnormalized count) or average pooling (normalized count)
# (default: average pooling)
#Set whether to enforce sum to one constraint across bins (default: True)
parallel = True
normalize_count = True
normalize_bins = True

#Set step_size and decay rate for scheduler
#In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
step_size = 10
gamma = .1

#Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
#training batch size is recommended to be 64. If using at least two GPUs,
#the recommended training batch size is 128 (as done in paper)
#May need to reduce batch size if CUDA out of memory issue occurs
batch_size = {'train': 128, 'val': 128, 'test': 128}
num_epochs = 30

#Resize the image before center crop. Recommended values for resize is 256 (used in paper), 384,
#and 512 (from http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf)
#Center crop size is recommended to be 256.
resize_size = 256
center_size = 256

#Pin memory for dataloader (set to True for experiments)
pin_memory = True

#Set number of workers, i.e., how many subprocesses to use for data loading.
#Usually set to 0 or 1. Can set to more if multiple machines are used.
#Number of workers for experiments for two GPUs was three
num_workers = 0
#cpus = os.getenv('SLURM_CPUS_PER_TASK')
#num_workers = int(cpus) - 1
#Output feature map size after histogram layer
feat_map_size = 4

#Set filter size and stride based on scale
# Current values will produce 2x2 local feature maps
if scale == 1:
    stride = [32, 32]
    in_channels = {"resnet50": 64, "resnet18": 64}
    kernel_size = {"resnet50": [64,64],  "resnet18": [64,64]}
elif scale == 2:
    stride = [16, 16]
    in_channels = {"resnet50": 256, "resnet18": 64}
    kernel_size = {"resnet50": [32,32],  "resnet18": [32,32]}
elif scale == 3:
    stride = [8, 8]
    in_channels = {"resnet50": 512, "resnet18": 128}
    kernel_size = {"resnet50": [16,16],  "resnet18": [16,16]}
elif scale == 4:
    stride = [4, 4]
    in_channels = {"resnet50": 1024, "resnet18": 256}
    kernel_size = {"resnet50": [8,8],  "resnet18": [8,8]}
else:
    stride = [2, 2]
    in_channels = {"resnet50": 2048, "resnet18": 512}
    kernel_size = {"resnet50": [4,4],  "resnet18": [4,4]}

#Visualization of results parameters
#Visualization parameters for figures
fig_size = 12
font_size = 16

#Flag for TSNE visuals, set to True to create TSNE visual of features
#Set to false to not generate TSNE visuals
#Separate TSNE will visualize histogram and GAP features separately
#If set to True, TSNE of histogram and GAP features will be created
#Number of images to view for TSNE (defaults to all training imgs unless
#value is less than total training images).
TSNE_visual = False
Separate_TSNE = False
Num_TSNE_images = 10000

#Set to True if more than one GPU was used
Parallelize_model = True

######## ONLY CHANGE PARAMETERS ABOVE ########
if feature_extraction:
    mode = 'Feature_Extraction'
else:
    mode = 'Fine_Tuning'

#Location of texture datasets
Data_dirs = {'DTD': './Datasets/DTD/',
             'MINC_2500': './Datasets/minc-2500/',
             'GTOS-mobile': './Datasets/gtos-mobile',
             'KTH_TIPS': './Datasets/KTH-TIPS2-b'}

#ResNet models to use for each dataset
Model_names = {'DTD': 'resnet50',
               'MINC_2500': 'resnet50',
               'GTOS-mobile': 'resnet18',
               'KTH_TIPS': 'resnet18'}

#Number of classes in each dataset
num_classes = {'DTD': 47,
               'MINC_2500': 23,
               'GTOS-mobile': 31,
               'KTH_TIPS': 11}

#Number of runs and/or splits for each dataset
Splits = {'DTD': 10,
          'MINC_2500': 5,
          'GTOS-mobile': 5,
          'KTH_TIPS': 4}

Dataset = Dataset_names[data_selection]
data_dir = Data_dirs[Dataset]

Hist_model_name = 'HistRes_' + str(numBins)

#Return dictionary of parameters
Network_parameters = {'save_results': save_results,'folder': folder,
                      'histogram': histogram,'Dataset': Dataset, 'data_dir': data_dir,
                      'num_workers': num_workers, 'mode': mode,'new_lr': new_lr,
                      'pt_lr': pt_lr,'divergence_method': divergence_method, 'dof': dof, 'alpha': alpha, 'weights': weights,
                      'step_size': step_size,
                      'gamma': gamma, 'batch_size' : batch_size, 
                      'num_epochs': num_epochs, 'resize_size': resize_size, 
                      'center_size': center_size, 'padding': padding, 
                      'stride': stride, 'kernel_size': kernel_size,
                      'in_channels': in_channels, 'out_channels': out_channels,
                      'normalize_count': normalize_count, 
                      'normalize_bins': normalize_bins,'parallel': parallel,
                      'numBins': numBins,'feat_map_size': feat_map_size,
                      'Model_names': Model_names, 'num_classes': num_classes, 
                      'Splits': Splits, 'feature_extraction': feature_extraction,
                      'hist_model': Hist_model_name, 'use_pretrained': use_pretrained,
                      'add_bn': add_bn, 'pin_memory': pin_memory, 'scale': scale,
                      'degrees': degrees, 'rotation': rotation, 'embed_dim': embed_dim,
                      'TSNE_visual': TSNE_visual,
                      'Separate_TSNE': Separate_TSNE, 'Parallelize': Parallelize_model,
                      'Num_TSNE_images': Num_TSNE_images,'fig_size': fig_size,
                      'font_size': font_size}