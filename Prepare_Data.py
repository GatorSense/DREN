# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:07:33 2019
Load datasets for models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import itertools
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
## PyTorch dependencies
import torch
from torchvision import transforms

## Local external libraries
from Datasets.DTD_loader import DTD_data
from Datasets.MINC_2500 import MINC_2500_data
from Datasets.GTOS_mobile_single_size import GTOS_mobile_single_data
from Datasets.KTH_TIPS_2b import KTH_TIPS_2b_data


def Prepare_DataLoaders(Network_parameters, split,input_size=224):
    
    Dataset = Network_parameters['Dataset']
    data_dir = Network_parameters['data_dir']
    
    # Data augmentation and normalization for training
    # Just normalization and resize for test
    # Data transformations as described in:
    # http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf
    if not(Network_parameters['rotation']):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(Network_parameters['resize_size']),
                transforms.RandomResizedCrop(input_size,scale=(.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(Network_parameters['center_size']),
                transforms.CenterCrop(input_size),
                transforms.RandomAffine(Network_parameters['degrees']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    
        # Create training and test datasets
    if Dataset=='DTD':
        train_val_dataset = DTD_data(data_dir, data='train and val',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        validation_dataset = DTD_data(data_dir, data='val',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        test_dataset = DTD_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['test'])

        indices = np.arange(len(train_val_dataset))
        y = [sub['label'] for sub in train_val_dataset.files]

        # Use stratified split to balance training validation splits, set random state to be same for each encoding method
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, stratify=y, test_size=.1, random_state=10)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}
        
    elif Dataset == 'MINC_2500':
        train_dataset = MINC_2500_data(data_dir, data='train',
                                           numset = split + 1,
                                           img_transform=data_transforms['train'])
        val_dataset = MINC_2500_data(data_dir, data='val',
                                           numset = split + 1,
                                           img_transform=data_transforms['test'])
        test_dataset = MINC_2500_data(data_dir, data = 'test',
                                           numset = split + 1,
                                           img_transform=data_transforms['test'])
    
    elif Dataset == 'KTH_TIPS': #Didn't use any data augmentation for initial experiments
        samples = ['a', 'b', 'c', 'd']
        # Set to 1 to train on 1, test 3; set to 2 to train on 2, test on 2;
        # set to 3 to train on 3 and test 1
        setting = 3
        
        sample_combos = list(itertools.combinations(samples, setting))
        train_setting = []
        test_setting = []
        for ii in range(0, len(sample_combos)):
            train_setting.append(list(sample_combos[ii]))
            test_setting.append(list(sorted(set(samples) - set(sample_combos[ii]))))
         
        train_val_dataset = KTH_TIPS_2b_data(data_dir,train=True,
                                         img_transform=data_transforms['train'],
                                         train_setting=train_setting[split])
        test_dataset = KTH_TIPS_2b_data(data_dir,train=False,
                                         img_transform=data_transforms['test'],
                                         test_setting=test_setting[split])
        indices = np.arange(len(train_val_dataset))
        y = [sub['label'] for sub in train_val_dataset.files]

        # Use stratified split to balance training validation splits, set random state to be same for each encoding method
        _, _, _, _, train_indices, val_indices = train_test_split(y, y, indices, stratify=y, test_size=.1,
                                                                  random_state= 10)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        dataset_sampler = {'train': train_sampler, 'val': val_sampler, 'test': None}

    else: #Need to create separate validation dataset from training
        # Create training and test datasets
        train_dataset = GTOS_mobile_single_data(data_dir, train = True,
                                           image_size=Network_parameters['resize_size'],
                                           img_transform=data_transforms['train']) 
        val_dataset = GTOS_mobile_single_data(data_dir, train = False,
                                           img_transform=data_transforms['test'])
        test_dataset = GTOS_mobile_single_data(data_dir, train = False,
                                           img_transform=data_transforms['test'])

    if((Dataset == 'KTH_TIPS') | (Dataset == 'DTD')):
        image_datasets = {'train': train_val_dataset, 'val': train_val_dataset, 'test': test_dataset}
        # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           sampler=dataset_sampler[x],
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
                            for x in ['train', 'val', 'test']}
        return dataloaders_dict
    else:

        image_datasets = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
        # Create training and test dataloaders
        dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                           batch_size=Network_parameters['batch_size'][x],
                                                           shuffle=True,
                                                           num_workers=Network_parameters['num_workers'],
                                                           pin_memory=Network_parameters['pin_memory'])
                                                           for x in ['train', 'val','test']}

        return dataloaders_dict