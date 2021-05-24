# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:01:36 2019

@author: jpeeples
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import pdb
import torch
import torchvision
import numpy as np
import itertools


class KTH_TIPS_2b_data(Dataset):

    def __init__(self, data_dir, train=True, img_transform=None, 
                 train_setting=None, test_setting=None):

        self.data_dir = data_dir
        self.img_transform = img_transform
        self.train_setting = train_setting
        self.test_setting = test_setting
        self.files = []
        self.targets = []

        imgset_dir = os.path.join(self.data_dir, 'Images')
        # indexing variable for label
        temp_label = 0
        for file in os.listdir(imgset_dir):
            if not file.startswith('.'):
                # Set class label
                label_name = file
                # Look inside each folder and grab samples
                texture_dir = os.path.join(imgset_dir, label_name)
                # pdb.set_trace()
                if train:
                    for ii in range(0, len(train_setting)):
                        # Only look at training samples of interest
                        sample_dir = os.path.join(texture_dir, 'sample_' + str(''.join(train_setting[ii])))
                        for image in os.listdir(sample_dir):
                            if not image.startswith('.'):
                                img_file = os.path.join(sample_dir, image)
                                label = temp_label
                                self.files.append({
                                    "img": img_file,
                                    "label": label
                                })
                                self.targets.append(label)
                else:
                    for ii in range(0, len(test_setting)):
                        # Only look at testing samples of interest
                        sample_dir = os.path.join(texture_dir, 'sample_' + str(''.join(test_setting[ii])))
                        for image in os.listdir(sample_dir):
                            if not image.startswith('.'):
                                img_file = os.path.join(sample_dir, image)
                                label = temp_label
                                self.files.append({
                                    "img": img_file,
                                    "label": label
                                })
                                self.targets.append(label)
                temp_label = temp_label + 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = torch.tensor(label_file)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label, index