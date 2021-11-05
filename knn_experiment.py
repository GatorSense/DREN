# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:15:33 2019
Generate results from saved models
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.cm as colormap
from sklearn.manifold import TSNE
import os
import pickle

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Texture_information import Class_names
from Demo_Parameters import Network_parameters as Results_parameters
from Utils.Network_functions import initialize_model
from Prepare_Data import Prepare_DataLoaders
from Utils.RBFHistogramPooling import HistogramLayer
from Out_of_Sample import embed_out_of_sample

# Location of experimental results
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fig_size = Results_parameters['fig_size']
font_size = Results_parameters['font_size']

# Set up number of runs and class/plots names
NumRuns = Results_parameters['Splits'][Results_parameters['Dataset']]
plot_name = Results_parameters['Dataset'] + ' Test Confusion Matrix'
avg_plot_name = Results_parameters['Dataset'] + ' Test Average Confusion Matrix'
class_names = Class_names[Results_parameters['Dataset']]

# Name of dataset
Dataset = Results_parameters['Dataset']

# Model(s) to be used
model_name = Results_parameters['Model_names'][Dataset]

# Number of classes in dataset
num_classes = Results_parameters['num_classes'][Dataset]

# Number of runs and/or splits for dataset
numRuns = Results_parameters['Splits'][Dataset]

# Number of bins and input convolution feature maps after channel-wise pooling
numBins = Results_parameters['numBins']
num_feature_maps = Results_parameters['out_channels'][model_name]

# Local area of feature map after histogram layer
feat_map_size = Results_parameters['feat_map_size']
# Parse through files and plot results
#for perplexity in Results_parameters['perplexity']:
for divergence_method in Results_parameters['divergence_method']:
    for alpha in Results_parameters['alpha']:
        for dimension in Results_parameters['embed_dim']:
            weight_0_accuracies = []
            for weight in Results_parameters['weights']:
                # Set directory location for experiments
                if (Results_parameters['histogram']):
                    if (Results_parameters['parallel']):
                        weight_dir = (Results_parameters['folder'] + '/' + Results_parameters['mode']
                                      + '/' + Results_parameters['Dataset'] + '/'
                                      + Results_parameters[
                                          'hist_model']  + '/' + divergence_method + '/' + 'alpha_' + str(
                                    alpha) + '/Weight_' + str(weight)
                                      + '/Embed_' + str(dimension) + 'D'
                                      + '/Parallel/')

                    else:
                        weight_dir = (Results_parameters['folder'] + '/' + Results_parameters['mode']
                                      + '/' + Results_parameters['Dataset'] + '/'
                                      + Results_parameters[
                                          'hist_model']  + '/' + divergence_method + '/' + 'alpha_' + str(
                                    alpha) + '/Weight_' +
                                      str(weight) + '/Embed_' + str(dimension)
                                      + 'D' '/Inline/')
                # Baseline model
                else:
                    weight_dir = (Results_parameters['folder'] + '/' + Results_parameters['mode']
                                  + '/' + Results_parameters['Dataset'] + '/GAP_' +
                                  Results_parameters['Model_names'][Results_parameters['Dataset']]
                                  + '/' + divergence_method + '/' + 'alpha_' + str(
                                alpha) + '/Weight_' + str(
                                weight) + '/Embed_' +
                                  str(dimension) + 'D/')
                model_accuracies = []
                tsne_accuracies = []
                tsne_accuracies_train = []
                model_accuracies_train = []
                for split in range(0, NumRuns):

                    sub_dir = weight_dir + 'Run_' + str(split + 1) + '/'
                    # Load training and testing files (Python)
                    train_pkl_file = open(sub_dir + 'train_dict.pkl', 'rb')
                    train_dict = pickle.load(train_pkl_file)
                    train_pkl_file.close()

                    test_pkl_file = open(sub_dir + 'test_dict.pkl', 'rb')
                    test_dict = pickle.load(test_pkl_file)
                    test_pkl_file.close()

                    # Remove pickle files
                    del train_pkl_file, test_pkl_file

                    # #Load model
                    histogram_layer = HistogramLayer(int(num_feature_maps / (feat_map_size * numBins)),
                                                     Results_parameters['kernel_size'][model_name],
                                                     num_bins=numBins, stride=Results_parameters['stride'],
                                                     normalize_count=Results_parameters['normalize_count'],
                                                     normalize_bins=Results_parameters['normalize_bins'])

                    # Initialize the histogram model for this run
                    model, input_size = initialize_model(model_name, num_classes,
                                                         Results_parameters['in_channels'][model_name],
                                                         num_feature_maps,
                                                         feature_extract=Results_parameters['feature_extraction'],
                                                         histogram=Results_parameters['histogram'],
                                                         histogram_layer=histogram_layer,
                                                         parallel=Results_parameters['parallel'],
                                                         use_pretrained=Results_parameters['use_pretrained'],
                                                         add_bn=Results_parameters['add_bn'],
                                                         scale=Results_parameters['scale'],
                                                         feat_map_size=feat_map_size,
                                                         embed_dim=dimension)

                    # Set device to cpu or gpu (if available)
                    device_loc = torch.device(device)

                    # If parallelized, need to set change model
                    if Results_parameters['Parallelize']:
                        model = nn.DataParallel(model)

                    model.load_state_dict(torch.load(sub_dir + 'Best_Weights.pt'
                                                     , map_location=device_loc))

                    #removes fc layer
                    model_no_fc = torch.nn.Sequential(*(list(model.children())[:-1]))
                    model = model.to(device)
                    model_no_fc = model_no_fc.to(device)


                    dataloaders_dict = Prepare_DataLoaders(Results_parameters, split,
                                                           input_size=input_size)
                    model.eval()
                    model_no_fc.eval()
                    model_tsne = model.module.features
                    model_tsne.eval()
                    model_tsne.to(device)
                    embedding = []
                    features_extracted_tsne = []
                    GT_val = np.array(0)
                    #gets training features
                    for idx, (inputs, classes, index) in enumerate(dataloaders_dict['train']):
                        images = inputs.to(device)
                        labels = classes.to(device, torch.long)
                        GT_val = np.concatenate((GT_val, labels.cpu().numpy()), axis=None)

                        _, temp_embedding, _ = model(inputs)
                        embedding.append(temp_embedding.detach().cpu().numpy())

                        features_tsne = model_tsne(images)
                        features_tsne = torch.flatten(features_tsne, start_dim=1)
                        features_tsne = features_tsne.cpu().detach().numpy()
                        features_extracted_tsne.extend(features_tsne)

                    #find training manifold from tsne
                    features_embedded_train = TSNE(n_components=3, verbose=1, init='random',
                                             random_state=42).fit_transform(features_extracted_tsne)
                    fig6 = plt.figure()
                    if dimension == 3:
                        ax6 = fig6.add_subplot(1, 1, 1, projection='3d')
                    else:
                        ax6 = fig6.add_subplot(1, 1,1)
                    colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
                    for texture in range(0, len(class_names)):
                        x = features_embedded_train[[np.where(GT_val[1:] == texture)], 0]
                        y = features_embedded_train[[np.where(GT_val[1:] == texture)], 1]

                        if dimension == 2:
                            ax6.scatter(x, y, color=colors[texture, :],
                                        label=class_names[texture])
                        else:  # 3D
                            z = features_embedded_train[[np.where(GT_val[1:] == texture)], 2]
                            ax6.scatter(x, y, z, color=colors[texture, :],
                                        label=class_names[texture])

                    plt.title('TSNE Visualization of Training Data Features')
                    #plt.legend(class_names,loc='lower right')

                    box = ax6.get_position()
                    ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
                    ax6.legend(loc='upper center', bbox_to_anchor=(.5, -.05), fancybox=True, ncol=4)
                    plt.axis('off')

                    fig6.savefig((sub_dir + 'TSNE_Visual_Train_Data_FC.png'), dpi=fig6.dpi)
                    plt.close()

                    fig6 = plt.figure()
                    if dimension == 3:
                        ax6 = fig6.add_subplot(1, 1, 1, projection='3d')
                    else:
                        ax6 = fig6.add_subplot(1, 1, 1)
                    colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
                    embedding = np.concatenate(embedding, axis=0)
                    for texture in range(0, len(class_names)):
                        x = embedding[[np.where(GT_val[1:] == texture)], 0]
                        y = embedding[[np.where(GT_val[1:] == texture)], 1]

                        if dimension == 2:
                            ax6.scatter(x, y, color=colors[texture, :],
                                        label=class_names[texture])
                        else:  # 3D
                            z = embedding[[np.where(GT_val[1:] == texture)], 2]
                            ax6.scatter(x, y, z, color=colors[texture, :],
                                        label=class_names[texture])

                    plt.title('Embedding Visualization of Training Data Features')
                    # plt.legend(class_names,loc='lower right')

                    box = ax6.get_position()
                    ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
                    ax6.legend(loc='upper center', bbox_to_anchor=(.5, -.05), fancybox=True, ncol=4)
                    plt.axis('off')

                    fig6.savefig((sub_dir + 'Embedding_Visual_Train_Data_FC.png'), dpi=fig6.dpi)
                    plt.close()

                    #train KNN from model
                    knn_model = KNeighborsClassifier(n_neighbors=3)
                    knn_model.fit(embedding,GT_val[1:])
                    #train KNN from tsne
                    knn_tsne = KNeighborsClassifier(n_neighbors=3)
                    knn_tsne.fit(features_embedded_train, GT_val[1:])

                    predicted_tsne_train = knn_tsne.predict(features_embedded_train)
                    accuracy_tsne_train = len(np.where(predicted_tsne_train == GT_val[1:])[0]) / len(predicted_tsne_train)
                    tsne_accuracies_train.append(accuracy_tsne_train)
                    print("Accuracy_tsne_train: " + str(accuracy_tsne_train))

                    predicted_model_train = knn_model.predict(embedding)
                    accuracy_model_train = len(np.where(predicted_model_train == GT_val[1:])[0]) / len(
                        predicted_model_train)
                    model_accuracies_train.append(accuracy_model_train)
                    print("Accuracy_model_train: " + str(accuracy_model_train))

                    #test each
                    features_extracted_tsne_test = []
                    embedding_test = []
                    num_correct = 0
                    num_total = 0
                    GT_val = np.array(0)
                    for idx, (inputs, classes, index) in enumerate(dataloaders_dict['test']):
                        images = inputs.to(device)
                        labels = classes.to(device, torch.long)
                        GT_val_test = labels.cpu().numpy()
                        GT_val = np.concatenate((GT_val, labels.cpu().numpy()), axis=None)

                        _, temp_embedding, _ = model(inputs)
                        temp_embedding = temp_embedding.detach().cpu().numpy()
                        embedding_test.append(temp_embedding)

                        features_tsne = model_tsne(images)
                        features_tsne = torch.flatten(features_tsne, start_dim=1)
                        features_tsne = features_tsne.cpu().detach().numpy()
                        features_extracted_tsne_test.extend(features_tsne)

                        predicted = knn_model.predict(temp_embedding)
                        num_correct = num_correct + len(np.where(predicted == GT_val_test)[0])
                        num_total = num_total + len(GT_val_test)
                    features_embedded_test = embed_out_of_sample(np.array(features_extracted_tsne),features_embedded_train,np.array(features_extracted_tsne_test),3,1,3)
                    #displaying tsne test
                    fig6 = plt.figure()
                    if dimension == 3:
                        ax6 = fig6.add_subplot(1, 1, 1, projection='3d')
                    else:
                        ax6 = fig6.add_subplot(1, 1, 1)
                    colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
                    for texture in range(0, len(class_names)):
                        x = features_embedded_test[[np.where(GT_val[1:] == texture)], 0]
                        y = features_embedded_test[[np.where(GT_val[1:] == texture)], 1]

                        if dimension == 2:
                            ax6.scatter(x, y, color=colors[texture, :],
                                        label=class_names[texture])
                        else:  # 3D
                            z = features_embedded_test[[np.where(GT_val[1:] == texture)], 2]
                            ax6.scatter(x, y, z, color=colors[texture, :],
                                        label=class_names[texture])

                    plt.title('TSNE Visualization of Test Data Features')
                     #plt.legend(class_names,loc='lower right')

                    box = ax6.get_position()
                    ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
                    ax6.legend(loc='upper center', bbox_to_anchor=(.5, -.05), fancybox=True, ncol=4)
                    plt.axis('off')

                    fig6.savefig((sub_dir + 'TSNE_Visual_Test_Data_FC.png'), dpi=fig6.dpi)
                    plt.close()

                    # displaying embedding test
                    embedding_test = np.concatenate(embedding_test, axis=0)
                    fig6 = plt.figure()
                    if dimension == 3:
                        ax6 = fig6.add_subplot(1, 1, 1, projection='3d')
                    else:
                        ax6 = fig6.add_subplot(1, 1, 1)
                    colors = colormap.rainbow(np.linspace(0, 1, len(class_names)))
                    for texture in range(0, len(class_names)):
                        x = embedding_test[[np.where(GT_val[1:] == texture)], 0]
                        y = embedding_test[[np.where(GT_val[1:] == texture)], 1]

                        if dimension == 2:
                            ax6.scatter(x, y, color=colors[texture, :],
                                        label=class_names[texture])
                        else:  # 3D
                            z = embedding_test[[np.where(GT_val[1:] == texture)], 2]
                            ax6.scatter(x, y, z, color=colors[texture, :],
                                        label=class_names[texture])

                    plt.title('Embedding Visualization of Test Data Features')
                    # plt.legend(class_names,loc='lower right')

                    box = ax6.get_position()
                    ax6.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
                    ax6.legend(loc='upper center', bbox_to_anchor=(.5, -.05), fancybox=True, ncol=4)
                    plt.axis('off')

                    fig6.savefig((sub_dir + 'Embedding_Visual_Test_Data_FC.png'), dpi=fig6.dpi)
                    plt.close()

                    predicted_tsne = knn_tsne.predict(features_embedded_test)
                    accuracy_tsne = len(np.where(predicted_tsne == GT_val[1:])[0])/len(predicted_tsne)
                    print("Accuracy_tsne: " + str(accuracy_tsne))
                    accuracy_model = num_correct/num_total
                    print("Accuracy_model: " + str(accuracy_model))
                    model_accuracies.append(accuracy_model)
                    tsne_accuracies.append(accuracy_tsne)
                directory = os.path.dirname(os.path.dirname(sub_dir)) + '/'
                # Write to text file
                with open((directory + 'FC_Accuracy.txt'), "w") as output:
                    output.write('Average accuracy for test model: ' + str(np.mean(model_accuracies)) + ' Std: ' + str(np.std(model_accuracies)))
                    output.write('\n Average accuracy for train model: ' + str(np.mean(model_accuracies_train)) + ' Std: ' + str(np.std(model_accuracies_train)))
                    output.write('\n Average accuracy for test tsne: ' + str(np.mean(tsne_accuracies)) + ' Std: ' + str(np.std(tsne_accuracies)))
                    output.write('\n Average accuracy for train tsne: ' + str(np.mean(tsne_accuracies_train)) + ' Std: ' + str(np.std(tsne_accuracies_train)))


