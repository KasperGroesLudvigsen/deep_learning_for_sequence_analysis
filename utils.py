# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 19:46:21 2021

@author: groes
"""

from pathlib import Path
from os import listdir
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools
import os

def save_file(filename, object_to_save):
    """ 
    filename shouls be string, object_to_save should be list of strings
    """
    with open(filename, 'w') as f:
            for item in object_to_save:
                f.write("%s\n" % item)
                
def print_size_of_model(model):
    # Source: https://discuss.pytorch.org/t/dynamic-quantization-not-reducing-model-size/59531
    torch.save(model.state_dict(), "temp.pt")
    print('Size (MB):', os.path.getsize("temp.pt")/1e6)
    os.remove('temp.pt')
    
def get_path_to_files(folder_name):
    """
    Function to create a list containing paths to all files in a folder.

    Parameters
    ----------
    folder_name : STR
        Name of folder

    Returns
    -------
    paths : LIST
        List of paths to all files in a folder. The paths are WindowsPath objects
        of the pathlib module

    """
    data_folder = Path(folder_name)
    file_names = listdir(folder_name)
    paths = []
    
    for name in file_names:
        file = data_folder / name
        paths.append(file)
    
    return paths



def generate_confusion_matrix(all_predicted, all_labels):
    cat_all_predicted = torch.cat((all_predicted[0], all_predicted[1]))
    for tensor in all_predicted[2:]:
        cat_all_predicted = torch.cat((cat_all_predicted, tensor))
    
    cat_all_labels = torch.cat((all_labels[0], all_labels[1]))
    for tensor in all_labels[2:]:
        cat_all_labels = torch.cat((cat_all_labels, tensor))

    confmatrix = confusion_matrix(cat_all_predicted, cat_all_labels) 
    return confmatrix, cat_all_predicted, cat_all_labels

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Source: https://deeplizard.com/learn/video/0LhiS6yu2qQ 
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_train_val_loss(avg_train_loss, avg_valid_loss, model_name):
    # Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1,len(avg_train_loss)+1),avg_train_loss, label='Training Loss')
    plt.plot(range(1,len(avg_valid_loss)+1),avg_valid_loss,label='Validation Loss')
    # find position of lowest validation loss
    minposs = avg_valid_loss.index(min(avg_valid_loss))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5) # consistent scale
    plt.xlim(0, len(avg_train_loss)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    figure_name = " loss_plot_" + model_name + ".png"
    fig.savefig(figure_name, bbox_inches='tight')