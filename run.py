# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 20:43:42 2021

@author: groes
"""

import preprocessing as prep
import dataexploration as dx
import torch
import models
import gym
from sklearn import metrics
import dataset as dataclass
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
import utils 
import matplotlib.pyplot as plt
import device_param
#from focal_loss.focal_loss import FocalLoss
import end_start_tokens as tokens


def run_experiment(params):
    
    """
    This method executes an experiment. It creates all the objects needed for
    the experiment and calls training and test methods in gym.py. 
    
    """
    print(params["experiment_name"])
    print("Description of experiment: \n {}".format(params["experiment_desc"]))
    print("The hyperparameters used are: \n {}".format(params))
    
    train_loader, valid_loader, test_loader = prep.get_data(params)
    
    try:
        filename = params["experiment_name"] + "_testloader.pt"
        torch.save(test_loader, filename)
    except:
        print("Could not save test laoder")
    
    print("Creating model")
    if params["big_punc"]:
        model = models.BIG_PUNC_LSTM(params["num_embeddings"],
                                     params["embedding_dim"],
                                     params["hidden_size"], 
                                     params["lstm_layers"])
    else:
        model = models.PUNC_LSTM(params["num_embeddings"],
                                 params["embedding_dim"],
                                 params["hidden_size"],
                                 params["lstm_layers"])
    model.to(device_param.DEVICE)
    
    if params["loss"] == "cel":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=tokens.IGNORE_ID)
    if params["loss"] == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2)
        
    if params["lr"] == "cyclic":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.0001,
                                    momentum=0.9)
    elif params["lr"] == "on_plateau":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,
                              weight_decay=params["weight_decay"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"],
                                      weight_decay=params["weight_decay"])
        
    print("Starting training")
    model, avg_train_losses, avg_valid_losses = gym.train_punc_lstm(
        model, train_loader, valid_loader, params["patience"], criterion, optimizer,
        params["max_norm"], params["epochs"], params)
    
    # Testing
    print("Starting testing")
    labels, predicts, _ = gym.test_punc_lstm(model, test_loader)
    precision, recall, fscore, support = score(labels, predicts)
    accuracy = accuracy_score(labels, predicts)
    
    
    # Printing stuff because print outs are automatically added to output file on 
    # Camber server
    print("-- MODEL CHARACTERISTICS -- \n {}".format(str(model)))
    print_msg = "-- MODEL PERFORMANCE -- \n"\
    "Precision non-comma: {} % \n" \
    "Precision comma: {} % \n"\
    "Recall non-comma: {} % \n"\
    "Recall comma: {} % \n"\
    "F score non-comma: {} \n"\
    "F score comma: {} \n"\
    "Support non-comma: {} \n"\
    "Support non-comma: {} \n"\
    "Accuracy: {} % ".format(round(precision[0]*100, 2), round(precision[1]*100, 2),
                             round(recall[0]*100, 2), round(recall[1]*100, 2), 
                             round(fscore[0], 2), round(fscore[1], 2),
                             round(support[0], 2), support[1], round(accuracy*100, 2))
    print(print_msg)
    print(metrics.classification_report(labels, predicts))
    
    # Making and plotting confusion matrix
    plotname = "conf_mat_" + params["experiment_name"] + str(type(model))[8:-2].replace(".", "_") + ".png"
    try:
        conf_mat = utils.generate_confusion_matrix(torch.tensor(predicts),
                                                   torch.tensor(labels))
        print("-- CONFUSION MATRIX --")
        print(conf_mat)
    except:
        print("Could not generate confusion matrix")
    try:
        utils.plot_confusion_matrix(conf_mat, ["Non-comma", "Comma"])
        plt.savefig(plotname)
    except:
        print("Confusion matrix not saved. Couldn't plot or save confusion matrix")
    
    
    # Plotting training and validation loss against epochs
    # Source:
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/MNIST_Early_Stopping_example.ipynb
    try:
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses, label='Training Loss')
        plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
        # find position of lowest validation loss
        minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
        plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, 0.5) # consistent scale
        plt.xlim(0, len(avg_train_losses)+1) # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plotname = "loss_plot" + params["experiment_name"] + ".png"
        fig.savefig(plotname, bbox_inches='tight')
    except:
        print("Couldn't generate or save loss plot")
    
    try:
        print("-- DATASET SIZES --\n" \
          "No. of training examples: {} \n" \
          "No. of validation examples: {} \n" \
          "No. of test examples: {} ".format(
              len(train_loader)*params["batchsize"],
              len(valid_loader)*params["batchsize"],
              len(test_loader)*params["batchsize"]
              ))
    except:
        pass