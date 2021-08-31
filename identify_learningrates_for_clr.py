# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:08:23 2021

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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from pytorchtools import EarlyStopping
import numpy as np
import dataset as dataclass
import device_param
import time
from torch.optim import lr_scheduler
import end_start_tokens as tokens
import matplotlib.pyplot as plt
import seaborn as sns

params = {
    "experiment_desc" : "First run (Baseline model)",
    "experiment_number" : "clr_boundaries",
    "interval" : (0, 45), # includes sentences of length longer than 0th index, shorter than 1st index
    "test_size" : 0.2,
    "validation_size" : 0.2,
    "batchsize" : 128, # original paper: 128
    "token_type" : "class", # should be "class" for word class or "lemma" for word lemma
    "corpus" : "1990", # should be "1990" or "2010". I think 1990 is better because it contains fewer texts from digital media like personal blogs with poor comma setting
    "skip_no_comma" : False, # if True, sentences that do not contain commas are not part of dataset
    "comma_representation" : ",", # making this a variable in case it needs to change at some point
    "max_norm" : 5,
    "lr" : 1e-3,
    "weight_decay" : 0.0,
    "patience" : 5, # if no improvement in validation loss after 'patience' epochs, training is terminated
    "epochs" : 1,
    "num_embeddings" : 100002,
    "embedding_dim" : 256,
    "hidden_size" : 512,
    "loss" : "cel" # should either be "focal" or "cel"
    }


# Define model and optimizer
model = models.PUNC_LSTM(params["num_embeddings"], params["embedding_dim"],
                         params["hidden_size"], 2)

train_loader, valid_loader, test_loader = prep.get_data(params)

model.to(device_param.DEVICE)

if params["loss"] == "cel":
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokens.IGNORE_ID)
if params["loss"] == "focal":
    criterion = FocalLoss(alpha=0.25, gamma=2)
    

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, base_lr=0.0000001, max_lr=1, step_size_up=100, mode="triangular")

learning_rates = []

total_loss = 0
start = time.time()
# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = [] 
    

for epoch in range(1):
    print("Starting epoch number {}".format(epoch+1))
    # Training
    model.train()
    for i, (data) in enumerate(train_loader):
        optimizer.zero_grad()
        padded_input, input_lengths, padded_target = data
        padded_input, input_lengths, padded_target = \
            padded_input.to(device_param.DEVICE), input_lengths.to(device_param.DEVICE),\
                padded_target.to(device_param.DEVICE)
        pred = model(padded_input, input_lengths) 
        pred = pred.view(-1, pred.size(-1))
        loss = criterion(pred, padded_target.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        train_losses.append(loss.item())
        total_loss += loss.item()
        
        # PUtting them here instead of in validation loop
        learning_rates.append(scheduler.get_lr())
        scheduler.step(loss) # take learning rate scheduler step
        valid_losses.append(loss.item())

   
learning_rates = np.array(learning_rates)
valid_losses = np.array(valid_losses)

np.save("clr_learning_rates", learning_rates)
np.save("clr_valid_losses", valid_losses)

len(np.unique(learning_rates))

plt.plot(learning_rates, valid_losses)
plt.plot(learning_rates)
plt.plot(valid_losses)

plt.scatter(learning_rates, valid_losses, s=0.05)
plt.ylabel("Loss")
plt.xlabel("Learning rate")
plt.title("Loss as learning rate increases")
sns.scatterplot()
