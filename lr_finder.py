# -*- coding: utf-8 -*-
"""
Created on Fri May 28 12:23:11 2021

@author: groes
"""

from torch_lr_finder import PUNC_LSTM_LRFinder
import models
import torch.nn as nn
import preprocessing as prep
import torch

params = {
    "experiment_desc" : "Baseline model but with cyclical lr",
    "experiment_name" : "experiment8",
    "interval" : (0, 45), # includes sentences of length longer than 0th index, shorter than 1st index
    "test_size" : 0.2,
    "validation_size" : 0.2,
    "batchsize" : 128, # original paper: 128
    "token_type" : "class", # should be "class" for word class or "lemma" for word lemma
    "corpus" : "1990", # should be "1990" or "2010". I think 1990 is better because it contains fewer texts from digital media like personal blogs with poor comma setting
    "skip_no_comma" : False, # if True, sentences that do not contain commas are not part of dataset
    "comma_representation" : ",", # making this a variable in case it needs to change at some point
    "max_norm" : 5,
    "lr" : "cyclic",
    "weight_decay" : 0.0,
    "patience" : 5, # if no improvement in validation loss after 'patience' epochs, training is terminated
    "epochs" : 100,
    "num_embeddings" : 100002,
    "embedding_dim" : 256,
    "hidden_size" : 512,
    "loss" : "cel" # should either be "focal" or "cel"
    }

train_loader, _, _ = prep.get_data(params)

model = models.PUNC_LSTM(params["num_embeddings"], params["embedding_dim"],
                             params["hidden_size"], 2)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
print("Starting lr range test")
lr_finder = PUNC_LSTM_LRFinder(model, optimizer, criterion, device="cpu")
lr_finder.range_test(train_loader, end_lr=10, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state


# Testing
#i, data = next(enumerate(trainloader))
#padded_input, input_lengths, padded_target = data

#train_iter = TrainDataLoaderIter(trainloader)
#inputs, labels = next(train_iter)

#from torch.utils.data import DataLoader

#isinstance(trainloader, DataLoader)

"""
class DataLoaderIter(object):
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self._iterator = iter(data_loader)

    @property
    def dataset(self):
        return self.data_loader.dataset

    def inputs_labels_from_batch(self, batch_data):
        if not isinstance(batch_data, list) and not isinstance(batch_data, tuple):
            raise ValueError(
                "Your batch type is not supported: {}. Please inherit from "
                "`TrainDataLoaderIter` or `ValDataLoaderIter` and override the "
                "`inputs_labels_from_batch` method.".format(type(batch_data))
            )

        inputs, labels, *_ = batch_data

        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self._iterator)
        return self.inputs_labels_from_batch(batch)




class TrainDataLoaderIter(DataLoaderIter):
    def __init__(self, data_loader, auto_reset=True):
        super().__init__(data_loader)
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)
        except StopIteration:
            if not self.auto_reset:
                raise
            self._iterator = iter(self.data_loader)
            batch = next(self._iterator)
            inputs, labels = self.inputs_labels_from_batch(batch)

        return inputs, labels
    
  

"""








