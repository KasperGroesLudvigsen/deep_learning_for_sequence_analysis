# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 13:06:27 2021

@author: groes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import device_param
import math


class PUNC_LSTM(nn.Module):
    # Adapted from: https://github.com/kaituoxu/X-Punctuator/blob/master/src/model/model.py
    def __init__(self, num_embeddings, embedding_dim, hidden_size, lstm_layers, 
                 num_class=2, bidirectional=True, batch_first=True): 
        super(PUNC_LSTM, self).__init__()
        self.batch_first = batch_first
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_class = num_class
        fc_in_dim = hidden_size * 2 if bidirectional else hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, lstm_layers,
                            batch_first=batch_first, bidirectional=bidirectional)
        self.fc = nn.Linear(fc_in_dim, num_class)
        
    def forward(self, padded_input, input_lengths):
        padded_input = self.embedding(padded_input) # N x T x D (D = embedding_dim)
        total_length = padded_input.size(1)
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(), # .cpu() added because I got the error reported here: https://github.com/pytorch/pytorch/issues/43227
                                            batch_first=True)
        packed_input = packed_input.to(device_param.DEVICE)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=self.batch_first,
                                        total_length=total_length)
        score = self.fc(output)
        return score
    
class BIG_PUNC_LSTM(nn.Module):
    # Adapted from: https://github.com/kaituoxu/X-Punctuator/blob/master/src/model/model.py
    def __init__(self, num_embeddings, embedding_dim, hidden_size, lstm_layers=3, 
                 num_class=2, bidirectional=True, batch_first=True): # num_class = 2 = {comma, no comma}
        super(BIG_PUNC_LSTM, self).__init__()
        self.batch_first = batch_first
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_class = num_class
        fc_in_dim = hidden_size * 2 if bidirectional else hidden_size
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, lstm_layers,
                            batch_first=batch_first, bidirectional=bidirectional)
        self.fc1 = nn.Linear(fc_in_dim, num_class)
        self.dropout = nn.Dropout()
        self.activation_func = nn.ReLU()

        
    def forward(self, padded_input, input_lengths):
        padded_input = self.embedding(padded_input) # N x T x D (D = embedding_dim)
        total_length = padded_input.size(1)
        packed_input = pack_padded_sequence(padded_input, input_lengths.cpu(), # .cpu() added because I got the error reported here: https://github.com/pytorch/pytorch/issues/43227
                                            batch_first=True)
        packed_input = packed_input.to(device_param.DEVICE)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output,
                                        batch_first=self.batch_first,
                                        total_length=total_length)
        score = self.fc1(output)
        return score



