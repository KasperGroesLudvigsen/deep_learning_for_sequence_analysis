# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:21:16 2021

@author: groes
"""
import torch
#DEVICE = torch.device('cpu')
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("DEVICE is: {}".format(DEVICE))



