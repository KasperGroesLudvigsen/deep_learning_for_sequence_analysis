# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 16:42:08 2021

@author: groes
"""
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
    
def train_punc_lstm(model, train_loader, valid_loader, patience, criterion,
                    optimizer, max_norm, epochs, params, print_epochs=False):
    total_loss = 0
    model.to(device_param.DEVICE)
    start = time.time()
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    checkpoint_path = params["experiment_name"] + "checkpoint.pt"
    early_stopping = EarlyStopping(patience=patience, verbose=True,
                                   path=checkpoint_path)
    
    # Train and validation loop with ReduceLROnPlateau
    if params["lr"] == "on_plateau":
        print("Training with ReduceLROnPlateau")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=0.001)
        
        for epoch in range(1, epochs + 1):
            
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                train_losses.append(loss.item())
                total_loss += loss.item()
    
            # Validating
            model.eval()
            for i, (data) in enumerate(valid_loader):
                padded_input, input_lengths, padded_target = data
                padded_input, input_lengths, padded_target = \
                    padded_input.to(device_param.DEVICE), \
                        input_lengths.to(device_param.DEVICE), \
                            padded_target.to(device_param.DEVICE)
                pred = model(padded_input, input_lengths)
                pred = pred.view(-1, pred.size(-1))
                loss = criterion(pred, padded_target.view(-1))
                scheduler.step(loss) # take learning rate scheduler step
                valid_losses.append(loss.item())
            
            if i % 100000 == 0 and print_epochs:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch, i + 1, total_loss / (i+1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                          flush=True)
            #print("Finished training on batch")
            #break
            # print("Finished epoch # {}".format(epoch))
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break # only keep this break statement
            
            total_loss = total_loss / (i+1)
            #break
            
    # Train and validation loop with CyclicLR
    elif params["lr"] == "cyclic":
        print("Training with cyclic learning rate")
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=1e-7,
                                                      max_lr=0.002)
        
        for epoch in range(1, epochs + 1):
            print("Starting epoch number {}".format(epoch))
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scheduler.step(loss) # take learning rate scheduler step
                optimizer.step()
                train_losses.append(loss.item())
                total_loss += loss.item()
    
            # Validating
            model.eval()
            for i, (data) in enumerate(valid_loader):
                padded_input, input_lengths, padded_target = data
                padded_input, input_lengths, padded_target = \
                    padded_input.to(device_param.DEVICE), \
                        input_lengths.to(device_param.DEVICE), \
                            padded_target.to(device_param.DEVICE)
                pred = model(padded_input, input_lengths)
                pred = pred.view(-1, pred.size(-1))
                loss = criterion(pred, padded_target.view(-1))
                
                valid_losses.append(loss.item())
            
            if i % 100000 == 0 and print_epochs:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch, i + 1, total_loss / (i+1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                          flush=True)
            #print("Finished training on batch")
            #break
            # print("Finished epoch # {}".format(epoch))
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break # only keep this break statement
            
            total_loss = total_loss / (i+1)

    # Train and validation loop with constant learning rate
    else:
        print("Training with constant learning rate")
        for epoch in range(1, epochs + 1):
            print("Epoch number {}".format(epoch))
            model.train()
            for i, (data) in enumerate(train_loader):
                # Training
                optimizer.zero_grad()
                padded_input, input_lengths, padded_target = data
                padded_input, input_lengths, padded_target = \
                    padded_input.to(device_param.DEVICE), input_lengths.to(device_param.DEVICE),\
                        padded_target.to(device_param.DEVICE)
                pred = model(padded_input, input_lengths) 
                pred = pred.view(-1, pred.size(-1))
                loss = criterion(pred, padded_target.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                train_losses.append(loss.item())
                total_loss += loss.item()
    
            # Validating
            model.eval()
            for i, (data) in enumerate(valid_loader):
                padded_input, input_lengths, padded_target = data
                padded_input, input_lengths, padded_target = \
                    padded_input.to(device_param.DEVICE), \
                        input_lengths.to(device_param.DEVICE), \
                            padded_target.to(device_param.DEVICE)
                pred = model(padded_input, input_lengths)
                pred = pred.view(-1, pred.size(-1))
                loss = criterion(pred, padded_target.view(-1))
                valid_losses.append(loss.item())
            
            if i % 100000 == 0 and print_epochs:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch, i + 1, total_loss / (i+1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                          flush=True)
            #print("Finished training on batch")
            #break
            # print("Finished epoch # {}".format(epoch))
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break # only keep this break statement
            
            total_loss = total_loss / (i+1)
            #break
    
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))
    
    return model, avg_train_losses, avg_valid_losses

    
def test_punc_lstm(model, test_loader, inspect=False):
    model.to(device_param.DEVICE)
    
    labels = np.array([])
    predicts = np.array([])
    wrong_preds = [] # contrain sentences in which some prediction was wrong
    
    model.eval()
    
    with torch.no_grad():
        # adding "untokenized" in order to be able to inspect sentences that contain prediction errors
        for i, (word_id_seq, label_id_seq, untokenized) in enumerate(test_loader): 
            input_lengths = torch.LongTensor([len(word_id_seq)])
            input = word_id_seq.unsqueeze(0)
            input, input_lengths = input.to(device_param.DEVICE), input_lengths.to(device_param.DEVICE) 
            scores = model(input, input_lengths)
            scores = scores.view(-1, scores.size(-1))
            _, predict = torch.max(scores, 1)
            predict = predict.data.cpu().numpy()
            # adding this to be able to inspect
            if inspect and not np.array_equal(predict, label_id_seq):
                wrong_preds.append(untokenized)
            # accumulate
            assert(len(label_id_seq) == len(predict))
            labels = np.append(labels, label_id_seq)
            predicts = np.append(predicts, predict)
        assert(len(labels) == len(predicts))
        if i % 20000 == 0:
            print("Testing on test example # {}".format(i))
    
    return labels, predicts, wrong_preds


def train(model, training_data):
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #counter = 0
    for epoch in range(5): 
        for sentence, tags in training_data:#idx, (sentence, tags) in enumerate(trainloader): # #enumerate(trainloader): #trainingdata: # tags = 0 if token not followed by comma; 1 if it is
    
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            sentence = torch.tensor(sentence)
            tags = torch.tensor(tags)
            #counter += 1
            #if counter % 1000 == 0:
            #    print("{} training examples seen".format(counter))
            model.zero_grad()
    
    
            # Step 3. Run our forward pass.
            output = model(sentence)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(output, tags)
            loss.backward()
            optimizer.step()
    
    return model

def test(model, testdata):
    # Saving in order to calculate precision and recall
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        correct = 0
        total = 0
        for sentence, labels in testdata:
            sentence = torch.tensor(sentence)
            labels = torch.tensor(labels)
            outputs = model(sentence)
            _, predicted = torch.max(outputs.data, 1)
            all_predicted.append(predicted)
            all_labels.append(labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    return all_labels, all_predicted

def generate_confusion_matrix(all_predicted, all_labels):
    cat_all_predicted = torch.cat((all_predicted[0], all_predicted[1]))
    for tensor in all_predicted[2:]:
        cat_all_predicted = torch.cat((cat_all_predicted, tensor))
    
    cat_all_labels = torch.cat((all_labels[0], all_labels[1]))
    for tensor in all_labels[2:]:
        cat_all_labels = torch.cat((cat_all_labels, tensor))

    confmatrix = confusion_matrix(cat_all_predicted, cat_all_labels) 
    return confmatrix, cat_all_predicted, cat_all_labels