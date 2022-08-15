# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 13:19:02 2022

@author: g51388dw
"""
import torch
import torch.nn as nn
import RubinCNN
import rubin_cnn_model
from torch import optim
import numpy as np
import metrics

def train_cnn_ensemble(model1p, model2p, model3p, model4p, model5p, 
                    labels, recordings, segmentations, demographic_features, 
                    learning_rate=0.00001, num_epochs = 6, verbose = 4):
    
    if verbose >= 4:
        print('loading models')
    model1 = RubinCNN.CNN()
    model1.load_state_dict(model1p)
    model2 = RubinCNN.CNN()
    model2.load_state_dict(model2p)
    model3 = RubinCNN.CNN()
    model3.load_state_dict(model3p)
    model4 = RubinCNN.CNN()
    model4.load_state_dict(model4p)
    model5 = RubinCNN.CNN()
    model5.load_state_dict(model5p)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Freeze these models
    for param in model1.parameters():
        param.requires_grad_(False)

    for param in model2.parameters():
        param.requires_grad_(False)

    for param in model3.parameters():
        param.requires_grad_(False)

    for param in model4.parameters():
        param.requires_grad_(False)
    
    for param in model5.parameters():
        param.requires_grad_(False)

    if verbose >= 4:
        print('processing data again')
    # 1. process the data and get it into right shape for Rubin model
    binary_labels_inv = labels // 2
    binary_labels = ~binary_labels_inv.astype(bool)

    X_train, dem_train, y_train = rubin_cnn_model.format_data(recordings, segmentations, demographic_features, binary_labels)
    
    # 2. pytorchify it
    train_dataloader, test_dataloader = rubin_cnn_model.get_train_dataloaders(X_train,
                                                              dem_train,
                                                              y_train,
                                                              tts=0.80,
    
                                                              batch_size=250)
    if verbose >= 4:
        print('training new model')
    # 3. instantiate and train the model
    model = RubinCNN.CNNEnsemble(model1, model2, model3, model4, model5)

    # loss_func = nn.CrossEntropyLoss()
    # ratio is about 5:1 in favour of 'Absent', try setting pos_weight
    # weights = torch.as_tensor([5])
    # loss_func = nn.BCEWithLogitsLoss(pos_weight=weights) 
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)

    # Train the model
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        # this sets the model mode - (i.e. layers like dropout, batchnorm etc behave differently 
        # during training compared to testing)
        # note that this function was not defined explicitly in CNN, but because CNN is a type 
        # of nn.Module, it inherits some functions from the more general nn class.
        model.train()
        for i, (mfcc, dems, labels) in enumerate(train_dataloader):  # i batch index
            with(torch.set_grad_enabled(True)):
                # Note from the cnn class, output gives the model output (1 x 6 values),
                # and x gives the inputs into the last FC layer
                output = model(mfcc, dems)  # output, x
                pred_y = sigmoid(output)  # this is a probability

                if i == 0:
                    epoch_preds = pred_y
                    epoch_labels = labels
                else:
                    epoch_preds = torch.cat((epoch_preds, pred_y), 0)
                    epoch_labels = torch.cat((epoch_labels, labels), 0)
                
                loss = loss_func(output, labels.float())
                # clear gradients for this training step   
                optimizer.zero_grad()
                # backpropagation, compute gradients 
                loss.backward()
                # apply gradients             
                optimizer.step()

            yp = (pred_y.cpu().detach().numpy().astype(float) > 0.5).astype(int)
            epoch_acc = np.mean(yp == labels.cpu().detach().numpy().astype(int))

            if verbose >= 3:
                if (i + 1) % 15 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), epoch_acc))

        # eval of training phase
        # get outputs and labels from torch - these are binary still
        y_true = epoch_labels.cpu().detach().numpy().astype(int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(float)
        y_preds = (y_probs > 0.5).astype(int)
        y_adj = metrics.probs_to_labels(y_probs)
        train_acc = metrics.calc_accuracy(y_preds, y_true)
        adjusted_acc = metrics.calc_accuracy(y_adj, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} training accuracy: {train_acc:.4f}, weighted: {adjusted_acc:.4f}')

        # 4. guess we should validate it as well
        model.eval()
        for i, (mfcc, dems, labels) in enumerate(test_dataloader):
            with torch.set_grad_enabled(False):
                output = model(mfcc, dems)
                pred_y = sigmoid(output)  # this is a probability

                if i == 0:
                    epoch_preds = pred_y
                    epoch_labels = labels
                else:
                    epoch_preds = torch.cat((epoch_preds, pred_y), 0)
                    epoch_labels = torch.cat((epoch_labels, labels), 0)

        # eval of testing phase
        y_true = epoch_labels.cpu().detach().numpy().astype(int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(float)
        y_preds = (y_probs > 0.5).astype(int)
        y_adj = metrics.probs_to_labels(y_probs)
        train_acc = metrics.calc_accuracy(y_preds, y_true)
        adjusted_acc = metrics.calc_accuracy(y_adj, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} testing accuracy: {train_acc:.4f}, weighted: {adjusted_acc:.4f}')

    return model.state_dict()

def train_outcome_ensemble(model1p, model2p, model3p, model4p, model5p, 
                    labels, recordings, segmentations, demographic_features, 
                    learning_rate=0.00001, num_epochs = 6, verbose = 4):
    
    if verbose >= 4:
        print('loading models')
    model1 = RubinCNN.CNN()
    model1.load_state_dict(model1p)
    model2 = RubinCNN.CNN()
    model2.load_state_dict(model2p)
    model3 = RubinCNN.CNN()
    model3.load_state_dict(model3p)
    model4 = RubinCNN.CNN()
    model4.load_state_dict(model4p)
    model5 = RubinCNN.CNN()
    model5.load_state_dict(model5p)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Freeze these models
    for param in model1.parameters():
        param.requires_grad_(False)

    for param in model2.parameters():
        param.requires_grad_(False)

    for param in model3.parameters():
        param.requires_grad_(False)

    for param in model4.parameters():
        param.requires_grad_(False)
    
    for param in model5.parameters():
        param.requires_grad_(False)

    
    if verbose >= 4:
        print('processing data again')
    # 1. process the data and get it into right shape for Rubin model
    binary_labels = labels

    X_train, dem_train, y_train = rubin_cnn_model.format_data(recordings, segmentations, demographic_features, binary_labels)
    
    # 2. pytorchify it
    train_dataloader, test_dataloader = rubin_cnn_model.get_train_dataloaders(X_train,
                                                              dem_train,
                                                              y_train,
                                                              tts=0.80,
                                                              batch_size=250)
    if verbose >= 4:
        print('training new model')
    # 3. instantiate and train the model
    model = RubinCNN.CNNEnsemble(model1, model2, model3, model4, model5)

    # loss_func = nn.CrossEntropyLoss()
    # ratio is about 5:1 in favour of 'Absent', try setting pos_weight
    # weights = torch.as_tensor([5])
    # loss_func = nn.BCEWithLogitsLoss(pos_weight=weights) 
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)

    # Train the model
    total_step = len(train_dataloader)
    for epoch in range(num_epochs):
        # this sets the model mode - (i.e. layers like dropout, batchnorm etc behave differently 
        # during training compared to testing)
        # note that this function was not defined explicitly in CNN, but because CNN is a type 
        # of nn.Module, it inherits some functions from the more general nn class.
        model.train()
        for i, (mfcc, dems, labels) in enumerate(train_dataloader):  # i batch index
            with(torch.set_grad_enabled(True)):
                # Note from the cnn class, output gives the model output (1 x 6 values),
                # and x gives the inputs into the last FC layer
                output = model(mfcc, dems)  # output, x
                pred_y = sigmoid(output)  # this is a probability

                if i == 0:
                    epoch_preds = pred_y
                    epoch_labels = labels
                else:
                    epoch_preds = torch.cat((epoch_preds, pred_y), 0)
                    epoch_labels = torch.cat((epoch_labels, labels), 0)
                
                loss = loss_func(output, labels.float())
                # clear gradients for this training step   
                optimizer.zero_grad()
                # backpropagation, compute gradients 
                loss.backward()
                # apply gradients             
                optimizer.step()

            yp = (pred_y.cpu().detach().numpy().astype(float) > 0.5).astype(int)
            epoch_acc = np.mean(yp == labels.cpu().detach().numpy().astype(int))

            if verbose >= 3:
                if (i + 1) % 15 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), epoch_acc))

        # eval of training phase
        # get outputs and labels from torch - these are binary still
        y_true = epoch_labels.cpu().detach().numpy().astype(int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(float)
        y_preds = (y_probs > 0.5).astype(int)
        y_adj = metrics.probs_to_labels(y_probs)
        train_acc = metrics.calc_accuracy(y_preds, y_true)
        adjusted_acc = metrics.calc_accuracy(y_adj, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} training accuracy: {train_acc:.4f}, weighted: {adjusted_acc:.4f}')

        # 4. guess we should validate it as well
        model.eval()
        for i, (mfcc, dems, labels) in enumerate(test_dataloader):
            with torch.set_grad_enabled(False):
                output = model(mfcc, dems)
                pred_y = sigmoid(output)  # this is a probability

                if i == 0:
                    epoch_preds = pred_y
                    epoch_labels = labels
                else:
                    epoch_preds = torch.cat((epoch_preds, pred_y), 0)
                    epoch_labels = torch.cat((epoch_labels, labels), 0)

        # eval of testing phase
        y_true = epoch_labels.cpu().detach().numpy().astype(int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(float)
        y_preds = (y_probs > 0.5).astype(int)
        y_adj = metrics.probs_to_labels(y_probs)
        train_acc = metrics.calc_accuracy(y_preds, y_true)
        adjusted_acc = metrics.calc_accuracy(y_adj, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} testing accuracy: {train_acc:.4f}, weighted: {adjusted_acc:.4f}')

    return model.state_dict()


def run_nn_rubin_ensemble(ensemble_model, model1p, model2p, model3p, model4p, model5p, recordings, segmentations, demographic_features, verbose=1, combine=1, MAGIC_THRESHOLD=0.33):
    """
    Parameters
        recordings (list): flattened list of all recordings
        segmentations (list): flattened list of all segmentations
        combine (bool): if 1, take average probability over all cepstrum features.
            Else, classify as murmur if some percentage of epochs is murmur.

    Returns
        y_preds (arr): binary predicted labels on a per-recording basis
        y_probs (arr): probabilities associated with 'Present', 'Absent'
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = RubinCNN.CNN()
    model1.load_state_dict(model1p)
    model2 = RubinCNN.CNN()
    model2.load_state_dict(model2p)
    model3 = RubinCNN.CNN()
    model3.load_state_dict(model3p)
    model4 = RubinCNN.CNN()
    model4.load_state_dict(model4p)
    model5 = RubinCNN.CNN()
    model5.load_state_dict(model5p)
    model = RubinCNN.CNNEnsemble(model1, model2, model3, model4, model5)
    model.load_state_dict(ensemble_model)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)
    model.eval()

    n_recordings = len(recordings)
    # print(n_recordings)
    y_preds = np.zeros((n_recordings, 3))  # Present, Unknown, Absent
    y_probs = np.zeros((n_recordings, 3))
    # pat_output = [] # SES: can remove - used to check I did outputs correctly

    for i in range(n_recordings):
        # need to wrap these in lists for method consistency
        X_train, dems, _ = rubin_cnn_model.format_data([recordings[i]], [segmentations[i]], demographic_features)

        test_dataloader = rubin_cnn_model.get_test_dataloaders(X_train,
                                               dems,
                                               batch_size=1)

        for j, (mfcc, dems) in enumerate(test_dataloader):
            with torch.set_grad_enabled(False):
                output = model(mfcc, dems)[0]
                pred_y = sigmoid(output)

                if j == 0:
                    epoch_outputs = pred_y
                else:
                    epoch_outputs = torch.cat((epoch_outputs, pred_y), 0)

        probs = epoch_outputs.cpu().detach().numpy().astype(np.float)

        # classic rubin approach - take average probability
        if combine:
            rec_output = np.mean(probs)
            y_probs[i] = np.array([rec_output, 0., 1 - rec_output])
            one_hot_idx = np.argmax(y_probs[i]).astype(int)
            y_preds[i, one_hot_idx] = 1.
            # pat_output stays as binary for validation
            # pat_output.append( (rec_output > 0.5).astype(int) )

        # alternative - classify as murmur if some percentage of epochs is murmur
        # SES: this can be done outside this method with pct_murmur, to test
        else:
            rec_output = np.round(probs)
            total_epochs = len(rec_output)
            pct_murmur = np.sum(rec_output) / total_epochs

            if pct_murmur > MAGIC_THRESHOLD:
                # pat_output.append(1)
                y_preds[i, 0] = 1.
            else:
                # pat_output.append(0)
                y_preds[i, 2] = 1.

            y_probs[i] = np.array([pct_murmur, 0., 1 - pct_murmur])

    return y_preds, y_probs