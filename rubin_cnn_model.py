# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:32:06 2022

@author: g51388dw
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch import optim
import MFCCdataset, RubinCNN
import metrics
# import rubin_proc
from rubin_proc import mfcc


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_frame_starts(segmentation, window_step):
    """ get the start of each cardiac cycle - used to create epochs that start at the same 
        point in the cycle.
        We assume that index 1 in the segmentation files relates to the start of S1, so use 
        that. However, it doesn't really matter which one we use, as long as we are consistent

        NB this expects segmentations to be in seconds (I think)
    """
    stage = 1
    stage_segs = np.where(segmentation[2, :] == stage)
    frame_starts = segmentation[0, stage_segs].reshape(-1)  # only remove outer dimension
    # segmentations are returned in indices, must divide by sample frequency (bit hacky)
    frame_starts = np.round(frame_starts / (4000 * window_step)).astype(int)
    return frame_starts


def format_data(recordings, segmentations, demographic_features, binary_labels=None):
    """ 
    Format data so that it is in the 6 x 300 format ready for the neural network.
    Note mfcc_feat is a numpy array of size (NUMFRAMES by numcep) containing features. 
        Each row holds 1 feature vector (mel filter features, whatever those are).
        It seems like we're taking the mfccs of the first seconds following the start of a cycle

    Parameters

    Returns
        X_train (list): 6x300 arrays of ceptral features (len 300 features, in batch size 6)
        dem_train (list): demographic information for recordings
        y_train (list): binary target labels 
    """
    samplerate = 4000
    window_step = 0.01  # DW: not quite sure what this is yet
    num_features = 6
    duration = 3
    frame_length = int(duration / window_step)
    X_train = []
    y_train = []
    dem_train = []
    
    # rough normalisation of demographics - not quite right, but close enough
    max_demo_fts = np.array([180., 1., 200., 180., 111., 1.])
    demographic_features = (demographic_features / max_demo_fts).astype(np.float)

    for idx, recording in enumerate(recordings):
        # derive mfcc features for entire recording
        mfcc_feat = mfcc(recording, samplerate, winstep=window_step, numcep=num_features)

        if segmentations[idx].shape[1] >= 4:
            frame_starts = get_frame_starts(segmentations[idx], window_step)

            empty_slice = True  # hacky way to make sure we end up with some data for each recording
            for curr_frame in frame_starts:
                # if count < MAX_SEGMENTS: # put this back in if we are at danger of running over time
                mfcc_slice = mfcc_feat[curr_frame:(curr_frame + frame_length), :]

                # Only include full duration slices
                if mfcc_slice.shape[0] == frame_length:
                    mu = np.mean(mfcc_slice)
                    sig = np.std(mfcc_slice)
                    mfcc_slice = (mfcc_slice - mu) / sig

                    X_train.append(mfcc_slice)
                    dem_train.append(demographic_features[idx, :])
                    if binary_labels is not None:  # check to see we are actually given labels
                        y_train.append(binary_labels[idx])
                    empty_slice = False

            if empty_slice:  # make sure we have at least one entry for this recording
                mfcc_slice = mfcc_feat[frame_starts[0]:(frame_starts[0] + frame_length), :]
                padding = int(frame_length - mfcc_slice.shape[0])
                if padding > 0:
                    mu = np.mean(mfcc_slice)
                    sig = np.std(mfcc_slice)
                    mfcc_slice = (mfcc_slice - mu) / sig
                    mfcc_slice = np.pad(mfcc_slice, ((0, padding), (0, 0)), 'constant')

                    X_train.append(mfcc_slice)
                    dem_train.append(demographic_features[idx, :])
                    if binary_labels is not None:  # check to see we are actually given labels
                        y_train.append(binary_labels[idx])
                
        else:
            print("missing segmentation file at {}".format(idx)) 
    
    return X_train, dem_train, y_train


def get_train_dataloaders(mfcc, demographics, binary_labels, tts=0.80, batch_size=1):
    """
    Parameters
        mfcc (list): X_train, list of 6x300 arrays 
        binary_labels (list): y_train, list of binary labels 
        tts (float): percentage of data to use for training, between 0 and 1
        batch_size (int)
    """
    device = get_device()
    tts_idx = int(len(mfcc) * tts)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = MFCCdataset.TrainDataset1ch(mfcc[:tts_idx],
                                                demographics[:tts_idx],
                                                binary_labels[:tts_idx],
                                                transform)
    test_dataset = MFCCdataset.TrainDataset1ch(mfcc[tts_idx:],
                                               demographics[tts_idx:],
                                               binary_labels[tts_idx:],
                                               transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=(True if device == 'cuda' else False),
                                  # num_workers=1),
                                  )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=(True if device == 'cuda' else False),
                                 # num_workers=1)
                                 )

    return train_dataloader, test_dataloader


def get_test_dataloaders(mfcc, demographics, batch_size=1):
    """
    Parameters
        mfcc (list): X_test, list of 6x300 arrays 
        binary_labels (list): y_test, list of binary labels 
        batch_size (int)
    """
    device = get_device()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    test_dataset = MFCCdataset.TestDataset1ch(mfcc, demographics, transform)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=(True if device == 'cuda' else False),
                                 # num_workers=1)
                                 )

    return test_dataloader


def train_rubin_cnn(recordings, segmentations, demographic_features, labels, verbose=1, num_epochs=1, learning_rate=0.0001):

    """
    Parameters
        recordings (list): flattened list of all recordings
        labels (arr): all murmur target labels, with 'Murmur', 'Unknown', 
            'Absent' as {0,1,2}
        segmentations (list): flattened list of all segmentations

    Returns
        model (torch.nn.Module): saved state dict of the trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. process the data and get it into right shape for Rubin model
    binary_labels_inv = labels // 2
    binary_labels = ~binary_labels_inv.astype(bool)

    X_train, dem_train, y_train = format_data(recordings, segmentations, demographic_features, binary_labels)
    # print(type(dem_train[0][0]))
    
    # 2. pytorchify it
    train_dataloader, test_dataloader = get_train_dataloaders(X_train,
                                                              dem_train,
                                                              y_train,
                                                              tts=0.80,
                                                              batch_size=250)

    # 3. instantiate and train the model
    cnn = RubinCNN.CNN()

    # loss_func = nn.CrossEntropyLoss()
    # ratio is about 5:1 in favour of 'Absent', try setting pos_weight
    # weights = torch.as_tensor([5])
    # loss_func = nn.BCEWithLogitsLoss(pos_weight=weights) 
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)

    # Train the model
    total_step = len(train_dataloader)

    for epoch in range(num_epochs):
        # this sets the model mode - (i.e. layers like dropout, batchnorm etc behave differently 
        # during training compared to testing)
        # note that this function was not defined explicitly in CNN, but because CNN is a type 
        # of nn.Module, it inherits some functions from the more general nn class.
        cnn.train()

        for i, (mfcc, dems, labels) in enumerate(train_dataloader):  # i batch index
            with(torch.set_grad_enabled(True)):
                # Note from the cnn class, output gives the model output (1 x 6 values),
                # and x gives the inputs into the last FC layer
                output = cnn(mfcc, dems)[0]  # output, x
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

            yp = (pred_y.cpu().detach().numpy().astype(np.float) > 0.5).astype(int)
            epoch_acc = np.mean(yp == labels.cpu().detach().numpy().astype(np.int))

            if verbose >= 3:
                if (i + 1) % 60 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), epoch_acc))

        # eval of training phase
        # get outputs and labels from torch - these are binary still
        y_true = epoch_labels.cpu().detach().numpy().astype(np.int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(np.float)
        y_preds = (y_probs > 0.5).astype(int)
        train_acc = metrics.calc_accuracy(y_preds, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} training accuracy: {train_acc:.4f} (per-mfcc sample)')

        # 4. guess we should validate it as well
        cnn.eval()
        for i, (mfcc, dems, labels) in enumerate(test_dataloader):
            with torch.set_grad_enabled(False):
                output = cnn(mfcc, dems)[0]
                pred_y = sigmoid(output)  # this is a probability

                if i == 0:
                    epoch_preds = pred_y
                    epoch_labels = labels
                else:
                    epoch_preds = torch.cat((epoch_preds, pred_y), 0)
                    epoch_labels = torch.cat((epoch_labels, labels), 0)

        # eval of testing phase
        y_true = epoch_labels.cpu().detach().numpy().astype(np.int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(np.float)
        y_preds = (y_probs > 0.5).astype(int)
        train_acc = metrics.calc_accuracy(y_preds, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} testing accuracy: {train_acc:.4f}')

    return cnn.state_dict()


def run_rubin_cnn(saved_model, recordings, segmentations, demographic_features, verbose=1, combine=1, MAGIC_THRESHOLD=0.33):
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

    cnn = RubinCNN.CNN()
    cnn.load_state_dict(saved_model)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)
    cnn.eval()

    n_recordings = len(recordings)
    # print(n_recordings)
    y_preds = np.zeros((n_recordings, 3))  # Present, Unknown, Absent
    y_probs = np.zeros((n_recordings, 3))
    # pat_output = [] # SES: can remove - used to check I did outputs correctly

    for i in range(n_recordings):
        # need to wrap these in lists for method consistency
        X_train, dems, _ = format_data([recordings[i]], [segmentations[i]], demographic_features)

        test_dataloader = get_test_dataloaders(X_train,
                                               dems,
                                               batch_size=1)

        for j, (mfcc, dems) in enumerate(test_dataloader):
            with torch.set_grad_enabled(False):
                output = cnn(mfcc, dems)[0]
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

def train_outcome_cnn(recordings, segmentations, demographic_features, labels, verbose=1, num_epochs=1, learning_rate=0.0001):

    """
    Parameters
        recordings (list): flattened list of all recordings
        labels (arr): all murmur target labels, with 'Murmur', 'Unknown', 
            'Absent' as {0,1,2}
        segmentations (list): flattened list of all segmentations

    Returns
        model (torch.nn.Module): saved state dict of the trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. process the data and get it into right shape for Rubin model
    binary_labels = labels.astype(bool)
    X_train, dem_train, y_train = format_data(recordings, segmentations, demographic_features, binary_labels)
    
    # 2. pytorchify it
    train_dataloader, test_dataloader = get_train_dataloaders(X_train,
                                                              dem_train,
                                                              y_train,
                                                              tts=0.80,
                                                              batch_size=250)

    # 3. instantiate and train the model
    cnn = RubinCNN.CNN()
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)

    # Train the model
    total_step = len(train_dataloader)

    for epoch in range(num_epochs):
        # this sets the model mode - (i.e. layers like dropout, batchnorm etc behave differently 
        # during training compared to testing)
        # note that this function was not defined explicitly in CNN, but because CNN is a type 
        # of nn.Module, it inherits some functions from the more general nn class.
        cnn.train()

        for i, (mfcc, dems, labels) in enumerate(train_dataloader):  # i batch index
            with(torch.set_grad_enabled(True)):
                # Note from the cnn class, output gives the model output (1 x 6 values),
                # and x gives the inputs into the last FC layer
                output = cnn(mfcc, dems)[0]  # output, x
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

            yp = (pred_y.cpu().detach().numpy().astype(np.float) > 0.5).astype(int)
            epoch_acc = np.mean(yp == labels.cpu().detach().numpy().astype(np.int))

            if verbose >= 3:
                if (i + 1) % 60 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), epoch_acc))

        # eval of training phase
        # get outputs and labels from torch - these are binary still
        y_true = epoch_labels.cpu().detach().numpy().astype(np.int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(np.float)
        y_preds = (y_probs > 0.5).astype(int)
        train_acc = metrics.calc_accuracy(y_preds, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} training accuracy: {train_acc:.4f} (per-mfcc sample)')

        # 4. guess we should validate it as well
        cnn.eval()
        for i, (mfcc, dems, labels) in enumerate(test_dataloader):
            with torch.set_grad_enabled(False):
                output = cnn(mfcc, dems)[0]
                pred_y = sigmoid(output)  # this is a probability

                if i == 0:
                    epoch_preds = pred_y
                    epoch_labels = labels
                else:
                    epoch_preds = torch.cat((epoch_preds, pred_y), 0)
                    epoch_labels = torch.cat((epoch_labels, labels), 0)

        # eval of testing phase
        y_true = epoch_labels.cpu().detach().numpy().astype(np.int)
        y_probs = epoch_preds.cpu().detach().numpy().astype(np.float)
        y_preds = (y_probs > 0.5).astype(int)
        train_acc = metrics.calc_accuracy(y_preds, y_true)

        if verbose >= 2:
            print(f'Epoch {epoch + 1} testing accuracy: {train_acc:.4f}')

    return cnn.state_dict()


def simple_rubin_ensemble(model0, model1, model2, model3, model4, recordings, segmentations, demographic_features, MAGIC_THRESHOLD = 0.4):
    # rubin_cnn outputs on a per-recording basis
    n_recordings = len(recordings)
    y_preds = np.zeros((5,n_recordings,3)) 
    y_probs = np.zeros((5,n_recordings,3))
    models = [model0, model1, model2, model3, model4]
    for i in range(5):
        y_preds[i,:,:], y_probs[i,:,:] = run_rubin_cnn(models[i], recordings, segmentations, demographic_features, verbose=1, combine=1, MAGIC_THRESHOLD=0.33)
        
    per_rec_murmur_probs = np.mean(y_probs, axis = 0) # average probabilities for single recordings, across models
    # murmur_prob = metrics.get_ngm(per_rec_murmur_prob) # get normalised geometric mean of predictions across recordings
    
    rec_output = np.round(per_rec_murmur_probs)
    total_epochs = len(rec_output)
    pct_murmur = np.sum(rec_output) / total_epochs

    y_preds = np.zeros((n_recordings, 3))
    for j in range(n_recordings):
        if pct_murmur > MAGIC_THRESHOLD:
            y_preds[j, 0] = 1.
        else:
            y_preds[j, 2] = 1.
    
    return y_preds, per_rec_murmur_probs