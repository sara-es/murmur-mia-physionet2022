import numpy as np
import ResNet
import torch
from torch import optim
from torch import nn
import os
import metrics
from evaluate_model import compute_weighted_accuracy
import PhonoDataset


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_train_dataloaders(recordings, features, labels, tts=0.80, batch_size=10):
    """
    Parameters
        recordings (arr)
        features (arr)
        labels (arr)
        tts (float): percentage of data to use for training
        batch_size (int)        
    """    

    device = get_device()
    tts_idx = int(recordings.shape[0]*tts)

    train_dataset = PhonoDataset.TrainDataset(recordings[:tts_idx], 
                                              features[:tts_idx],
                                              labels[:tts_idx] )
    test_dataset = PhonoDataset.TrainDataset(recordings[tts_idx:], 
                                             features[tts_idx:],
                                             labels[tts_idx:] )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=(True if device == 'cuda' else False)) 
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=(True if device == 'cuda' else False))

    return train_dataloader, test_dataloader


def get_test_dataloader(recordings, features, batch_size=1):
    """
    Parameters
        recordings (arr)
        features (arr)
        batch_size (int)
    """    
    device = get_device()

    test_dataset = PhonoDataset.TestDataset(recordings, features)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=(True if device == 'cuda' else False))

    return test_dataloader


def train_resnet_model(model_folder, recordings, data_features, class_labels, verbose=1, max_epochs=20):
    """
    Used for training model, when loading in all recordings.
    If 1-channel (binary classification), probability is that of 'Present'.
    
    Parameters
        model_folder (str): path to folder in which the model should be saved
        train_dataloader (torch DataLoader): a dataloader for TrainDataset with train data
        test_dataloader (torch DataLoader): a dataloader for TrainDataset with test data
        class_labels (arr): target labels. Needed to use challenge score as evaluation metric.
        verbose (int): how many print statements do you want? 0==none, 4==TMI
    
    Returns
        model (torch.nn.Module): saved state dict of the trained model

    Other notables
        y_pred (arr): predicted bool label, 0 if 'Absent', of len(n_recordings)
        probs (arr): 32-bit floats of probabilities for class 'Present'
    
    """    
    # labels is array of all labels, with 'Murmur', 'Unknown', 'Absent' as {0,1,2}
    # cast Murmur (0) and Unknown (1) to 1 and Absent (2) to 0
    binary_labels_inv = class_labels//2
    binary_labels = ~binary_labels_inv.astype(bool)
    
    # Dataloaders for torch.
    train_dataloader, test_dataloader = get_train_dataloaders(recordings, data_features, binary_labels, tts=0.80)
    device = get_device()
    classes = ['Present', 'Unknown', 'Absent'] # for compute_challenge_score
    
    model = ResNet.ResNet(ResNet.BasicBlock, [2, 2, 2, 2], in_channel=1, out_channel=1)
    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=0.003, weight_decay=1e-5)
    pos_weight = torch.tensor([[3]], device=device) 
    # pos_weight adds a weight multiplier for positive classes, may help in imbalanced dataset
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # binary CX loss, can use BCELoss() if sigmoid in forward pass. 
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)
    model.to(device)

    best_accuracy = -1 # changed from 0 to force at least one model save
    best_challenge_score = 3000
    epoch_loss = 0
    for i in range(max_epochs):

        # training phase
        model.train()
        for batch_i, (recordings, features, label) in enumerate(train_dataloader):
            inputs = recordings.to(device)
            fts = features.to(device)
            labels = label.to(device)
            # print(f"inputs: {inputs}, fts: {fts}, labels: {labels}")

            with(torch.set_grad_enabled(True)):
                pred_labels = model(inputs, fts)
                sig_labels = sigmoid(pred_labels)
                if batch_i == 0:
                    labels_all = labels
                    sig_labels_all = sig_labels
                else:
                    labels_all = torch.cat((labels_all, labels), 0)
                    sig_labels_all = torch.cat((sig_labels_all, sig_labels), 0)
                loss = criterion(pred_labels, labels)
                loss_temp = loss.item() * inputs.size(0)
                epoch_loss += loss_temp

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
        
        if verbose >=2:
            print('epoch {0}'.format(i))
        
        y_true = labels_all.cpu().detach().numpy().astype(np.int)
        y_pr = sig_labels_all.cpu().detach().numpy().astype(np.float)
        y_pred = metrics.probs_to_labels(y_pr)

        train_acc = metrics.calc_accuracy(y_pred, y_true)
        sensitivity = metrics.calc_sensitivity(y_pred, y_true)
        
        if verbose >=2:
            print(f'training accuracy:{train_acc:.4f}, sensitivity:{sensitivity:.4f}')

        # validation phase
        model.eval()
        for batch_i, (recordings, features, label) in enumerate(test_dataloader):
            inputs = recordings.to(device)
            fts = features.to(device)
            labels = label.to(device)

            with torch.set_grad_enabled(False):
                pred_labels = model(inputs, fts)
                sig_labels = sigmoid(pred_labels)
                if batch_i == 0:
                    labels_all = labels
                    sig_labels_all = sig_labels
                else:
                    labels_all = torch.cat((labels_all, labels), 0)
                    sig_labels_all = torch.cat((sig_labels_all, sig_labels), 0)

        y_true = labels_all.cpu().detach().numpy().astype(np.int)
        y_pr = sig_labels_all.cpu().detach().numpy().astype(np.float)
        y_pred = metrics.probs_to_labels(y_pr)

        val_acc = metrics.calc_accuracy(y_pred, y_true)
        sensitivity = metrics.calc_sensitivity(y_pred, y_true)

        # horrible hacky workaround to use challenge metric as eval
        # assumes 20% holdout in train test split
        y_pred_one_hot = metrics.probs_to_one_hot(y_pr)
        tts_idx = int(class_labels.shape[0]*0.80)
        class_labels_one_hot = metrics.class_labels_to_one_hot(class_labels[tts_idx:])
        if class_labels_one_hot.shape == y_pred_one_hot.shape:
            challenge_score = compute_weighted_accuracy(class_labels_one_hot, y_pred_one_hot, classes)
        else: 
            print("Could not compute challenge score, check labels and TTS.")
            print(f"Class labels have shape {class_labels.shape}, output labels \
                have shape {y_pred_one_hot.shape}")
            challenge_score = 3000

        if verbose >=2:
            print(f'val accuracy:{val_acc:.4f}, sensitivity:{sensitivity:.4f}, challenge score: {challenge_score}')

        # save model if it's decent
        if challenge_score < best_challenge_score:
            best_challenge_score = challenge_score
        if val_acc > best_accuracy:
            best_accuracy = val_acc
        # Create a folder for the model if it does not already exist.
        if verbose >=2:
            print('saving resnet model...')
        os.makedirs(model_folder, exist_ok=True)
        torch.save(model.state_dict(), 
                   os.path.join(model_folder, f'run_{i}_{challenge_score:.0f}cs{100*val_acc:.0f}acc.pth'))            
        
        if challenge_score < 200: # TODO
            return model.state_dict()
        elif i == max_epochs-1:
            return model.state_dict()
    

def run_resnet_model(saved_model, recordings, data_features, verbose=1):
    """
    Used for testing/running model, when loading in one patient's recordings.
    For 1-channel (binary classification), probability is that of 'Present'.
    
    Parameters
        model (torch model): state dictionary of the saved model
        test_dataloader (torch DataLoader): a dataloader for TestDataset with test data
        verbose (int): how many print statements do you want? 0==none, 4==TMI
    Returns
        y_pred (arr): ints, binary 
        probs (arr): floats, probability of 'Present' for each epoch
    """
    test_dataloader = get_test_dataloader(recordings, data_features)
    device = get_device() 
    
    model = ResNet.ResNet(ResNet.BasicBlock, [2, 2, 2, 2], in_channel=1, out_channel=1)
    model.load_state_dict(saved_model)
    sigmoid = nn.Sigmoid()
    sigmoid.to(device)
    model.to(device)

    model.eval()
    for batch_i, (recordings, features) in enumerate(test_dataloader):
        inputs = recordings.to(device)
        fts = features.to(device)

        with torch.set_grad_enabled(False):
            pred_labels = model(inputs, fts)
            sig_labels = sigmoid(pred_labels)

        if batch_i == 0:
            sig_labels_all = sig_labels
        else:
            sig_labels_all = torch.cat((sig_labels_all, sig_labels), 0)
    
    # get probabilities, find index of highest prob for each recording
    probs = sig_labels_all.cpu().detach().numpy() 

    # binary labels
    y_pred = metrics.probs_to_labels(probs)
    
    if verbose >= 4:
        print(f'predicted label: {y_pred}\n and positive probability: {probs}')
    
    return probs



    