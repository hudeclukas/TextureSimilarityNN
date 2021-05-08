import numpy as np
import torch


"""Early stops the training if validation loss doesn't improve after a given patience."""
class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                           Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement.
                           Default: False
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.val_loss_min = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss is None:
            self.val_loss_min = val_loss
        elif val_loss > self.val_loss_min + self.delta:
            self.counter+=1
            if self.verbose:
                print(f'EarlyStopping time to die: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
        elif val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.counter = 0
        else:
            self.counter = 0

        return self.early_stop