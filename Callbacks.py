import numpy as np
import os
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

"""Saves the model only if the metric has improved. It is possible to set to save only K best models."""
class ModelSaver:
    def __init__(self, models_root, base_name, K_best, metric_operation):
        """
        Args:
            root (str): path to model directory
            base_name (str): base name of model with format placeholder for epoch number
            K_best (int): keeps only the #K best models and removes other
            metric_operation (function): function to determine the improvement of watched metric
        """
        self.root = models_root
        self.base_name = base_name
        self.K_best = K_best
        self.models_array = [] # stores [score, epoch]
        self.reverse = metric_operation(0,1)

    def __call__(self, model, score, epoch, verbose=True):
        new_model_path = os.path.join(self.root, self.base_name.format(epoch, 0))
        if len(self.models_array) < self.K_best:
            self.models_array.append([score, epoch])
            torch.save(model.state_dict(), new_model_path)
            if verbose:
                print(f'Model saved: {new_model_path}')
            self.models_array.sort(reverse=self.reverse)
            return

        worst = self.models_array[-1]
        if self.improved(score, worst[0]):
            self.models_array[-1] = [score, epoch]
            os.remove(os.path.join(self.root, self.base_name.format(worst[1], 0)))
            torch.save(model.state_dict(), new_model_path)
            if verbose:
                print(f'Model saved: {new_model_path}')
            self.models_array.sort(reverse=self.reverse)
        return

