import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataset_io import SimilarityDataset
from Callbacks import EarlyStopping

def trainer_euclidean_standard(model, model_name, data_root, models_root, device):
    config = wandb.config

    train_data = SimilarityDataset(os.path.join(data_root,'train'), max_samples=50)
    val_data = SimilarityDataset(os.path.join(data_root,'val'), max_samples=50)

    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_data, config.batch_size, shuffle=True, num_workers=8)
    # train_img1, train_img2, train_label = next(iter(train_dataloader))

    wandb.watch(model, log_freq=50, log="all")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.7, 0.99))
    early_stopper = EarlyStopping(patience=7, delta=0.2)

    last_best_loss = np.inf
# Run epochs
    for epoch in tqdm(range(config.epochs),desc='Epoch'):
        loss_mean = 0
        batches = 0
# Run Training
        for batch, (x1,x2,y) in tqdm(enumerate(train_dataloader), desc='Train Batch'):
            if len(x1) != config.batch_size:
                print(f" ... Skipping small batch <{len(x1)}>")
                continue
            out1,out2 = model.forward(x1.to(device),x2.to(device))
            loss = model.loss_contrastive(net1=out1,net2=out2,target=y.to(device),margin=2.5,distance_metric='eucl')
            loss_mean += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f' ... Loss {loss.item()}', end='...')
            batches += 1

        wandb.log({"train_loss": loss_mean/batches})
        print(f"Epoch: {epoch} Loss: {loss_mean/batches}")

        train_data.reset()

        loss_mean = 0
        accuracy_mean = 0
        batches = 0
# Run Validation
        for batch, (x1,x2,y) in tqdm(enumerate(val_dataloader), desc='Val Batch'):
            if len(x1) != config.batch_size:
                print(f" ... Skipping small batch <{len(x1)}>")
                continue
            out1, out2 = model.forward(x1.to(device), x2.to(device))
            val_loss = model.loss_contrastive(net1=out1,net2=out2,target=y.to(device),margin=2.5,distance_metric='eucl')
            distance, _ = model.distance_euclid(out1, out2)
            similarity = (distance > 1).float()
            correct = (similarity==y.to(device)).float().sum()

            print(f" ... Loss: {val_loss.item()} Acc: {correct/x1.shape[0]}", end='...')
            loss_mean += val_loss.item()
            accuracy_mean += correct/x1.shape[0]
            batches += 1

        wandb.log({"val_loss": loss_mean/batches, "val_acc":accuracy_mean/batches})
        val_data.reset()

        if loss_mean < last_best_loss:
            torch.save(model.state_dict(), os.path.join(models_root,model_name.format(epoch, 0)))
            last_best_loss = loss_mean

        if early_stopper(loss_mean):
            print('Early stopping')
            break