from configuration import configuration
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from dataset_io import SimilarityDataset
from Callbacks import EarlyStopping, ModelSaver

def trainer_standard(model, model_name, distance_metric, data_root, models_root, device, last_epoch=0):
    config = configuration()

    train_data = SimilarityDataset(os.path.join(data_root,'train'), max_samples=config.max_samples, max_images=config.max_images)
    val_data = SimilarityDataset(os.path.join(data_root,'val'), max_samples=100, max_images=16)

    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_data, config.batch_size, shuffle=True, num_workers=8)
    # train_img1, train_img2, train_label = next(iter(train_dataloader))

    wandb.watch(model, log_freq=50, log="all")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.7, 0.99))
    early_stopper = EarlyStopping(patience=config.patience, delta=0.05)
    improved = lambda new, old: new < old
    model_saver = ModelSaver(models_root, model_name, 5, improved)

    last_best_loss = np.inf
    last_best_score = 0
# Run epochs
    for epoch in tqdm(range(last_epoch, config.epochs), desc='Epoch'):
        train_loss_mean = 0
        batches = 0
# Run Training
        model.train()
        for batch, (x1,x2,y) in tqdm(enumerate(train_dataloader), desc='Train Batch', total=int(np.ceil(len(train_data)/config.batch_size))):
            if len(x1) != config.batch_size:
                print(f" ... Skipping small batch <{len(x1)}>")
                continue
            out1,out2 = model.forward(x1.to(device),x2.to(device))
            loss = model.loss_contrastive(net1=out1,net2=out2,target=y.to(device),margin=config.margin,distance_metric='eucl')
            train_loss_mean += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({"train_batch_loss": loss})
            # print(f' ... Loss {loss.item()}', end='...')
            batches += 1

        wandb.log({"train_loss": train_loss_mean/batches})
        print(f"Epoch: {epoch} Loss: {train_loss_mean/batches}")
# Reset Training data
        train_data.reset(other_images=config.other_images)

        loss_mean = 0
        batches = 0
# Run Validation
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        if epoch % 1 == 0:
            model.eval()
            for batch, (x1,x2,y) in tqdm(enumerate(val_dataloader), desc='Val Batch',total=int(np.ceil(len(val_data)/config.batch_size))):
                if len(x1) != config.batch_size:
                    print(f" ... Skipping small batch <{len(x1)}>")
                    continue
                out1, out2 = model.forward(x1.to(device), x2.to(device))
                val_loss = model.loss_contrastive(net1=out1,net2=out2,target=y.to(device),margin=config.margin,distance_metric=distance_metric)
                distance, _ = distance_metric(out1, out2)
                # similarity = (distance < 1).float()
                # correct = (similarity==y.to(device)).float().sum()
                for dist,t in zip(distance,y):
                    if dist < 1 and t == 1: tp += 1
                    if dist < 1 and t == 0: fp += 1
                    if dist >= 1 and t == 0 : tn += 1
                    if dist >= 1 and t == 1: fn += 1
                # print(f" ... Loss: {val_loss.item()} Acc: {correct/x1.shape[0]}", end='...')
                loss_mean += val_loss.item()
                # accuracy_mean += correct/x1.shape[0]
                batches += 1

            wandb.log({"val_loss": loss_mean/batches, "val_acc":(tp+tn)/(tp+tn+fp+fn), "val_prec":(tp)/(tp+fp), "val_rec":(tp)/(tp+fn)})
            print(f"Val Loss: {loss_mean/batches} Acc: {(tp+tn)/(tp+tn+fp+fn)}")
# Reset validation data
            val_data.reset(other_images=True)

# Save model if loss is better
        model_saver(model, train_loss_mean, epoch)

    if early_stopper(loss_mean):
        print('Early stopping')
        torch.save(model.state_dict(), os.path.join(models_root, model_name.format(epoch, 0)))
        return

    torch.save(model.state_dict(), os.path.join(models_root, model_name.format(config.epochs, 0)))
