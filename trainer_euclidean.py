import os
import torch
from torch.utils.data import DataLoader
import wandb
from dataset_io import SimilarityDataset


def trainer_euclidean_standard(model, model_name, data_root, models_root, device):
    config = wandb.config

    train_data = SimilarityDataset(os.path.join(data_root,'train'), max_samples=50)
    # test_data = SimilarityDataset(os.path.join(data_root,'test'), max_samples=50, max_images=2)

    train_dataloader = DataLoader(train_data, config.batch_size, shuffle=True, num_workers=8)
    # test_dataloader = DataLoader(test_data, config.batch_size, shuffle=True, num_workers=4)
    # train_img1, train_img2, train_label = next(iter(train_dataloader))

    wandb.watch(model, log_freq=50, log="all")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.7, 0.99))

    last_loss = 1000000
    for epoch in range(config.epochs):
        loss_mean = 0
        batches = 0
        for batch, (x1,x2,y) in enumerate(train_dataloader):
            out1,out2 = model.forward(x1.to(device),x2.to(device))
            loss = model.loss_contrastive(net1=out1,net2=out2,target=y.to(device),margin=2.5,distance_metric='eucl')
            loss_mean += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batches += 1

        wandb.log({"loss": loss_mean})
        print(f"Epoch: {epoch} Loss: {loss_mean}")
        if loss_mean < last_loss:
            torch.save(model.state_dict(), os.path.join(models_root,model_name.format(epoch, 0)))
            last_loss = loss_mean

        train_data.reset()