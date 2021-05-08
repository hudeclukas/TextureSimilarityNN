import numpy as np
import torch
from model_factory import prepare_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ["WANDB_MODE"]='offline'
import wandb

from network import SiameseNetwork
from dataset_io import SimilarityDataset


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device.format(device)} device')

    wandb.init(project="texsim2018", entity='shamann', group='FirstTraining', config={
        "image_size":(1,150,150),
        "batch_size": 6,
        "architecture": "SiameseNetwork",
        "dataset": "SimTex2018",
    })
    config = wandb.config

    models_root = 'models/FirstTraining_e100_b32_lr001'
    model, model_name = prepare_model(SiameseNetwork, models_root, device)

    test_data = SimilarityDataset('data/test', max_samples=50, max_images=4)
    test_dataloader = DataLoader(test_data, config.batch_size, shuffle=True, num_workers=8)

    for batch, (x1, x2, y) in tqdm(enumerate(test_dataloader), total=int(np.ceil(len(test_data)/config.batch_size))):
        if len(x1) != config.batch_size:
            print(f" ... Skipping small batch <{len(x1)}>")
            continue

        out1, out2 = model.forward(x1.to(device), x2.to(device))
        distance, _ = model.distance_euclid(out1, out2)

        _, axs = plt.subplots(1, config.batch_size, figsize=[12, 4])
        for ii, (i1, i2, t, d) in enumerate(zip(x1, x2, y, distance)):
            axs[ii].imshow(torch.cat((i1, i2), dim=1).permute(1, 2, 0), cmap='gray')
            axs[ii].set_title("D: {:.3f} Y:{:.1f}".format(d.item(),t.item()))
        plt.show()
        plt.close()
