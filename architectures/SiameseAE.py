from configuration import configuration
from torch import nn
import torch
from SiameseBase import SiameseBase
import numpy as np

class SiameseAE(SiameseBase):
    def __init__(self, batch_size, in_channels=1, device='cuda'):
        super(SiameseAE, self).__init__(batch_size, in_channels, device)
        self.config = configuration()
        self.encoder()
        self.decoder()

    def encoder(self):
        f = self.config.filters
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=f[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(f[0])
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=f[0], out_channels=f[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(f[1])
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=f[1], out_channels=f[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f[2])
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=f[2], out_channels=f[3], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(f[3])
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=f[3], out_channels=f[4], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f[4])
        )
        self.cnn6 = nn.Sequential(
            nn.Conv2d(in_channels=f[4], out_channels=f[5], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(f[5])
        )
        self.glob_pool = nn.Sequential(
            nn.Flatten(2, -1)
        )
        self.cnn7 = nn.Sequential(
            nn.Conv2d(in_channels=f[5], out_channels=self.config.out_channels, kernel_size=(1, 1))
        )

    def decoder(self, size=(10,10)):
        f = self.config.filters
        self.cnn7d = nn.Sequential(
            nn.Conv2d(in_channels=self.config.out_channels, out_channels=f[5], kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.Upsample(size=size),
            nn.BatchNorm2d(f[5])
        )
        self.cnn6d = nn.Sequential(
            nn.Conv2d(in_channels=f[5], out_channels=f[4], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(f[4])
        )
        self.cnn5d = nn.Sequential(
            nn.Conv2d(in_channels=f[4], out_channels=f[3], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f[3])
        )
        self.cnn4d = nn.Sequential(
            nn.Conv2d(in_channels=f[3], out_channels=f[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(f[3])
        )
        self.cnn3d = nn.Sequential(
            nn.Conv2d(in_channels=f[2], out_channels=f[1], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f[1])
        )
        self.cnn2d = nn.Sequential(
            nn.Conv2d(in_channels=f[1], out_channels=f[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(f[0])
        )
        self.cnn1d = nn.Sequential(
            nn.Conv2d(in_channels=f[0], out_channels=f[0], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(f[0])
        )
        self.cnnd = nn.Sequential(
            nn.Conv2d(in_channels=f[0], out_channels=self.in_channels, kernel_size=(3,3), padding=(1,1))
        )

    def forward_branch(self, x):
        net1 = self.cnn1(x)
        net1 = self.cnn2(net1)
        net1 = self.cnn3(net1)
        net1 = self.cnn4(net1)
        net1 = self.cnn5(net1)
        net1 = self.cnn6(net1)
        net1 = self.glob_pool(net1)
        net1 = torch.mean(net1, 2)
        net1 = torch.unsqueeze(net1, 2)
        net1 = torch.unsqueeze(net1, 3)
        self.latent = self.cnn7(net1)
        net2 = self.cnn7d(self.latent)
        net2 = self.cnn6d(net2)
        net2 = self.cnn5d(net2)
        net2 = self.cnn4d(net2)
        net2 = self.cnn3d(net2)
        net2 = self.cnn2d(net2)
        net2 = self.cnn1d(net2)
        reconstruct = self.cnnd(net2)
        return self.latent, reconstruct


    def forward(self, x1, x2):
        # branch 1
        latent1, reconstruct1 = self.forward_branch(x1)
        # branch 2
        latent2, reconstruct2 = self.forward_branch(x2)
        return (latent1, latent2), (reconstruct1, reconstruct2)

    def loss_reconstrastive(self, out1, out2, target):
        latent1, reconstruct1 = out1
        latent2, reconstruct2 = out2
        L_sim = torch.mul(torch.sqrt(torch.sum(torch.pow(torch.subtract(latent1, latent2), exponent=2), dim=1)), target)
        L_rec = torch.mean(torch.square(reconstruct1-reconstruct2))
        return L_sim, L_rec

if __name__ == '__main__':
    from configuration import configuration

    config = configuration('../config.json')
    image1 = torch.from_numpy(np.zeros([1] + list(config.image_size), np.float32))
    image2 = image1 + 10
    net = SiameseAE(16)
    net.forward(image1,image2)
    print(net)
    print(list(net.parameters()))
    with open('../models/test/model_config.txt', 'w') as file:
        file.write(str(net))

