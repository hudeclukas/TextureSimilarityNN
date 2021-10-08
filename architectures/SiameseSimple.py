import numpy as np

from configuration import configuration
from torch import nn
import torch
from SiameseBase import SiameseBase

class SiameseSimple(SiameseBase):
    def __init__(self, batch_size, in_channels=1, device='cuda'):
        super(SiameseSimple, self).__init__(batch_size, in_channels, device)
        self.config = configuration()
        f = self.config.filters
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=f[0], kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            nn.BatchNorm2d(f[0])
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=f[0], out_channels=f[1], kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            nn.BatchNorm2d(f[1])
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=f[1], out_channels=f[2], kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f[2])
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=f[2], out_channels=f[3], kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(f[3])
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=f[3], out_channels=f[4], kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f[4])
        )
        self.cnn6 = nn.Sequential(
            nn.Conv2d(in_channels=f[4], out_channels=f[5], kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(f[5])
        )
        self.cnn7 = nn.Sequential(
            nn.Conv2d(in_channels=f[5], out_channels=f[6], kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(f[6])
        )
        self.glob_pool = nn.Sequential(
            nn.Flatten(2,-1)
        )
        self.cnn8 = nn.Sequential(
            nn.Conv2d(in_channels=f[6], out_channels=self.config.out_channels, kernel_size=(1,1))
        )

    def forward_branch(self, x):
        net1 = self.cnn1(x)
        net1 = self.cnn2(net1)
        net1 = self.cnn3(net1)
        net1 = self.cnn4(net1)
        net1 = self.cnn5(net1)
        net1 = self.cnn6(net1)
        net1 = self.cnn7(net1)
        net1 = self.glob_pool(net1)
        net1 = torch.mean(net1, 2)
        net1 = torch.unsqueeze(net1, 2)
        net1 = torch.unsqueeze(net1, 3)
        output1 = self.cnn8(net1)
        return output1

    def forward(self, x1, x2):
        # branch 1
        output1 = self.forward_branch(x1)
        # branch 2
        output2 = self.forward_branch(x2)
        return output1, output2


if __name__ == '__main__':
    from configuration import configuration
    config = configuration('../config.json')
    image = torch.from_numpy(np.zeros([1]+list(config.image_size), np.float32))
    net = SiameseSimple(16)
    out = net.forward(image,image)
    print(net)
    print(list(net.parameters()))
    with open('../models/test/model_config.txt', 'w') as file:
        file.write(str(net))


