from torch import nn
import torch
from .SiameseBase import SiameseBase

class SiameseSimple(SiameseBase):
    def __init__(self, batch_size, in_channels=1, device='cuda'):
        super(SiameseSimple, self).__init__(batch_size, in_channels, device)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            nn.BatchNorm2d(16)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            nn.BatchNorm2d(32)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(64)
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)
        )
        self.cnn6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(64)
        )
        self.cnn7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128)
        )
        self.glob_pool = nn.Sequential(
            nn.Flatten(2,-1)
        )
        self.cnn8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=(1,1))
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
    net = SiameseSimple(16)
    print(net)
    print(list(net.parameters()))
    with open('../models/test/model_config.txt', 'w') as file:
        file.write(str(net))


