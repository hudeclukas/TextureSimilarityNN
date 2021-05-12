from torch import nn
import torch
from .SiameseBase import SiameseBase

class SiameseNetworkIWSSIP(SiameseBase):
    def __init__(self, batch_size, in_channels=1, device='cuda'):
        super(SiameseNetworkIWSSIP, self).__init__(batch_size, in_channels, device)

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3), padding=0),
            nn.BatchNorm2d(16)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            nn.BatchNorm2d(32)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
            nn.BatchNorm2d(64)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        self.cnn6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        self.glob_pool = nn.Sequential(
            nn.Flatten(2,-1)
        )
        self.cnn7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=(1,1))
        )

    def forward(self, x1, x2):
        # branch 1
        net1 = self.cnn1(x1)
        net1 = self.cnn2(net1)
        net1 = self.cnn3(net1)
        net1 = self.cnn4(net1)
        net1 = self.cnn5(net1)
        net1 = self.cnn6(net1)
        net1 = self.glob_pool(net1)
        net1 = torch.mean(net1, 2)
        net1 = torch.unsqueeze(net1, 2)
        net1 = torch.unsqueeze(net1, 3)
        output1 = self.cnn7(net1)
        # branch 2
        net2 = self.cnn1(x2)
        net2 = self.cnn2(net2)
        net2 = self.cnn3(net2)
        net2 = self.cnn4(net2)
        net2 = self.cnn5(net2)
        net2 = self.cnn6(net2)
        net2 = self.glob_pool(net2)
        net2 = torch.mean(net2, 2)
        net2 = torch.unsqueeze(net2, 2)
        net2 = torch.unsqueeze(net2, 3)
        output2 = self.cnn7(net2)

        return output1, output2


if __name__ == '__main__':
    net = SiameseNetworkIWSSIP(16)
    print(net)
    print(list(net.parameters()))
    with open('../models/test/model_config.txt', 'w') as file:
        file.write(str(net))


