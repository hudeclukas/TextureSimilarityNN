from torch import nn
import torch


class SiameseNetwork(nn.Module):
    def __init__(self, batch_size, in_channels=1, device='cuda'):
        super(SiameseNetwork, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.loss_ones = torch.ones(size=(batch_size,), dtype=torch.float32, device=device)
        self.loss_max_zeros = torch.zeros(size=(batch_size,), dtype=torch.float32, device=device)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(3,3), padding=0)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(5,5), padding=(2,2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=0)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
        )
        self.glob_pool = nn.Sequential(
            nn.Flatten(2,-1)
        )
        self.cnn7 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=(1,1))
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

    def distance_euclid(self, tensor1, tensor2):
        euclid2 = torch.sum(torch.pow(torch.subtract(tensor1, tensor2), exponent=2), dim=1)
        euclid = torch.sqrt(euclid2)
        return torch.squeeze(euclid), torch.squeeze(euclid2)

    def distance_canberra(self, tensor1, tensor2):
        canberra = torch.sum(torch.divide(torch.abs(torch.subtract(tensor1, tensor2)),torch.add(torch.abs(tensor1),torch.abs(tensor2))), dim=1)
        return canberra, torch.pow(canberra, exponent=2)

    def loss_contrastive(self, net1, net2, target, margin, distance_metric:str):
        dist=0
        dist2=0
        if distance_metric=='eucl':
            dist, dist2 = self.distance_euclid(net1, net2)
        elif distance_metric=='canb':
            dist, dist2 = self.distance_canberra(net1, net2)
        similar = torch.multiply(target, dist)
        dissimilar = torch.multiply(self.loss_ones - target, torch.pow(torch.maximum(self.loss_ones * margin - dist2, self.loss_max_zeros), 2))
        return torch.mean(torch.add(similar, dissimilar))

if __name__ == '__main__':
    net = SiameseNetwork(16)
    print(net)
    print(list(net.parameters()))
    with open('models/test/model_config.txt','w') as file:
        file.write(str(net))


