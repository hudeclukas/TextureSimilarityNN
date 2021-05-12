from torch import nn
import torch


class SiameseBase(nn.Module):
    def __init__(self, batch_size, in_channels=1, device='cuda'):
        super(SiameseBase, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.loss_ones = torch.ones(size=(batch_size,), dtype=torch.float32, device=device)
        self.loss_max_zeros = torch.zeros(size=(batch_size,), dtype=torch.float32, device=device)
        self.in_channels = in_channels

    def forward(self, x1, x2):
        return x1, x2

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
    net = SiameseBase(16)
    print(net)
    print(list(net.parameters()))
    with open('../models/test/model_config.txt', 'w') as file:
        file.write(str(net))


