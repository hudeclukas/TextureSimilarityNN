from torch import nn
import torch


def distance_euclid(tensor1, tensor2):
    euclid2 = torch.sum(torch.pow(torch.subtract(tensor1, tensor2), exponent=2), dim=1)
    euclid = torch.sqrt(euclid2)
    return torch.squeeze(euclid), torch.squeeze(euclid2)


def distance_canberra(tensor1, tensor2):
    canberra = torch.sum(
        torch.divide(torch.abs(torch.subtract(tensor1, tensor2)), torch.add(torch.abs(tensor1), torch.abs(tensor2))),
        dim=1)
    return canberra, 2*canberra

def distance_bray_curtis(tensor1, tensor2):
    distance = torch.abs(torch.sum(
        torch.divide(torch.abs(torch.subtract(tensor1, tensor2)), torch.add(tensor1, tensor2)),
        dim=1))
    return distance.squeeze(), 2*distance.squeeze()

def distance_mahalanobis(tensor1, tensor2):
    def cov(x,y):
        ddof=1
        fact = x.shape[1] - ddof
        c = torch.matmul(torch.unsqueeze(x.squeeze(),-1),torch.transpose(torch.unsqueeze(y.squeeze(),-1),dim0=1,dim1=2))
        c = c / fact
        return c.squeeze()
    delta = tensor1 - tensor2
    inv_cov = torch.inverse(cov(tensor1,tensor2))
    distance_sqr = torch.abs(torch.bmm(torch.bmm(torch.transpose(torch.squeeze(delta,dim=-1),dim1=1,dim0=2),torch.squeeze(inv_cov)),torch.squeeze(delta,dim=-1)))
    return torch.sqrt(distance_sqr).squeeze(), distance_sqr.squeeze()

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

    def loss_contrastive(self, net1, net2, target, margin, distance_metric):
        dist, dist2 = distance_metric(net1, net2)
        similar = torch.multiply(target, dist)
        dissimilar = torch.multiply(self.loss_ones - target, torch.pow(torch.maximum(self.loss_ones * margin - dist2, self.loss_max_zeros), 2))
        return torch.mean(torch.add(similar, dissimilar))

if __name__ == '__main__':
    net = SiameseBase(16)
    print(net)
    print(list(net.parameters()))
    with open('../models/test/model_config.txt', 'w') as file:
        file.write(str(net))


