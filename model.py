import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class doublemodel(nn.Module):
    """Some Information about doublemodel"""
    def __init__(self, resnet1, resnet2):
        super(doublemodel, self).__init__()
        self.model1 = resnet1
        self.model2 = resnet2

    def forward(self, x, y):
        res1 = self.model1(x)
        res2 = self.model2(y)

        return res1 + res2