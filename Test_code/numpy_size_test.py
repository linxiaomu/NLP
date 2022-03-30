import torch
from torch.utils.tensorboard import SummaryWriter
# print(torch.rand(2,2,3,3))

X = torch.rand(2,6,6)
print(X)
import torch as t
import torch.nn as nn


class A(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 5, 3)
        self.conv2 = nn.Conv2d(5, 2, 3)
        self.conv3 = nn.Conv2d(2, 2, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)

        return x

a = A()
# print(list(a.parameters()))
# print(list(a.named_parameters()))

writer = SummaryWriter()
writer.add_graph(a,X)
