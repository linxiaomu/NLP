import torch
import torch.nn as nn

# torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
# 用正太分布的值填充输入张量
w1 = torch.empty(3, 5)
print(w1)
print(nn.init.normal_(w1))


w2 = torch.empty(3, 5)
print(w2)
print(nn.init.xavier_normal_(w2))