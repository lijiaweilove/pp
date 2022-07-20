import torch
from torch import nn
from torch.nn import Conv2d


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x


test = Test()
input = torch.randn(1,1,3,3)
print("input:{}".format(input))
output = test(input)
print("output：{}".format(output))
