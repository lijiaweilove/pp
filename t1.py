import torch
from torch import nn
from torch.nn import Conv2d

from pcdet.utils.loss_utils import _transpose_and_gather_feat


class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        return x


test = Test()
input = torch.randn(1, 2, 248, 216)
ind = torch.randn(1, 500)
pred = _transpose_and_gather_feat(input, ind)
# print("input:{}".format(input))
# # output = test(input)
print("pred：{}".format(pred.shape))
