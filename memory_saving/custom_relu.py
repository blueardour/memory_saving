
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import packbit

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inplace=False, training=False, dim=1):
        if inplace:
            output = x.clamp_(min=0)
        else:
            output = x.clamp(min=0)

        if training:
            y = x <= 0
            y = packbit.packbits_padded(y, dim=dim) 
            ctx.relu_flag = y
            ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.relu_flag
        y = packbit.unpackbits_padded(y, dim=ctx.dim).to(dtype=torch.bool)
        grad_input = grad_output.masked_fill(y, 0)
        return grad_input, None, None, None

class ReLU(nn.ReLU):
    def __init__(self, inplace=False, dim=1, memory_saving=True):
        super(ReLU, self).__init__(inplace)
        self.dim = dim
        self.memory_saving = memory_saving

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.memory_saving:
            return "ms.ReLU(inplace = {}, dim={})".format(self.inplace, self.dim)
        else:
            return "ReLU({})".format(self.inplace)

    def forward(self, x):
        if self.memory_saving:
            y = relu.apply(x, self.inplace, self.training, self.dim)
        else:
            y = F.relu(x, inplace=self.inplace)
        return y

