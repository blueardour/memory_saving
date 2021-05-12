
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import packbit

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inplace=False, training=False, dim=1, keep_tensor=True):
        if inplace:
            output = x.clamp_(min=0)
        else:
            output = x.clamp(min=0)

        if training:
            if keep_tensor:
                y = output
            else:
                y = x <= 0
                y = packbit.packbits_padded(y, dim=dim) 
            ctx.relu_output = y
            ctx.relu_dim = dim
            ctx.relu_keep_tensor = keep_tensor
        return output

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.relu_output
        if ctx.relu_keep_tensor:
            y = y <= 0
        else:
            y = packbit.unpackbits_padded(y, dim=ctx.relu_dim).to(dtype=torch.bool)
        grad_input = grad_output.masked_fill(y, 0)
        ctx.relu_output= None
        ctx.relu_dim = None
        ctx.relu_keep_tensor = None
        return grad_input, None, None, None, None

class ReLU(nn.ReLU):
    def __init__(self, inplace=False, dim=1, memory_saving=False, keep_tensor=False):
        super(ReLU, self).__init__(inplace)
        self.dim = dim
        self.memory_saving = memory_saving
        self.keep_tensor = keep_tensor

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.memory_saving:
            return "ms.ReLU(inplace = {}, dim={}, keep_tensor={})".format(self.inplace, self.dim, self.keep_tensor)
        else:
            return "ReLU({})".format(self.inplace)

    def forward(self, x):
        if self.memory_saving:
            y = relu.apply(x, self.inplace, self.training, self.dim, self.keep_tensor)
        else:
            y = F.relu(x, inplace=self.inplace)
        return y

