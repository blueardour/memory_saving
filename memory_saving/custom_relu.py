
import torch
import torch.nn as nn
import torch.nn.functional as F

import packbit

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inplace=False, dim=1):
        ctx.inplace = inplace
        if inplace:
            output = x.clamp_(min=0)
        else:
            y = x < 0
            #y = packbit.packbits_padded(y, dim=dim) 
            ctx.save_for_backward(y)
            ctx.dim = dim
            output = x.clamp(min=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.inplace:
            return grad_output, None, None
        else:
            y, = ctx.saved_tensors
            #y = packbit.unpackbits_padded(z, dim=ctx.dim)
            grad_input = grad_output.clone()
            grad_input[y] = 0
            return grad_input, None, None

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
            y = relu.apply(x, self.inplace, self.dim)
        else:
            y = F.relu(x, inplace=self.inplace)
        return y

