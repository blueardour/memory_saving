
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from . import native
from . import packbit

class layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, level, training=False):
        y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        if training:
            ctx.gelu_input = x
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.gelu_input

        pdb.set_trace()
        grad_input = native.gelu_backward_cuda(grad_output, x)
        ctx.gelu_input= None
        return grad_input, None, None

class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, memory_saving=False, bit=8):
        super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.memory_saving = memory_saving
        self.bit = bit
        self.level = int(pow(2, self.bit) - 1)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.memory_saving:
            return "ms.GELU(bit={})".format(self.bit)
        else:
            return "GELU()"

    def forward(self, x):
        if self.memory_saving:
            y = layer_norm.apply(x, self.level, self.training)
        else:
            y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return y

