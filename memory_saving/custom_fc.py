
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

from . import custom_quant
from . import packbit

class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class Linear(nn.Linear, custom_quant.Quant):
    def __init__(self, in_features, out_features, bias=True, \
            memory_saving=False, args=None, logger=None):
        custom_quant.Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger)
        nn.Linear.__init__(self, in_features, out_features, bias=bias)
        self.repr = nn.Linear.__repr__(self)

    def forward(self, x):
        if self.memory_saving:
            y = linear.apply(x, self.weight, self.bias)
        else:
            y = F.linear(x, self.weight, self.bias)
        self.iteration.add_(1)
        return y

