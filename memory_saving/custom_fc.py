
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import logging

if 'memory_saving' not in __name__:
    import custom_quant
else:
    from . import custom_quant
    
class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, clip_val=None, level=256, iteration=None, ema_decay=None, groups=None, shift=None):

        custom_quant.Quant.forward(ctx, x, clip_val, level, iteration, ema_decay, groups, shift)
        ctx.save_for_backward(weight, bias)
        return F.linear(x, weight, bias)
        # output = x.matmul(weight.t())
        # if bias is not None:
        #     output += bias.unsqueeze(0).expand_as(output)
        # return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = None

        weight, bias = ctx.saved_tensors
        input = custom_quant.Quant.restore(ctx)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(dtype=grad_output.dtype))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

class Linear(nn.Linear, custom_quant.Quant):
    def __init__(self, in_features, out_features, bias=True,
            memory_saving=False, args=None, logger=None, groups=1):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        custom_quant.Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger, groups=groups)
        self.tag = 'fc'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable and self.training:
            y = linear.apply(x, self.weight, self.bias, self.clip_val, self.level,
                             self.iteration, self.ema_decay, self.groups, self.shift)
        else:
            y = F.linear(x, self.weight, self.bias)
        return y

if __name__ == "__main__":
    model = Linear(100, 100)
    print(model)
    model.enable = True
    print(model)

