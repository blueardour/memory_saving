
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

import native
import custom_quant
import packbit

class gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=False):
        y = F.gelu(x)
        if training:
            ctx.gelu_input = x
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.gelu_input

        if x.is_cuda:
            grad_input = native.gelu_backward_cuda(grad_output, x)
        else:
            grad_input = native.gelu_backward_cpu(grad_output, x)
        ctx.gelu_input= None
        return grad_input, None

class GELU(nn.GELU, custom_quant.Quant):
    def __init__(self, memory_saving=False, args=None, logger=None):
        super(GELU, self).__init__()
        custom_quant.Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger)

    def forward(self, x):
        if self.memory_saving:
            y = gelu.apply(x, self.training)
        else:
            y = F.gelu(x)
        return y

if __name__ == "__main__":
    model = GELU()
    print(model)
    model.memory_saving = True
    print(model)

