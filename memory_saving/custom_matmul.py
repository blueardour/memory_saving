
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

if __name__ == "__main__":
    import custom_quant
    import packbit
    import native
else:
    from . import custom_quant
    from . import packbit
    from . import native
    
class matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class MatMul(nn.Module):
    def __init__(self, memory_saving=False, args=None, logger=None):
        super(MatMul, self).__init__()
        self.quant1 = custom_quant.Quant(memory_saving=memory_saving, args=args, logger=logger)
        self.quant2 = custom_quant.Quant(memory_saving=memory_saving, args=args, logger=logger)

    def forward(self, x1, x2):
        if self.quant1.memory_saving and self.quant2.memory_saving:
            y = matmul.apply(x1, x2)
        else:
            y = torch.matmul(x1, x2)
        return y


if __name__ == "__main__":
    model = MatMul()
    print(model)

