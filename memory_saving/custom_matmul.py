
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

if 'memory_saving' not in __name__:
    import custom_quant
    import packbit
    import native
else:
    from . import custom_quant
    from . import packbit
    from . import native
    
class matmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2,
                clip_val1=None, level1=256, iteration1=None, ema_decay1=None, groups1=None, shift1=None,
                clip_val2=None, level2=256, iteration2=None, ema_decay2=None, groups2=None, shift2=None):
  
        # input1 = custom_quant.Quant.forward(ctx, input1, clip_val1, level1, iteration1, ema_decay1, groups1, shift1, '_1')
        # input2 = custom_quant.Quant.forward(ctx, input2, clip_val2, level2, iteration2, ema_decay2, groups2, shift2, '_2')
        custom_quant.Quant.forward(ctx, input1, clip_val1, level1, iteration1, ema_decay1, groups1, shift1, '_1')
        custom_quant.Quant.forward(ctx, input2, clip_val2, level2, iteration2, ema_decay2, groups2, shift2, '_2')
        output = input1.matmul(input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = None

        input1 = custom_quant.Quant.restore(ctx, '_1')
        input2 = custom_quant.Quant.restore(ctx, '_2')

        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output.matmul(input2.transpose(-2, -1).to(dtype=grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_input2 = input1.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)

        return grad_input1, grad_input2, None, None, None, None, None, None, None, None, None, None, None, None

class MatMul(nn.Module):
    def __init__(self, memory_saving=False, args=None, logger=None, groups=1):
        super(MatMul, self).__init__()
        self.quant1 = custom_quant.quantization(tag='matmul-1', groups=groups)
        self.quant2 = custom_quant.quantization(tag='matmul-2', groups=groups)
        self.tag = 'matmul'

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)

    def forward(self, x1, x2):
        if self.quant1.enable and self.quant2.enable and self.training:
            y = matmul.apply(x1, x2,
                             self.quant1.clip_val, self.quant1.level, self.quant1.iteration, self.quant1.ema_decay, self.quant1.groups, self.quant1.shift,
                             self.quant2.clip_val, self.quant2.level, self.quant2.iteration, self.quant2.ema_decay, self.quant2.groups, self.quant2.shift)
        else:
            y = torch.matmul(x1, x2)
        return y

if __name__ == "__main__":
    model = MatMul()
    print(model)

    model.quant1.enable = True
    model.quant2.enable = True
    print(model)

