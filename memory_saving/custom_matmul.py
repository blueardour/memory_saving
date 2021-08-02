
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
    def forward(ctx, input1, input2, training=True, \
            fp_forward1=False, clip_val1=None, level1=256, non_negative_only1=True, iteration1=None, ema_decay1=None, groups1=None, stochastic_round1=False, shift1=None, \
            fp_forward2=False, clip_val2=None, level2=256, non_negative_only2=True, iteration2=None, ema_decay2=None, groups2=None, stochastic_round2=False, shift2=None ):
  
        input1 = custom_quant.Quant.forward(ctx, input1, training, fp_forward1, clip_val1, level1, non_negative_only1, iteration1, ema_decay1, groups1, stochastic_round1, shift1, '_1')
        input2 = custom_quant.Quant.forward(ctx, input2, training, fp_forward2, clip_val2, level2, non_negative_only2, iteration2, ema_decay2, groups2, stochastic_round2, shift2, '_2')
        output = input1.matmul(input2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input1 = grad_input2 = grad_clip1 = grad_clip2 = None

        input1 = custom_quant.Quant.restore(ctx, '_1')
        input2 = custom_quant.Quant.restore(ctx, '_2')

        if ctx.needs_input_grad[0]:
            grad_input1 = grad_output.matmul(input2.transpose(-2, -1).to(dtype=grad_output.dtype))
        if ctx.needs_input_grad[1]:
            grad_input2 = input1.transpose(-2, -1).to(dtype=grad_output.dtype).matmul(grad_output)

        if ctx.needs_input_grad[4] and grad_input1 is not None:
            grad_clip1 = custom_quant.Quant.backward(ctx, grad_input1, '_1')
        else:
            setattr(ctx, 'clip_val{}'.format('_1'), None)
            setattr(ctx, 'shift{}'.format('_1'), None)
            setattr(ctx, 'non_negative_only{}'.format('_1'), None)
            setattr(ctx, 'level{}'.format('_1'), None)

        if ctx.needs_input_grad[11] and grad_input2 is not None:
            grad_clip2 = custom_quant.Quant.backward(ctx, grad_input2, '_2')
        else:
            setattr(ctx, 'clip_val{}'.format('_2'), None)
            setattr(ctx, 'shift{}'.format('_2'), None)
            setattr(ctx, 'non_negative_only{}'.format('_2'), None)
            setattr(ctx, 'level{}'.format('_2'), None)

        return grad_input1, grad_input2, None, None, grad_clip1, None, None, None, None, None, None, None, None, grad_clip2, None, None, None, None, None, None, None

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
            y = matmul.apply(x1, x2, self.training, \
                             self.quant1.fp_forward, self.quant1.clip_val, self.quant1.level,
                             self.quant1.non_negative_only, self.quant1.iteration, self.quant1.ema_decay, self.quant1.groups, self.quant1.stochastic_round, self.quant1.shift,
                             self.quant2.fp_forward, self.quant2.clip_val, self.quant2.level,
                             self.quant2.non_negative_only, self.quant2.iteration, self.quant2.ema_decay, self.quant2.groups, self.quant2.stochastic_round, self.quant2.shift)
        else:
            y = torch.matmul(x1, x2)
        return y

        # if self.quant1.init_phase and self.quant2.init_phase:
        #     assert self.training == False
        #     self.quant1.init_base_on_search(x1)
        #     self.quant2.init_base_on_search(x2)
        #     y = torch.matmul(x1, x2)
        #     return y



    # def forward(self, x1, x2):
    #     if self.quant1.enable and self.quant2.enable:
    #         if self.quant1.stable > self.quant1.iteration.item() and self.quant2.stable > self.quant2.iteration.item():
    #             self.quant1.init_based_on_warmup(x1)
    #             self.quant2.init_based_on_warmup(x2)
    #             y = torch.matmul(x1, x2)
    #         else:
    #             y = matmul.apply(x1, x2, self.training, \
    #             self.quant1.fp_forward, self.quant1.clip_val, self.quant1.level, self.quant1.non_negative_only, self.quant1.iteration, self.quant1.ema_decay,\
    #             self.quant2.fp_forward, self.quant2.clip_val, self.quant2.level, self.quant2.non_negative_only, self.quant2.iteration, self.quant2.ema_decay,)
    #     else:
    #         y = torch.matmul(x1, x2)
    #     return y


if __name__ == "__main__":
    model = MatMul()
    print(model)

    model.quant1.enable = True
    model.quant2.enable = True
    print(model)

