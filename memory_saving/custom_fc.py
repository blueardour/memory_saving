
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
    
class linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, training=True, fp_forward=False, clip_val=None, level=256, non_negative_only=True, iteration=None, ema_decay=None, groups=None, stochastic_round=False):

        input = custom_quant.Quant.forward(ctx, x, training, fp_forward, clip_val, level, non_negative_only, iteration, ema_decay, groups, stochastic_round)
        ctx.save_for_backward(weight, bias)

        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = grad_clip = None

        weight, bias = ctx.saved_tensors
        input = custom_quant.Quant.restore(ctx)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(dtype=grad_output.dtype))

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input.to(dtype=grad_output.dtype))

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        if ctx.needs_input_grad[5] and grad_input is not None:
            grad_clip = custom_quant.Quant.backward(ctx, grad_input)
        else:
            setattr(ctx, 'clip_val{}'.format('_'), None)
            setattr(ctx, 'non_negative_only{}'.format('_'), None)
            setattr(ctx, 'level{}'.format('_'), None)

        return grad_input, grad_weight, grad_bias, None, None, grad_clip, None, None, None, None, None, None

class Linear(nn.Linear, custom_quant.Quant):
    def __init__(self, in_features, out_features, bias=True, \
            memory_saving=False, args=None, logger=None, groups=1):
        super(Linear, self).__init__(in_features, out_features, bias=bias)
        custom_quant.Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger, groups=groups)
        self.tag = 'fc'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable:
            y = linear.apply(x, self.weight, self.bias, self.training, self.fp_forward, self.clip_val, self.level,
                             self.non_negative_only, self.iteration, self.ema_decay, self.groups, self.stochastic_round)
        else:
            y = F.linear(x, self.weight, self.bias)
        return y

        # if self.init_phase:
        #     assert self.training == False
        #     self.init_base_on_search(x)
        #     y = F.linear(x, self.weight, self.bias)
        #     return y

        # Previous 'warmup' can not save memory when training for the first few iterations
        #
        # if self.enable:
        #     if self.stable > self.iteration.item():
        #         self.init_based_on_warmup(x)
        #         y = F.linear(x, self.weight, self.bias)
        #     else:
        #         y = linear.apply(x, self.weight, self.bias, self.training, self.fp_forward, self.clip_val, self.level, \
        #             self.non_negative_only, self.iteration, self.ema_decay)
        # else:
        #     y = F.linear(x, self.weight, self.bias)
        # return y

if __name__ == "__main__":
    model = Linear(100, 100)
    print(model)
    model.enable = True
    print(model)

