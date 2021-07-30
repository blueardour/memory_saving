
import torch
import torch.nn as nn
import torch.nn.functional as F

if 'memory_saving' not in __name__:
    import custom_quant
    import packbit
    import native
else:
    from . import custom_quant
    from . import packbit
    from . import native

class softmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, training=False, \
            fp_forward1=False, clip_val1=None, level1=256, non_negative_only1=True, iteration1=None, ema_decay1=None, groups1=None, \
            fp_forward2=False, clip_val2=None, level2=256, non_negative_only2=True, iteration2=None, ema_decay2=None, groups2=None):

        x = custom_quant.Quant.forward(ctx, x, training, fp_forward1, clip_val1, level1, non_negative_only1, iteration1, ema_decay1, groups1, '_1')
        y = F.softmax(x, dim)
        y = custom_quant.Quant.forward(ctx, y, training, fp_forward2, clip_val2, level2, non_negative_only2, iteration2, ema_decay2, groups2, '_2')
        ctx.dim = dim
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_clip1 = grad_clip2 = None

        x = custom_quant.Quant.restore(ctx, '_1')
        y = custom_quant.Quant.restore(ctx, '_2')

        if ctx.needs_input_grad[11] and grad_output is not None:
            grad_clip2 = custom_quant.Quant.backward(ctx, grad_output, '_2')

        if x.is_cuda:
            grad_input = native.softmax_backward_cuda(grad_output, y, ctx.dim, x)
        else:
            grad_input = native.softmax_backward_cpu(grad_output, y, ctx.dim, x)

        if ctx.needs_input_grad[4] and grad_input is not None:
            grad_clip1 = custom_quant.Quant.backward(ctx, grad_input, '_1')

        return grad_input, None, None, None, grad_clip1, None, None, None, None, None, None, grad_clip2, None, None, None, None, None

class Softmax(nn.Softmax):
    def __init__(self, dim=None, memory_saving=False, args=None, logger=None, groups=1):
        super(Softmax, self).__init__(dim=dim)
        self.quant1 = custom_quant.quantization(tag='softmax-1', groups=groups)
        self.quant2 = custom_quant.quantization(tag='softmax-2', groups=groups)
        self.tag = 'softmax'

    def update_quantization_parameter(self, **parameters):
        self.quant1.update_quantization_parameter(**parameters)
        self.quant2.update_quantization_parameter(**parameters)

    def forward(self, x):
        if self.quant1.enable and self.quant2.enable:
            y = softmax.apply(x, self.dim, self.training,
                              self.quant1.fp_forward, self.quant1.clip_val, self.quant1.level,
                              self.quant1.non_negative_only, self.quant1.iteration, self.quant1.ema_decay, self.quant1.groups,
                              self.quant2.fp_forward, self.quant2.clip_val, self.quant2.level,
                              self.quant2.non_negative_only, self.quant2.iteration, self.quant2.ema_decay, self.quant2.groups)
        else:
            y = F.softmax(x, self.dim)
        return y

        # if self.quant1.init_phase and self.quant2.init_phase:
        #     assert self.training == False
        #     self.quant1.init_base_on_search(x)
        #     y = F.softmax(x, self.dim)
        #     self.quant2.init_base_on_search(y)
        #     return y


    # def forward(self, x):
    #     if self.quant1.enable and self.quant2.enable:
    #         if self.quant1.stable > self.quant1.iteration.item() and self.quant2.stable > self.quant2.iteration.item():
    #             self.quant1.init_based_on_warmup(x)
    #             y = F.softmax(x, self.dim)
    #             self.quant2.init_based_on_warmup(y)
    #         else:
    #             y = softmax.apply(x, self.dim, self.training, \
    #                 self.quant1.fp_forward, self.quant1.clip_val, self.quant1.level, self.quant1.non_negative_only, self.quant1.iteration, self.quant1.ema_decay, \
    #                 self.quant2.fp_forward, self.quant2.clip_val, self.quant2.level, self.quant2.non_negative_only, self.quant2.iteration, self.quant2.ema_decay,)
    #     else:
    #         y = F.softmax(x, self.dim)
    #     return y

if __name__ == "__main__":
    model = Softmax()
    print(model)
    model.enable = True
    print(model)

    import memory_saving as  ms
    model = ms.Softmax()
    print(model)
    model.enable = True
    print(model)
