
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

class gelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=False, fp_forward=False, clip_val=None, level=256, non_negative_only=True):
        x = custom_quant.Quant.forward(ctx, x, training, fp_forward, clip_val, level, non_negative_only)
        y = F.gelu(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_clip = None

        x = custom_quant.Quant.restore(ctx)
        if x.is_cuda:
            grad_input = native.gelu_backward_cuda(grad_output, x)
        else:
            grad_input = native.gelu_backward_cpu(grad_output, x)

        grad_clip = custom_quant.Quant.backward(ctx, grad_input)
        return grad_input, None, None, grad_clip, None, None

class GELU(nn.GELU, custom_quant.Quant):
    def __init__(self, memory_saving=False, args=None, logger=None):
        super(GELU, self).__init__()
        custom_quant.Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger)
        self.tag = 'gelu'

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.enable:
            y = gelu.apply(x, self.training, self.fp_forward, self.clip_val.abs(), self.level, self.non_negative_only)
        else:
            y = F.gelu(x)
        return y

if __name__ == "__main__":
    model = GELU()
    print(model)
    model.enable = True
    print(model)

    import memory_saving as  ms
    model = ms.GELU()
    print(model)
    model.enable = True
    print(model)
