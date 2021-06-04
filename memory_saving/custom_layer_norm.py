
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import native
from . import custom_quant
from . import packbit

class layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, training=False, level=255, non_negative_only=True):
        if torch.__version__  >= "1.8":
            if x.is_cuda:
                y, mean, rstd = native.layer_norm_forward_cuda(x, normalized_shape, weight, bias, eps)
            else:
                y, mean, rstd = native.layer_norm_forward_cpu(x, normalized_shape, weight, bias, eps)

            if training:
                ctx.layer_norm_parameters = (mean, rstd, weight, bias, normalized_shape)
        else:
            N = 1
            if isinstance(normalized_shape, int):
                N = normalized_shape
            elif isinstance(normalized_shape, (list, tuple)):
                for i in normalized_shape:
                    N *= i
            else:
                raise RuntimeError("type of normalized_shape".format(type(normalized_shape)))
            M = x.nelement() // N

            if x.is_cuda:
                y, mean, rstd = native.layer_norm_forward_cuda(x, weight, bias, M, N, eps)
            else:
                y, mean, rstd = native.layer_norm_forward_cpu(x, weight, bias, M, N, eps)

            if training:
                ctx.layer_norm_parameters = (mean, rstd, weight, M, N)

        if training:
            ctx.layer_norm_input = x
        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        x = ctx.layer_norm_input

        output_mask = [ctx.needs_input_grad[0], ctx.needs_input_grad[2], ctx.needs_input_grad[3]]

        if torch.__version__  >= "1.8":
            mean, rstd, weight, bias, normalized_shape = ctx.layer_norm_parameters
            if grad_output.is_cuda:
                grad_input, grad_weight, grad_bias = \
                    native.layer_norm_backward_cuda(grad_output, x, normalized_shape, mean, rstd, weight, bias, output_mask)
            else:
                grad_input, grad_weight, grad_bias = \
                    native.layer_norm_backward_cpu(grad_output, x, normalized_shape, mean, rstd, weight, bias, output_mask)
        else:
            mean, rstd, weight, M, N = ctx.layer_norm_parameters

            if grad_output.is_cuda:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cuda(grad_output, x, mean, rstd, weight, M, N, output_mask)
            else:
                grad_input, grad_weight, grad_bias = native.layer_norm_backward_cpu(grad_output, x, mean, rstd, weight, M, N, output_mask)
        ctx.layer_norm_input = None
        ctx.layer_norm_parameters = None
        return grad_input, None, grad_weight, grad_bias, None, None, None, None

class LayerNorm(nn.LayerNorm, custom_quant.Quant):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, \
                memory_saving=False, args=None, logger=None):
        super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        custom_quant.Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger)

    def __repr__(self):
        return self.__str__()

    def forward(self, x):
        if self.memory_saving:
            y = layer_norm.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, \
                    self.training, self.level, self.non_negative_only)
        else:
            y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return y

