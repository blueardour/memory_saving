
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import native
from . import custom_quant
from . import packbit

class layer_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, bias, eps, training=False, fp_forward=False, clip_val=None, level=255, \
            non_negative_only=True, iteration=None, ema_decay=None, groups=None, stochastic_round=False, shift=None):
        if x.dtype != weight.data.dtype:
            x = x.to(dtype=weight.data.dtype)
        custom_quant.Quant.forward(ctx, x, training, fp_forward, clip_val, level, non_negative_only, iteration, ema_decay, groups, stochastic_round, shift)
        # x = custom_quant.Quant.forward(ctx, x, training, fp_forward, clip_val, level, non_negative_only, iteration, ema_decay, groups, stochastic_round, shift)
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

        return y

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_weight = grad_bias = grad_clip = None

        grad_output = grad_output.contiguous()
        x = custom_quant.Quant.restore(ctx)
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
        ctx.layer_norm_parameters = None

        # if ctx.needs_input_grad[7] and grad_input is not None:
        #     grad_clip = custom_quant.Quant.backward(ctx, grad_input)
        # else:
        #     setattr(ctx, 'clip_val{}'.format('_'), None)
        #     setattr(ctx, 'shift{}'.format('_'), None)
        #     setattr(ctx, 'non_negative_only{}'.format('_'), None)
        #     setattr(ctx, 'level{}'.format('_'), None)

        return grad_input, None, grad_weight, grad_bias, None, None, None, grad_clip, None, None, None, None, None, None, None

class LayerNorm(nn.LayerNorm, custom_quant.Quant):
    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, \
                memory_saving=False, args=None, logger=None, groups=1):
        super(LayerNorm, self).__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        custom_quant.Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger, groups=groups)
        self.tag = 'layernorm'

    def __repr__(self):
        return self.__str__()


    def forward(self, x):
        if self.enable and self.training:
            y = layer_norm.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, \
                                 self.training, self.fp_forward, self.clip_val, self.level, self.non_negative_only,
                                 self.iteration, self.ema_decay, self.groups, self.stochastic_round, self.shift)
        else:
            y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return y

        # if self.init_phase:
        #     assert self.training == False
        #     self.init_base_on_search(x)
        #     y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        #     return y



    # def forward(self, x):
    #     if self.enable:
    #         if self.stable > self.iteration.item():
    #             self.init_based_on_warmup(x)
    #             y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    #         else:
    #             y = layer_norm.apply(x, self.normalized_shape, self.weight, self.bias, self.eps, \
    #                 self.training, self.fp_forward, self.clip_val, self.level, self.non_negative_only, self.iteration, self.ema_decay)
    #     else:
    #         y = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    #     return y

