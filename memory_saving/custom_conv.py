
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

if 'memory_saving' not in __name__:
    import packbit
    import native
else:
    from . import packbit
    from . import native
# Uniform Quantization based Convolution
class conv2d_uniform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, interval=None, level=255, non_negative_only=True, training=True):
        # quant
        is_filtered = None
        if level < 256:
            if isinstance(interval, torch.Tensor) and interval.dtype != x.dtype:
                interval = interval.to(dtype=x.dtype)
            if non_negative_only:
                is_filtered = x.ge(interval.item())
                x = torch.where(is_filtered, interval, x)
                scale = interval / level
                x.div_(scale)
                y = torch.round(x)
                x = y.mul(scale)
                if training:
                    # pack
                    is_filtered = packbit.packbits_padded(is_filtered, dim=1)
                    if level < 256 and level > 15:
                        y = y.to(dtype=torch.uint8)
                    elif level < 16 and level > 3:
                        y = packbit.packbits_padded(y, dim=1, mask=0b1111)
                    else:
                        raise NotImplementedError
                else:
                    y = None
                    is_filtered = None
            else:
                raise NotImplementedError
        else:
            y = x
            interval = None

        # conv
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        # save tensor
        if training: # or True:
            #print("saved_tensor id in custom_conv-> y: {} with {}, is_filtered: {}\n".format(id(y), y.shape, id(is_filtered) if is_filtered is not None else None))
            ctx.conv_input = y
            ctx.conv_weight = (weight, bias, interval, is_filtered)
            ctx.hyperparameters_conv = (stride, padding, dilation, groups)
            ctx.hyperparameters_quant= (level, non_negative_only)
        return x

    @staticmethod
    def restore_conv_input(ctx):
        # restore
        y = ctx.conv_input
        weight, bias, interval, is_filtered = ctx.conv_weight
        level, non_negative_only = ctx.hyperparameters_quant

        if level > 255:
            x = y
        elif level < 256 and level > 15:
            x = y.to(dtype=interval.dtype)
            x = x.mul_(interval / level)
        elif level < 16 and level > 3:
            x = packbit.unpackbits_padded(y, dim=1, mask=0b1111).to(dtype=interval.dtype)
            x = x.mul_(interval / level)
        else:
            raise NotImplementedError("level: {}".format(level))

        ctx.conv_input = None
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias, grad_interval = None, None, None, None

        weight, bias, interval, is_filtered = ctx.conv_weight
        stride, padding, dilation, groups = ctx.hyperparameters_conv
        level, non_negative_only = ctx.hyperparameters_quant

        x = conv2d_uniform.restore_conv_input(ctx)

        # conv
        benchmark = True
        deterministic = True
        allow_tf32 = True
        output_mask = [True, True] #ctx.needs_input_grad[:2]
        if torch.__version__ >= "1.7":
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups, \
                    benchmark, deterministic, allow_tf32, output_mask)
        else:
            grad_input, grad_weight = native.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups, \
                    benchmark, deterministic, output_mask)
        x = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        # quant
        if is_filtered is not None and interval is not None: # and False:
            # unpack
            is_filtered = packbit.unpackbits_padded(is_filtered, dim=1).to(dtype=torch.bool)
            grad_interval = grad_input.masked_select(is_filtered).sum().reshape(interval.shape)
            grad_input.masked_fill_(is_filtered, 0.0)

        ctx.conv_input = None
        ctx.conv_weight = None
        ctx.hyperparameters_conv = None
        ctx.hyperparameters_quant= None
        return grad_input, grad_weight, grad_bias, None, None, None, None, \
                grad_interval, None, None, None


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, \
            memory_saving=False, args=None, logger=None, force_fp=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, \
                dilation=dilation, groups=groups, bias=bias)

        # constraints
        if self.padding_mode != 'zeros':
            raise NotImplementedError

    def forward(self, x):
        if self.memory_saving:
            if self.iteration.data < self.stable:
                self.init_based_on_warmup(x)
                y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                y = conv2d_uniform.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, \
                        self.clip_val.abs(), self.level, self.non_negative_only, self.training)
        else:
            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        self.iteration.add_(1)
        return y

