
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

from . import native
from . import packbit

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
                    if level == 255:
                        y = y.to(dtype=torch.uint8)
                    else:
                        raise NotImplementedError
                else:
                    y = None
                    is_filtered = None
            else:
                raise NotImplementedError
        else:
            y = x
            interval=None

        # conv
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        # save tensor
        if training: # or True:
            print("saved_tensor id in custom_conv-> y: {} with {}, is_filtered: {}\n".format(id(y), y.shape, id(is_filtered) if is_filtered is not None else None))
            ctx.conv_input = y
            ctx.conv_weight = (weight, bias, interval, is_filtered)
            ctx.hyperparameters_conv = (stride, padding, dilation, groups)
            ctx.hyperparameters_quant= (level, non_negative_only)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias, grad_interval = None, None, None, None

        # restore
        y = ctx.conv_input
        weight, bias, interval, is_filtered = ctx.conv_weight
        stride, padding, dilation, groups = ctx.hyperparameters_conv
        level, non_negative_only = ctx.hyperparameters_quant

        if level < 256:
            x = y.to(dtype=interval.dtype)
            x = x.mul_(interval / level)
        else:
            x = y

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

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        # quant
        if is_filtered is not None and interval is not None: # and False:
            grad_interval = grad_input.masked_select(is_filtered).sum().reshape(interval.shape)
            grad_input.masked_fill_(is_filtered, 0.0)

        ctx.conv_input = x = y = None
        ctx.conv_weight = None
        ctx.hyperparameters_conv = None
        ctx.hyperparameters_quant= None
        return grad_input, grad_weight, grad_bias, None, None, None, None, \
                grad_interval, None, None, None


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, \
            memory_saving=True, args=None, logger=None, force_fp=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, \
                dilation=dilation, groups=groups, bias=bias)

        # constraints
        if self.padding_mode != 'zeros':
            raise NotImplementedError

        self.memory_saving = memory_saving

        # lsq
        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.clip_val = nn.Parameter(torch.Tensor([1.0]))
        self.level = 255
        self.stable = -1
        self.correlate = 1.0
        self.non_negative_only = True
        self.tag = 'fm'
        self.index = -1
        self.logger = logger
        self.args = args
        self.string = 'ms.cc.'
        self.repr = super(Conv2d, self).__repr__()

        if logger is None:
            if hasattr(args, 'logger'):
                self.logger = args.logger
            else:
                self.logger = logging.getLogger(__name__)

        if args is not None:
            if hasattr(args, 'fm_bit') and args.fm_bit is not None and args.fm_bit <= 8:
                self.level = int(2 ** args.fm_bit) - 1
            if hasattr(args, 'fm_level') and args.fm_level is not None and args.fm_level <= 256:
                self.level = args.fm_level
            if hasattr(args, 'fm_boundary') and args.fm_boundary is not None:
                self.clip_val.fill_(args.fm_boundary)
            if hasattr(args, 'fm_stable'):
                self.stable = args.fm_stable
            if hasattr(args, 'fm_correlate'):
                self.correlate = args.fm_correlate
            if hasattr(args, 'fm_enable'):
                self.memory_saving = self.memory_saving or args.fm_enable
            if hasattr(args, 'fm_nno'):
                self.non_negative_only = self.non_negative_only and args.nno
            if hasattr(args, 'fm_half_range'):
                self.non_negative_only = self.non_negative_only and args.fm_half_range
            self.logger.info("index({})-clip_val({})-level({})-stable({})-correlate({})-non_negative_only({})".format(
                self.index, self.clip_val.item(), self.level, self.stable, self.correlate, self.non_negative_only))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.memory_saving:
            string = self.string + self.repr + "-clip_val({})-level({})-stable({})-correlate({})-non_negative_only({})".format(
                self.clip_val.item(), self.level, self.stable, self.correlate, self.non_negative_only)
        else:
            string = self.repr
        if hasattr(self, 'norm'):
            string += "\n\t-" + str(self.norm)
        return string

    def init_based_on_warmup(self, data=None):
        if not self.memory_saving and data is None:
            return

        iteration = self.iteration.item()
        with torch.no_grad():
            max_value = data.abs().max().item()
            if hasattr(self, 'clip_val') and isinstance(self.clip_val, torch.Tensor):
                if self.correlate > 0:
                    max_value = max_value * self.correlate
                self.clip_val.data = max_value + iteration * self.clip_val.data
                self.clip_val.div_(iteration + 1)
                if iteration == (self.stable - 1):
                    self.logger.info('update %s clip_val for index %d to %r' % (self.tag, self.index, self.clip_val.item()))

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

