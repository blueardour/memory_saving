
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import custom_conv
from . import custom_bn

class conv2d_bn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, \
            interval, level, non_negative_only, \
            bn_weight, bn_bias, bn_mean, bn_var, average_factor, bn_training, need_sync, process_group, world_size, bn_eps):

        # conv
        x = custom_conv.conv2d_uniform.forward(ctx, input, weight, bias, stride, padding, dilation, groups, \
                interval, level, non_negative_only, bn_training)

        # bn
        x = custom_bn.batchnorm2d.forward(ctx, x, bn_weight, bn_bias, bn_mean, bn_var, \
                average_factor, bn_training, need_sync, process_group, world_size, bn_eps)
        ctx.bn_input = None

        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, grad_interval = None, None, None, None, None, None

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

        # checkpoint
        ctx.bn_input = z = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        # overwrite
        ctx.conv_input = x
        ctx.level = 256
        y = None

        # bn
        grad_output, grad_bn_weight, grad_bn_bias, _, _, _, _, _, _, _, _ = custom_bn.batchnorm2d.backward(ctx, grad_output)

        # conv
        grad_input, grad_weight, grad_bias, _, _, _, _, grad_interval, _, _, _ = custom_conv.conv2d_uniform.backward(ctx, grad_output)

        return grad_input, grad_weight, grad_bias, None, None, None, None, \
                grad_interval, None, None, \
                grad_bn_weight, grad_bn_bias, None, None, None, None, None, None, None, None, \

class Conv2d(custom_conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, \
            norm=None, memory_saving=True, args=None, logger=None, force_fp=False):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, \
                dilation=dilation, groups=groups, bias=bias, memory_saving=memory_saving, args=args, logger=logger)

        self.string = 'ms.'

        # norm
        self.norm = norm
        if self.norm is None:
            if torch.cuda.device_count() > 1:
                self.norm = nn.SyncBatchNorm(out_channels)
            else:
                self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.register_norm(norm)

    def register_norm(self, norm=None):
        if isinstance(norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            self.norm = norm
            self.logger.info("register existing norm layer in ms.Conv2d")
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.memory_saving:
            level = self.level
            if self.iteration.data < self.stable:
                self.init_based_on_warmup(x)
                level = 256
            average_factor, training, mean, var, need_sync, process_group, world_size = custom_bn.bn_pre_forward(self.norm, x)
            y = conv2d_bn.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, \
                    self.clip_val.abs(), level, self.non_negative_only, \
                    self.norm.weight, self.norm.bias, mean, var, \
                    average_factor, training, need_sync, process_group, world_size, self.norm.eps)
        else:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            y = self.norm(x)
        self.iteration.add_(1)
        return y

