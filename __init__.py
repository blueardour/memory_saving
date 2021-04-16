
import torch.nn as nn
from .memory_saving import custom_relu, custom_conv_bn, custom_conv

def ReLU(inplace=True):
    return custom_relu(inplace)


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, memory_saving=False, args=None, force_fp=False):
    if args is not None and hasattr(args, 'keyword'):
        if 'ms_conv' in args.keyword:
            return custom_conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                    memory_saving=memory_saving, args=args)
        elif 'ms_conv_bn' in args.keyword:
            return custom_conv_bn(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                    memory_saving=memory_saving, args=args)
        else:
            return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

