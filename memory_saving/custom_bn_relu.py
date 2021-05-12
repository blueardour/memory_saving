
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import custom_bn
from . import custom_relu

class batchnorm2d_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mean, var, average_factor, training, need_sync, process_group, world_size, eps, inplace, dim, keep_tensor):
        x = custom_bn.batchnorm2d.forward(ctx, input, weight, bias, mean, var, average_factor, training, need_sync, process_group, world_size, eps)

        if training:
            ctx.tmp = ctx.bn_input
            ctx.bn_input = None

        x = custom_relu.relu.forward(ctx, x, inplace, training, dim, keep_tensor)
        return x

    @staticmethod
    def backward(ctx, grad_output):

        y = ctx.relu_output

        grad_input, _, _, _, _ = custom_relu.relu.backward(ctx, grad_output)

        # restore
        ctx.bn_input = y
        ctx.bn_input = ctx.tmp

        grad_input, grad_weight, grad_bias, _, _, _, _, _, _, _, _ = custom_bn.batchnorm2d.backward(ctx, grad_input)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None

class BatchNorm2d(custom_bn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, inplace=False, dim=1, relu=None, memory_saving=False):
        super(BatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, memory_saving=memory_saving)

        self.relu = relu
        if relu is None:
            self.relu = custom_relu.ReLU(inplace=inplace, dim=dim, memory_saving=memory_saving, keep_tensor=True)

    def forward(self, x):
        if self.memory_saving:
            average_factor, training, mean, var, need_sync, process_group, world_size = custom_bn.bn_pre_forward(self, x)
            y = batchnorm2d_relu.apply(x, self.weight, self.bias, mean, var, \
                average_factor, training, need_sync, process_group, world_size, self.eps, self.relu.inplace, self.relu.dim, self.relu.keep_tensor)
        else:
            y = super().forward(x)
            y = self.relu(y)
        return y




