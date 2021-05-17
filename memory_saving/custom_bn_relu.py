
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
        # restore
        bn_output = ctx.relu_input
        #if not ctx.need_sync:
        #    bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_save_mean, bn_save_var, bn_reverse, bn_eps = ctx.bn_parameter
        #    #print("shape", bn_output.shape, bn_bias.shape, bn_weight.shape, bn_save_var.shape, bn_save_mean.shape)
        #    bn_output = bn_output.subtract(bn_bias.reshape(1,-1,1,1))
        #    bn_output = bn_output.div(bn_weight.mul(torch.rsqrt(bn_save_var + bn_eps)).reshape(1,-1,1,1))
        #    bn_output = bn_output.add(bn_save_mean.reshape(1,-1,1,1))
        #    # compare with ctx.tmp
        #    compare = bn_output == ctx.tmp
        #    print("debug", compare.sum(), compare.size())
        #ctx.bn_input = bn_output
        out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        ctx.bn_input = ctx.tmp.masked_fill_(bn_output <= 0, 0)
        #ctx.bn_input = torch.where(bn_output < 0, bn_save_mean.reshape(1,-1,1,1), ctx.tmp)
        bn_output = None

        # ReLU
        grad_input, _, _, _, _ = custom_relu.relu.backward(ctx, grad_output)

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




