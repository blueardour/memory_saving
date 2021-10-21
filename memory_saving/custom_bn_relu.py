
import torch
import torch.nn as nn
import torch.nn.functional as F

if 'memory_saving' not in __name__:
    import custom_bn
    import custom_relu
else:
    from . import custom_bn
    from . import custom_relu

class batchnorm2d_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mean, var, average_factor, training, need_sync, process_group, world_size, eps, inplace, dim=1, keep_tensor=True):
        x = custom_bn.batchnorm2d.forward(ctx, input, weight, bias, mean, var, average_factor, training, need_sync, process_group, world_size, eps)
        if training:
            ctx.tmp = ctx.bn_input
            ctx.bn_input = None

        x = custom_relu.relu.forward(ctx, x, inplace, training, dim, keep_tensor)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # restore
        if ctx.bn_input is None:
            if not ctx.need_sync:
                if hasattr(ctx, 'bn_output') and ctx.bn_output is not None:
                    bn_output = ctx.bn_output
                else:
                    bn_output = ctx.relu_output # there should be mapping from relu_output to relu_input
                bn_weight, bn_bias, bn_running_mean, bn_running_var, bn_save_mean, bn_save_var, bn_reverse, bn_eps = ctx.bn_parameter
                ctx.bn_input = torch.batch_norm_elemt(bn_output, torch.reciprocal(bn_save_var), bn_save_mean, bn_bias, torch.reciprocal(bn_weight), bn_eps)
                bn_output = None

        # ReLU
        grad_input, _, _, _, _ = custom_relu.relu.backward(ctx, grad_output)

        # BN
        grad_input, grad_weight, grad_bias, _, _, _, _, _, _, _, _ = custom_bn.batchnorm2d.backward(ctx, grad_input)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None, None, None

class BatchNorm2d(custom_bn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, inplace=False, relu=None, args=None, logger=None):
        super(BatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, \
            args=args, logger=logger)
        self.relu = relu
        if relu is None:
            self.relu = custom_relu.ReLU(inplace=inplace, args=args, logger=logger)

    def forward(self, x):
        if self.enable:
            average_factor, training, mean, var, need_sync, process_group, world_size = custom_bn.bn_pre_forward(self, x)
            y = batchnorm2d_relu.apply(x, self.weight, self.bias, mean, var, \
                average_factor, training, need_sync, process_group, world_size, self.eps, self.relu.inplace)
        else:
            y = super().forward(x)
            y = F.relu(y, inplace=self.relu.inplace)
        return y

if __name__ == "__main__":
    model = BatchNorm2d(64, inplace=True, args=None)
    print(model)



