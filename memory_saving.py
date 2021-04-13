
import torch
import torch.nn as nn
import torch.nn.functional as F
import setuptools

from torch.utils.cpp_extension import load

conv = load(name='conv', sources=['./conv2d.cpp'])

##########
class Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class conv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, clip=None, level=255):
        # quant
        x = input.div(clip)
        x = torch.clamp(x, min=0, max=1).mul(level)
        x = Round.apply(x)
        y = x.to(dtype=torch.uint8)

        # conv
        x = x.mul(clip / level)
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        # save tensor
        ctx.save_for_backward(y, weight, bias, clip)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.level = level
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias, grad_clip = None, None, None, None
        y, weight, bias, clip = ctx.saved_tensors
        x = y.to(dtype=torch.float)

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        level = ctx.level
        x = x.mul(clip / level)

        benchmark = True
        deterministic = True
        output_mask = ctx.needs_input_grad[:2]
        grad_input, grad_weight = conv.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        return grad_input, grad_weight, grad_bias, None, None, None, None, grad_clip, None


class custom_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, memory_saving=False, args=None):
        super(custom_conv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # constraints
        if self.padding_mode != 'zeros':
            raise NotImplementedError

        self.memory_saving = memory_saving
        if args is not None and hasattr(args, 'keyword') and hasattr(args, 'fm_enable'):
            self.memory_saving = self.memory_saving or args.fm_enable

        # configuration
        string = "ms.Conv2d"

        # lsq
        self.fm_clip_val = nn.Parameter(torch.Tensor([1.0]))
        self.fm_intervals = 256

        if args is not None:
            if hasattr(args, 'fm_bit') and args.fm_bit is not None and args.fm_bit <= 8:
                self.fm_intervals = int(2 ** args.fm_bit)
            if hasattr(args, 'fm_level') and args.fm_level is not None and args.fm_level <= 256:
                self.fm_intervals = args.fm_level

        self.fm_intervals = self.fm_intervals - 1

        string = string + "-clip_val({})-intervals({})".format(self.fm_clip_val.item(), self.fm_intervals)
        if self.memory_saving:
            self.string = string
        else:
            self.string = "Conv2d in ms"

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.string

    def forward(self, x):
        if self.memory_saving:
            y = conv2d.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.fm_clip_val.abs(), self.fm_intervals)
        else:
            y = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

def SyncBatchNorm_forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
    if not input.is_contiguous(memory_format=torch.channels_last):
        input = input.contiguous()
    if weight is not None:
        weight = weight.contiguous()

    size = int(input.numel() // input.size(1))
    if size == 1 and world_size < 2:
        raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

    # calculate mean/invstd for input.
    mean, invstd = torch.batch_norm_stats(input, eps)
    
    count = torch.full((1,), input.numel() // input.size(1), dtype=mean.dtype, device=mean.device)
    
    num_channels = input.shape[1]
    # C, C, 1 -> (2C + 1)
    combined = torch.cat([mean, invstd, count], dim=0)
    # world_size * (2C + 1)
    combined_list = [ torch.empty_like(combined) for k in range(world_size) ]
    # Use allgather instead of allreduce since I don't trust in-place operations ..
    dist.all_gather(combined_list, combined, process_group, async_op=False)
    combined = torch.stack(combined_list, dim=0)
    # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
    mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
    
    # calculate global mean & invstd
    mean, invstd = torch.batch_norm_gather_stats_with_counts(
        input,
        mean_all,
        invstd_all,
        running_mean,
        running_var,
        momentum,
        eps,
        count_all.view(-1)
    )
    
    self.bn_weight = weight
    self.bn_mean = mean
    self.bn_invstd = invstd
    self.bn_count_all = count_all.to(torch.int32)
    self.bn_process_group = process_group

    # apply element-wise normalization
    out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
    return out

def SyncBatchNorm_backward(saved_input, weight, mean, invstd, count_tensor, process_group, needs_input_grad, grad_output):
    if not grad_output.is_contiguous(memory_format=torch.channels_last):
        grad_output = grad_output.contiguous()
    #saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
    #process_group = self.process_group
    grad_input = grad_weight = grad_bias = None

    # calculate local stats as well as grad_weight / grad_bias
    sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
        grad_output,
        saved_input,
        mean,
        invstd,
        weight,
        True,
        needs_input_grad[0],
        needs_input_grad[1]
    )

    if True:
        # synchronizing stats used to calculate input gradient.
        num_channels = sum_dy.shape[0]
        combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
        torch.distributed.all_reduce(
            combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
        sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

        # backward pass for gradient calculation
        grad_input = torch.batch_norm_backward_elemt(
            grad_output,
            saved_input,
            mean,
            invstd,
            weight,
            sum_dy,
            sum_dy_xmu,
            count_tensor
        )

    # synchronizing of grad_weight / grad_bias is not needed as distributed
    # training would handle all reduce.
    if weight is None or not needs_input_grad[0]:
        grad_weight = None

    if weight is None or not needs_input_grad[1]:
        grad_bias = None

    return grad_input, grad_weight, grad_bias #, None, None, None, None, None, None

class conv2d_bn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, \
            bn_weight, bn_bias, average_factor, bn_training, need_sync, process_group, world_size, bn_mean, bn_var, bn_eps, \
            clip, level):
        # quant
        if level < 256:
            x = input.div(clip)
            x = torch.clamp(x, min=0, max=1).mul(level)
            x = Round.apply(x)
            y = x.to(dtype=torch.uint8)
            x = x.mul(clip / level)
        else:
            x = input
            y = input

        # conv
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        # bn
        ctx.need_sync = need_sync
        if not need_sync:
            x = F.batch_norm(x, bn_mean, bn_var, bn_weight, bn_bias, bn_training, average_factor, bn_eps)
            bn_scale = bn_weight * (bn_var + bn_eps).rsqrt()
            if bn_bias is not None:
                bn_bias = bn_bias - bn_mean * bn_scale
            else:
                bn_bias = - bn_mean * bn_scale
            ctx.bn_scale = bn_scale.reshape(1, -1, 1, 1).detach()
            ctx.bn_bias = bn_bias.reshape(1, -1, 1, 1).detach()
            ctx.bn_weight = bn_weight.reshape(1, -1, 1, 1).detach()
        else:
            x = SyncBatchNorm_forward(ctx, x, bn_weight, bn_bias, bn_mean, bn_var, bn_eps, average_factor, process_group, world_size)

        # save tensor
        ctx.save_for_backward(y, weight, bias, clip)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.level = level
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias, grad_bn_weight, grad_bn_bias, grad_clip = None, None, None, None, None, None

        # restore
        y, weight, bias, clip = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        level = ctx.level
        need_sync = ctx.need_sync
        if not need_sync:
            bn_scale = ctx.bn_scale
            bn_bias = ctx.bn_bias
            bn_weight = ctx.bn_weight
        else:
            bn_weight = self.bn_weight
            bn_mean = self.bn_mean
            bn_invstd = self.bn_invstd
            bn_count_all = self.bn_count_all
            bn_process_group = self.bn_process_group

        if level < 256:
            x = y.to(dtype=torch.float)
            x = x.mul(clip / level)
        else:
            x = y

        # checkpoint
        z = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        # bn
        if not need_sync:
            grad_input = grad_output * bn_scale
            if ctx.needs_input_grad[7]:
                grad_bn_weight = grad_output * (z - bn_bias).div(bn_weight)
                grad_bn_weight = grad_bn_weight.sum(dim=[0,2,3])

            if ctx.needs_input_grad[8]:
                grad_bn_bias = grad_output.sum(dim=[0,2,3])
            grad_output = grad_input
        else:
            grad_output, grad_bn_weight, grad_bn_bias = SyncBatchNorm_backward(z, bn_weight, bn_mean, bn_invstd, bn_count_all, bn_process_group, \
                    ctx.needs_input_grad[7:9], grad_output)
        z = None

        # conv
        benchmark = True
        deterministic = True
        output_mask = ctx.needs_input_grad[:2]
        grad_input, grad_weight = conv.conv2d_backward(x, grad_output, weight, padding, stride, dilation, groups, benchmark, deterministic, output_mask)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        grad_output = None
        return grad_input, grad_weight, grad_bias, None, None, None, None, \
                grad_bn_weight, grad_bn_bias, None, None, None, None, None, None, None, None, \
                grad_clip, None

class custom_conv_bn(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, memory_saving=False, args=None):
        super(custom_conv_bn, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        # constraints
        if self.padding_mode != 'zeros':
            raise NotImplementedError

        # configuration
        string = "ms.Conv2d"

        # norm
        if args is not None and hasattr(args, 'device_ids') and len(args.device_ids) > 1:
            self.norm = nn.SyncBatchNorm(out_channels)
            string = string + "-SyncBatchNorm"
        else:
            self.norm = nn.BatchNorm2d(out_channels)
            string = string + "-BatchNorm2d"

        self.memory_saving = memory_saving
        if args is not None and hasattr(args, 'keyword') and hasattr(args, 'fm_enable'):
            self.memory_saving = self.memory_saving or args.fm_enable

        # lsq
        self.fm_clip_val = nn.Parameter(torch.Tensor([1.0]))
        self.fm_intervals = 256

        if args is not None:
            if hasattr(args, 'fm_bit') and args.fm_bit is not None and args.fm_bit <= 8:
                self.fm_intervals = int(2 ** args.fm_bit)
            if hasattr(args, 'fm_level') and args.fm_level is not None and args.fm_level <= 256:
                self.fm_intervals = args.fm_level

        self.fm_intervals = self.fm_intervals - 1

        string = string + "-enable({})-FM(clip_val({})-intervals({}))".format(self.memory_saving, self.fm_clip_val.item(), self.fm_intervals)
        if self.memory_saving:
            self.string = string
        else:
            self.string = "Conv2d + BN in ms"

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.string

    def forward_bn(self, input):
        self.norm._check_input_dim(input)

        if self.norm.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.norm.momentum
            
        if self.training and self.norm.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.norm.num_batches_tracked is not None:  # type: ignore
                self.norm.num_batches_tracked = self.norm.num_batches_tracked + 1  # type: ignore
                if self.norm.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.norm.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.norm.momentum

        if self.training:
            bn_training = True
        else:
            bn_training = (self.norm.running_mean is None) and (self.norm.running_var is None)
            
        assert self.norm.running_mean is None or isinstance(self.norm.running_mean, torch.Tensor)
        assert self.norm.running_var is None or isinstance(self.norm.running_var, torch.Tensor)
        running_mean = self.norm.running_mean if not self.training or self.norm.track_running_stats else None
        running_var = self.norm.running_var if not self.training or self.norm.track_running_stats else None

        need_sync = bn_training and hasattr(self.norm, 'process_group') and hasattr(self.norm, 'ddp_gpu_size')
        process_group = None
        world_size = 1
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.norm.process_group:
                process_group = self.norm.process_group
            try:
                world_size = torch.distributed.get_world_size(process_group)
            except AssertionError:
                world_size = 1
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if need_sync:
            if not input.is_cuda:
                raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

            if not self.norm.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')
            assert bn_training
            
        return exponential_average_factor, bn_training, running_mean, running_var, need_sync, process_group, world_size

    def forward(self, x):
        if self.memory_saving:
            average_factor, bn_training, running_mean, running_var, need_sync, process_group, world_size = self.forward_bn(x)
            y = conv2d_bn.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,\
                    self.norm.weight, self.norm.bias, average_factor, bn_training, need_sync, process_group, world_size, running_mean, running_var, self.norm.eps,\
                    self.fm_clip_val.abs(), self.fm_intervals)
        else:
            x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            y = self.norm(x)

        return y

class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inplace=False):
        y = x < 0
        ctx.save_for_backward(y)
        #x = x.clamp(min=0)  # faster
        if inplace:
            x[y] = 0
        else:
            x = torch.where(y, torch.zeros_like(x), x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        #grad_output[y] = 0 # more memory ?
        grad_output = torch.where(y, torch.zeros_like(grad_output), grad_output)
        return grad_output, None

class custom_relu(nn.ReLU):
    def __init__(self, inplace=True):
        super(custom_relu, self).__init__(inplace)
        self.inplace = inplace

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "ms.ReLU(False)"

    def forward(self, x):
        y = relu.apply(x)
        return y

