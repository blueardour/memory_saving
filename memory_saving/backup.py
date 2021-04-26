

class bn_relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bn_weight, bn_bias, average_factor, bn_training, need_sync, process_group, world_size, bn_mean, bn_var, bn_eps):
        output = batchnorm2d.forward(ctx, input, bn_weight, bn_bias, average_factor, bn_training, need_sync, process_group, world_size, bn_mean, bn_var, bn_eps)
        is_filtered = output < 0
        output = torch.where(is_filtered, 0, output) 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #input, weight, running_mean, running_var, save_mean, save_var, reverse = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[is_filtered] = 0
        grad_input, grad_weight, grad_bias = batchnorm2d.backward(grad_input)

class custom_bn_relu(nn.SyncBatchNorm):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None, inplace=False, memory_saving=False):
        super(custom_bn_relu, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, process_group=process_group)
        self.memory_saving = memory_saving
        self.inplace = inplace

    def forward(self, x):
        if self.memory_saving:
            average_factor, bn_training, running_mean, running_var, need_sync, process_group, world_size = bn_pre_forward(x)
            x = bn_relu.apply(x, self.weight, self.bias, average_factor, bn_training, need_sync, process_group, world_size, running_mean, running_var, self.eps)
        else:
            x = super(custom_bn_relu, self).forward(x)
            x = F.relu(x, inplace=self.inplace)
        return x

# Uniform Quantization based Convolution
class conv2d_uniform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, clip=None, level=255, non_negative_only=True):
        # quant
        index = None
        if level < 256:
            if isinstance(clip, torch.Tensor) and clip.dtype != x.dtype:
                clip = clip.half()
            if non_negative_only:
                index = x.ge(clip.item())
                x = torch.where(index, clip, x)
                scale = clip / (float)level
                x.div_(scale)
                y = torch.round(x)
                x = x.mul_(scale)
                if level == 255:
                    y = y.to(dtype=torch.uint8)
                else:
                    raise NotImplementedError
            else:
                #index = x.abs().ge(clip.item())
                #x = torch.where(index, clip * x.sign(), x)
                #x.div_(clip)
                #y = torch.round(x)
                #x = x.mul_(scale)
                raise NotImplementedError
        else:
            y = x

        # conv
        x = F.conv2d(x, weight, bias, stride, padding, dilation, groups)

        # save tensor
        ctx.save_for_backward(y, weight, bias, clip, index)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.level = level













class relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inplace=False, dim=1):
        y = x < 0
        #x = x.clamp(min=0)  # faster
        if inplace:
            x[y] = 0
        else:
            x = torch.where(y, torch.zeros_like(x), x)

        z = packbits_padded(y, dim=dim) 
        #y1 = unpackbits_padded(z)
        #print(y.sum(), y.numel(), z.dtype, z.numel())
        #print((y == y1).sum())
        ctx.save_for_backward(z)
        ctx.dim = dim
        return x

    @staticmethod
    def backward(ctx, grad_output):
        z, = ctx.saved_tensors
        y = unpackbits_padded(z, dim=ctx.dim)
        #grad_output[y] = 0 # more memory ?
        grad_output = torch.where(y, torch.zeros_like(grad_output), grad_output)
        return grad_output, None, None

class custom_relu(nn.ReLU):
    def __init__(self, inplace=True, dim=1, memory_saving=False):
        super(custom_relu, self).__init__(inplace)
        self.inplace = inplace
        self.dim = dim
        self.memory_saving = memory_saving

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.memory_saving:
            return "ms.ReLU(inplace = Forcefully False, dim={})".format(self.dim)
        else:
            return 'nn.ReLU'

    def forward(self, x):
        if self.memory_saving:
            y = relu.apply(x)
        else:
            y = F.relu(x, inplace=self.inplace)
        return y

def test_relu():
    model = custom_relu()
    x = torch.rand(512,64,56,56)
    x = x - x.mean()
    x = x.cuda()
    y = model(x)


