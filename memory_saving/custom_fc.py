
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import logging

from . import packbit

class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, \
            memory_saving=False, args=None, logger=None, force_fp=False):
        super(Linear, self).__init__(in_features, out_features, bias=bias)

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
                y = F.conv2d(x, self.weight, self.bias)
            else:
                y = conv2d_uniform.apply(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, \
                        self.clip_val.abs(), self.level, self.non_negative_only, self.training)
        else:
            y = F.linear(x, self.weight, self.bias)
        self.iteration.add_(1)
        return y

