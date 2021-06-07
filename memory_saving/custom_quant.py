
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

if 'memory_saving' not in __name__:
    import packbit
    import native
else:
    from . import packbit
    from . import native

class Quant(object):
    def __init__(self, memory_saving=False, args=None, logger=None):
        assert isinstance(self, nn.Module)

        self.memory_saving = memory_saving
        self.fp_forward = False
        # quantizer
        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.clip_val = nn.Parameter(torch.Tensor([1.0]))
        self.level = 257
        self.stable = -1
        self.correlate = 1.0
        self.non_negative_only = False
        self.tag = 'fm'
        self.index = -1
        self.logger = logger
        self.args = args
        self.string = 'ms.'
        self.repr = super(type(self), self).__repr__()

        if logger is None:
            if hasattr(args, 'logger'):
                self.logger = args.logger
            else:
                logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
                self.logger = logging.getLogger(logger_root + __name__)

        if args is not None:
            if hasattr(args, 'fm_bit') and args.fm_bit is not None:
                self.level = int(2 ** args.fm_bit)
            if hasattr(args, 'fm_level') and args.fm_level is not None:
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
            if self.level > 256:
                self.level = 257
            
            self.logger.info("index({})-clip_val({})-level({})-stable({})-correlate({})-non_negative_only({})".format(
                self.index, self.clip_val.item(), self.level, self.stable, self.correlate, self.non_negative_only))

    def __str__(self):
        if hasattr(self, 'repr'):
            string = self.repr

        if self.memory_saving:
            if hasattr(self, 'string'):
                string = self.string + string
            string = string + "-clip_val({})-level({})-stable({})-correlate({})-non_negative_only({})-fp_forward({})".format(
                self.clip_val.item(), self.level, self.stable, self.correlate, self.non_negative_only, self.fp_forward)

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

    @staticmethod
    def forward(ctx, x, training, fp_forward, clip_val, level, non_negative_only, identifier="_"):
        if level > 256:
            y = x
            if training:
                setattr(ctx, 'input{}'.format(identifier), y)
        else:
            if non_negative_only:
                y = x / clip_val * (level-1)
                y = torch.round(y)
                y = torch.clamp(y, min=0, max=level-1)
                if training:
                    setattr(ctx, 'input{}'.format(identifier), y.to(torch.int8))
                y = y / (level-1) * clip_val
                is_filtered = x >= clip_val
            else:
                y = x / clip_val * (level//2)
                y = torch.round(y)
                y = torch.clamp(y, min=-level//2, max=level-level//2-1)
                if training:
                    setattr(ctx, 'input{}'.format(identifier), y.to(torch.int8))
                y = y / (level//2) * clip_val
                is_filtered = (x >= (clip_val * (level-level//2-1) / (level//2))) | (x <= -clip_val)
            if training:
                setattr(ctx, 'is_filtered{}'.format(identifier), packbit.packbits_padded(is_filtered, dim=1))
                setattr(ctx, 'clip_val{}'.format(identifier), clip_val)

        if training:
            setattr(ctx, 'level{}'.format(identifier), level)
            setattr(ctx, 'non_negative_only{}'.format(identifier), non_negative_only)
        return x if fp_forward else y

    @staticmethod
    def restore(ctx, identifier="_"):
        input = getattr(ctx, 'input{}'.format(identifier))
        level = getattr(ctx, 'level{}'.format(identifier))
        non_negative_only = getattr(ctx, 'non_negative_only{}'.format(identifier))
        if level > 256:
            y = input
        else:
            clip_val = getattr(ctx, 'clip_val{}'.format(identifier))
            if non_negative_only:
                y = input.to(torch.float) / (level- 1) * clip_val
            else:
                y = input.to(torch.float) / (level//2) * clip_val
        return y

    @staticmethod
    def backward(ctx, grad_input, identifier="_"):
        level = getattr(ctx, 'level{}'.format(identifier))
        if level > 256:
            grad_clip = None
        else:
            clip_val = getattr(ctx, 'clip_val{}'.format(identifier))
            is_filtered = getattr(ctx, 'is_filtered{}'.format(identifier))
            is_filtered = packbit.unpackbits_padded(is_filtered, dim=1).to(dtype=torch.bool)
            grad_clip = grad_input.masked_select(is_filtered).sum().reshape(clip_val.shape)
            grad_input.masked_fill_(not is_filtered, 0.0)
            setattr(ctx, 'is_filtered{}'.format(identifier), None)
            setattr(ctx, 'clip_val{}'.format(identifier), None)
            
        setattr(ctx, 'input{}'.format(identifier), None)
        return grad_clip

class quantization(nn.Module, Quant):
    def __init__(self, memory_saving=False, args=None, logger=None):
        super(quantization, self).__init__()
        Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger)

    def __repr__(self):
        return self.__str__()

