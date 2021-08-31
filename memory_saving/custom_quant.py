import math

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
    # from .clip import find_clip_aciq, find_clip_entropy, find_clip_mmse
    from .cpp_extension import quantization as ext_quant
import pydevd

def pack_group(x, groups):
    input_shape = x.shape
    if len(input_shape) == 3:
        B, N, C = input_shape
        x = x.reshape(B, N, groups, C // groups).permute(0, 1, 3, 2).reshape(-1, groups)
    elif len(input_shape) == 2:
        B, C = input_shape
        x = x.reshape(B, groups, C // groups).permute(0, 2, 1).reshape(-1, groups)
    else:
        assert len(input_shape) == 4
        B, H, N, D = input_shape
        if groups != H:
            assert groups == 1
            x = x.reshape(-1, 1)
        else:
            x = x.permute(0, 2, 3, 1).reshape(-1, groups)
    return x

def depack_group(x, groups, input_shape):
    if len(input_shape) == 3:
        B, N, C = input_shape
        x = x.reshape(B, N, C // groups, groups).permute(0, 1, 3, 2).reshape(B, N, C)
    elif len(input_shape) == 2:
        B, C = input_shape
        x = x.reshape(B, C // groups, groups).permute(0, 2, 1).reshape(B, C)
    else:
        B, H, N, D = input_shape
        if groups != H:
            assert groups == 1
            x = x.reshape(B, H, N, D)
        else:
            x = x.reshape(B, N, D, groups).permute(0, 3, 1, 2)
    return x

def update_clip_val_shift(input, clip_val, shift, iteration, ema_decay):
    max_value, _ = torch.max(input, dim=0)
    min_value, _ = torch.min(input, dim=0)
    clip_range = max_value - min_value
    if iteration == 0:
        clip_val.data = clip_range
        shift.data = min_value
    else:
        clip_val.sub_((1 - ema_decay) * (clip_val - clip_range))
        shift.sub_((1 - ema_decay) * (shift - min_value))
    iteration.add_(1)


class Quant(object):
    def __init__(self, memory_saving=False, args=None, logger=None, enable=False, tag='fm', groups=1):
        assert isinstance(self, nn.Module)

        self.enable = memory_saving or enable
        # self.fp_forward = True
        # quantizer
        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.groups = groups
        self.clip_val = nn.Parameter(torch.Tensor([1.0] * groups))
        self.level = 256
        # self.non_negative_only = False
        self.tag = tag
        self.index = -1
        self.args = args
        self.string = 'ms.'
        self.repr = super(type(self), self).__repr__()
        self.logger = logger
        # self.stable = -1
        # self.correlate = 1.0
        # self.warmup_choice = 'MA'  # or 'EMA'
        self.ema_decay = 0.9
        self.requires_grad = False
        # self.init_choice = 'mse'  # or 'entropy', 'mse'
        # self.init_phase = False
        # self.stochastic_round = False
        self.shift = nn.Parameter(torch.Tensor([0.] * groups))

        class logger_wrapper(object):
            def info(self, string):
                print(string)

        if logger is None:
            if hasattr(args, 'logger'):
                self.logger = args.logger
            else:
                if args is None:
                    self.logger = logger_wrapper()
                else:
                    logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
                    self.logger = logging.getLogger(logger_root + __name__)
        self.verbose = self.logger.info

        if args is not None:
            if hasattr(args, 'fm_bit') and args.fm_bit is not None:
                self.level = int(2 ** args.fm_bit)
            if hasattr(args, 'fm_level') and args.fm_level is not None:
                self.level = args.fm_level
            # if hasattr(args, 'fm_boundary') and args.fm_boundary is not None:
            #     self.clip_val.fill_(args.fm_boundary)
            if hasattr(args, 'fm_stable'):
                self.stable = args.fm_stable
            if hasattr(args, 'fm_correlate'):
                self.correlate = args.fm_correlate
            if hasattr(args, 'fm_enable'):
                self.enable = self.enable or args.fm_enable
            if hasattr(args, 'fm_nno'):
                self.non_negative_only = self.non_negative_only and args.nno
            if hasattr(args, 'fm_half_range'):
                self.non_negative_only = self.non_negative_only and args.fm_half_range
            # if self.level > 256:
            #    self.level = 257

            self.verbose(
                "index({})-clip_val({})-level({})-non_negative_only({})-groups({})-stochastic_round({})-({})".format(
                    self.index, self.clip_val.tolist(), self.level, self.non_negative_only,
                    self.groups, self.stochastic_round, self.shift.tolist()))
        self.items = ['clip_val', 'level', 'stable', 'ema_decay',
                      'requires_grad', 'groups']
        self.clip_val.requires_grad = self.enable and self.requires_grad
        self.shift.requires_grad = self.enable and self.requires_grad

    def __str__(self):
        if hasattr(self, 'repr'):
            string = self.repr
        if hasattr(self, 'string'):
            string = self.string + string
        string = string + "-index({})-tag({})".format(self.index, self.tag)

        if self.enable:
            for item in self.items:
                if hasattr(self, item):
                    value = getattr(self, item)
                    if isinstance(value, torch.Tensor):
                        if value.numel() == 1:
                            value = value.item()
                        else:
                            continue
                    string = string + "-{}({})".format(item, value)

        if hasattr(self, 'norm'):
            string += "\n\t-" + str(self.norm)

        return string

    def update_quantization_parameter(self, **parameters):
        feedback = dict()
        index = self.index
        if 'index' in parameters:
            if isinstance(parameters['index'], int):
                index = parameters['index']
            elif isinstance(parameters['index'], dict) and self.tag in parameters['index']:
                index = parameters['index'][self.tag]
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.logger.warning('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                    for k, v in list(parameters.items()):
                        if hasattr(self, "{}".format(k)):
                            if isinstance(getattr(self, k), bool):
                                v = False if v in ['False', 'false', False] else True
                            elif isinstance(getattr(self, k), int):
                                v = int(v)
                            elif isinstance(getattr(self, k), float):
                                v = float(v)
                            elif isinstance(getattr(self, k), str):
                                v = v.replace("'", "").replace('"', '')
                                if 'same' in v:
                                    v = v.replace('same', str(self.index))
                                elif "last" in v:
                                    v = v.replace('last', str(self.index - 1))
                            if isinstance(getattr(self, k), torch.Tensor):
                                with torch.no_grad():
                                    if getattr(self, 'progressive', False):
                                        if 'lsq' in self.args.keyword or '{}_lsq'.format(self.tag) in self.args.keyword:
                                            if k in ['level_num']:
                                                # if hasattr(self, 'clip_val'):
                                                v = float(v)
                                                # if negative number provide, it indicates decreasing on current
                                                v = v if v > 0 else self.level_num.item() + v

                                                assert v > 1.9, "level_num should be at least 2"
                                                scale = (v - 1) / (self.level_num.item() - 1)
                                                self.clip_val.mul_(scale)
                                                self.verbose('update {}_clip_val to {} for index {}'.format(
                                                    self.tag, self.clip_val, self.index))

                                                # remember to patch the momentum in SGD optimizer. set it to zero or multiple by scale
                                                if 'reset_momentum_list' in feedback:
                                                    feedback['reset_momentum_list'].append(self.clip_val)
                                                else:
                                                    feedback['reset_momentum_list'] = [self.clip_val]
                                    getattr(self, k).fill_(float(v))
                                self.verbose('update {}_{} to {} for index {}'.format(
                                    self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                            else:
                                setattr(self, "{}".format(k), v)
                                self.verbose('update {}_{} to {} for index {}'.format(
                                    self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                            if self.enable:
                                assert hasattr(self, 'iteration'), \
                                    "cannot enable quantization for current layer. Likely an error in policy file"
                        # global_buffer
                        if k in ['global_buffer'] and hasattr(self.args, 'global_buffer'):
                            v = str(v)
                            if isinstance(getattr(self.args, k, None), dict) and hasattr(self, v) and self.enable:
                                key = "{}-{}-{}".format(v, self.index, self.tag)
                                self.args.global_buffer[key] = getattr(self, v)
                                self.verbose('update global_buffer (current length: {}), key: {}'.format(
                                    len(self.args.global_buffer), key))

        self.clip_val.requires_grad = self.enable and self.requires_grad
        self.shift.requires_grad = self.enable and self.requires_grad
        if not self.enable:
            return None
        else:
            if hasattr(self, 'quant_loss_function') and isinstance(self.quant_loss_function, str):
                qlf = self.quant_loss_function.split()
                quant_loss_function = []
                for loss_method in qlf:
                    if loss_method == 'L2':
                        quant_loss_function.append(nn.MSELoss())
                    elif loss_method == 'L1':
                        quant_loss_function.append(nn.L1Loss())
                if len(quant_loss_function) != 0:
                    self.quant_loss_function = quant_loss_function
                    self.verbose('update quant_loss_function: {} for layer(index:{}, tag:{})'.format(
                        self.quant_loss_function, self.index, self.tag))
                else:
                    self.quant_loss_function = 'none'
            if hasattr(self, 'method'):
                assert self.method != 'none', "quantization enable but without specific method in layer(index:{}, tag:{})".format(
                    self.index, self.tag)
            return feedback

    @staticmethod
    def forward(ctx, x, clip_val, level, iteration, ema_decay, groups, shift, identifier="_"):

        input_shape = x.shape
        y = pack_group(x, groups)
        quant_shape = y.shape

        update_clip_val_shift(y.detach(), clip_val, shift, iteration, ema_decay)
        setattr(ctx, 'clip_val{}'.format(identifier), clip_val)
        setattr(ctx, 'shift{}'.format(identifier), shift)

        scale = ((level - 1) / clip_val.abs()).to(dtype=x.dtype)
        shift = shift.to(dtype=x.dtype)
        group_size = input_shape[0] if input_shape[0] <= 512 else 512
        y = ext_quant.pack_single_precision(y, scale, shift, 8, True, group_size)
        setattr(ctx, 'input_type{}'.format(identifier), x.dtype)
        setattr(ctx, 'quant_shape{}'.format(identifier), quant_shape)
        setattr(ctx, 'input{}'.format(identifier), y)
        # noise = y.new(y.shape).uniform_(-0.5, 0.5)
        # y = (y - shift) / clip_val * (level - 1)
        # y = torch.round(y + noise)
        # y = torch.clamp(y, min=0, max=level - 1)


        # setattr(ctx, 'input{}'.format(identifier), y.to(torch.uint8))
        setattr(ctx, 'input_shape{}'.format(identifier), input_shape)
        setattr(ctx, 'level{}'.format(identifier), level)

    @staticmethod
    def restore(ctx, identifier="_"):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        input = getattr(ctx, 'input{}'.format(identifier))
        level = getattr(ctx, 'level{}'.format(identifier))
        input_shape = getattr(ctx, 'input_shape{}'.format(identifier))
        clip_val = getattr(ctx, 'clip_val{}'.format(identifier))
        shift = getattr(ctx, 'shift{}'.format(identifier))

        quant_shape = getattr(ctx, 'quant_shape{}'.format(identifier))
        input_type = getattr(ctx, 'input_type{}'.format(identifier))
        scale = ((level - 1) / clip_val.abs()).to(dtype=input_type)
        shift = shift.to(dtype=input_type)
        group_size = input_shape[0] if input_shape[0] <= 512 else 512
        y = ext_quant.unpack_single_precision(input, 8, scale, shift, quant_shape[0], quant_shape[1], group_size)
        y = depack_group(y, clip_val.size()[0], input_shape)
        setattr(ctx, 'quant_shape{}'.format(identifier), None)
        setattr(ctx, 'input_type{}'.format(identifier), None)

        # y = input.to(dtype=clip_val.dtype) / (level - 1) * clip_val + shift
        # y = depack_group(y, clip_val.size()[0], input_shape)

        setattr(ctx, 'input{}'.format(identifier), None)
        setattr(ctx, 'clip_val{}'.format(identifier), None)
        setattr(ctx, 'shift{}'.format(identifier), None)
        setattr(ctx, 'input_shape{}'.format(identifier), None)
        setattr(ctx, 'level{}'.format(identifier), None)

        return y

    @staticmethod
    def backward(ctx, grad_input, identifier="_"):
        assert 1 == 0, 'temporarily stop supporting gradient update for clip value'
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        level = getattr(ctx, 'level{}'.format(identifier))
        if level == 0:
            grad_clip = None
        else:
            clip_val = getattr(ctx, 'clip_val{}'.format(identifier))
            input_shape = getattr(ctx, 'input_shape{}'.format(identifier))
            non_negative_only = getattr(ctx, 'non_negative_only{}'.format(identifier))
            groups = clip_val.size()[0]

            grad_input = pack_group(grad_input, clip_val.size()[0])
            grad_clip = torch.zeros(groups).to(clip_val.device)

            if non_negative_only:
                max_clipped = getattr(ctx, 'max_clipped{}'.format(identifier))
                max_clipped = packbit.unpackbits_padded(max_clipped, dim=0).to(dtype=torch.bool)
                for i in range(groups):
                    grad_clip[i] = grad_input[:, i].masked_select(max_clipped[:, i]).sum()

                grad_input.masked_fill_(max_clipped, 0.0)
                setattr(ctx, 'max_clipped{}'.format(identifier), None)
            else:
                max_clipped = getattr(ctx, 'max_clipped{}'.format(identifier))
                max_clipped = packbit.unpackbits_padded(max_clipped, dim=0).to(dtype=torch.bool)

                min_clipped = getattr(ctx, 'min_clipped{}'.format(identifier))
                min_clipped = packbit.unpackbits_padded(min_clipped, dim=0).to(dtype=torch.bool)

                for i in range(groups):
                    grad_clip[i] = grad_input[:, i].masked_select(max_clipped[:, i]).sum() - grad_input[:, i].masked_select(min_clipped[:, i]).sum()

                grad_input.masked_fill_(max_clipped, 0.0)
                grad_input.masked_fill_(min_clipped, 0.0)

                setattr(ctx, 'max_clipped{}'.format(identifier), None)
                setattr(ctx, 'min_clipped{}'.format(identifier), None)

            setattr(ctx, 'clip_val{}'.format(identifier), None)
            setattr(ctx, 'non_negative_only{}'.format(identifier), None)
            setattr(ctx, 'level{}'.format(identifier), None)

            grad_input = depack_group(grad_input, groups, input_shape)

        return grad_clip


class quantization(nn.Module, Quant):
    def __init__(self, memory_saving=False, args=None, logger=None, tag='fm', groups=None):
        super(quantization, self).__init__()
        Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger, tag=tag, groups=groups)

    def __repr__(self):
        return self.__str__()

