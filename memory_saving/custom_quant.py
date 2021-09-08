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
    # from .cpp_extension import quantization as ext_quant
import pydevd


def save_for_backward(ctx, y, level, identifier, signed=True):
    if level == 65536 and signed:
        setattr(ctx, 'input{}'.format(identifier), y.to(torch.int16))
    elif level == 256 and signed:
        setattr(ctx, 'input{}'.format(identifier), y.to(torch.int8))
    elif level == 256 and not signed:
        setattr(ctx, 'input{}'.format(identifier), y.to(torch.uint8))
    elif level == 16:  # not verified
        setattr(ctx, 'input{}'.format(identifier), packbit.packbits_padded(y, dim=0, mask=3))
    elif level == 4:  # not verified
        setattr(ctx, 'input{}'.format(identifier), packbit.packbits_padded(y, dim=0, mask=15))
    else:
        raise RuntimeError("un-supported quanitzation bit: {level}")

def update_clip_val_shift(input, clip_val, shift, iteration, ema_decay):
    reduce_dim = tuple(range(len(input.size()) - 1))
    max_value = torch.amax(input, reduce_dim)
    min_value = torch.amin(input, reduce_dim)
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
        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.groups = groups
        self.clip_val = nn.Parameter(torch.Tensor([1.0] * groups), requires_grad=False)
        self.level = 0
        self.tag = tag
        self.index = -1
        self.args = args
        self.string = 'ms.'
        self.repr = super(type(self), self).__repr__()
        self.logger = logger
        self.ema_decay = 0.9
        self.requires_grad = False
        self.shift = nn.Parameter(torch.Tensor([0.] * groups), requires_grad=False)

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
                "index({})-level({})-groups({})".format(
                    self.index, self.level,
                    self.groups, ))
        self.items = ['clip_val', 'level',  'ema_decay', 'groups', 'shift']
        # self.clip_val.requires_grad = self.enable and self.requires_grad
        # self.shift.requires_grad = self.enable and self.requires_grad

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

        # self.clip_val.requires_grad = self.enable and self.requires_grad
        # self.shift.requires_grad = self.enable and self.requires_grad
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
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)

        update_clip_val_shift(x, clip_val, shift, iteration, ema_decay)
        setattr(ctx, 'clip_val{}'.format(identifier), clip_val)
        setattr(ctx, 'shift{}'.format(identifier), shift)

        noise = x.new(x.shape).uniform_(-0.5, 0.5)
        x = (x - shift) / clip_val * (level - 1)
        x = torch.round(x + noise)
        x = torch.clamp(x, min=0, max=level - 1)
        setattr(ctx, 'input{}'.format(identifier), x.to(torch.uint8))
        setattr(ctx, 'input_shape{}'.format(identifier), x.shape)
        setattr(ctx, 'level{}'.format(identifier), level)

    @staticmethod
    def restore(ctx, identifier="_"):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        # def saved_tensors(input, level, dtype):
        #     if level in [65536, 256]:
        #         output = input.to(dtype=dtype)
        #     elif level == 16:
        #         output = packbit.unpackbits_padded(input, dim=0, mask=3).to(dtype=dtype)
        #     elif level == 4:
        #         output = packbit.unpackbits_padded(input, dim=0, mask=15).to(dtype=dtype)
        #     else:
        #         raise RuntimeError("un-supported quanitzation bit: {level}")
        #     return output

        y = getattr(ctx, 'input{}'.format(identifier))
        level = getattr(ctx, 'level{}'.format(identifier))
        input_shape = getattr(ctx, 'input_shape{}'.format(identifier))
        clip_val = getattr(ctx, 'clip_val{}'.format(identifier))
        shift = getattr(ctx, 'shift{}'.format(identifier))

        y = y.to(dtype=clip_val.dtype) / (level - 1) * clip_val + shift
        if len(input_shape) == 4:
            y = y.permute(0, 3, 1, 2)

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

