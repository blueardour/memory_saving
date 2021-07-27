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
import pydevd


class Quant(object):
    def __init__(self, memory_saving=False, args=None, logger=None, enable=False, tag='fm'):
        assert isinstance(self, nn.Module)

        self.enable = memory_saving or enable
        self.fp_forward = False
        # quantizer
        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.clip_val = nn.Parameter(torch.Tensor([1.0]))
        self.level = 0
        self.non_negative_only = False
        self.tag = tag
        self.index = -1
        self.args = args
        self.string = 'ms.'
        self.repr = super(type(self), self).__repr__()
        self.logger = logger
        self.stable = -1
        self.correlate = 1.0
        self.warmup_choice = 'MA'  # or 'EMA'
        self.ema_decay = 0.9999
        self.requires_grad = False

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
            if hasattr(args, 'fm_boundary') and args.fm_boundary is not None:
                self.clip_val.fill_(args.fm_boundary)
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
                "index({})-clip_val({})-level({})-stable({})-correlate({})-non_negative_only({})-requires_grad({})".format(
                    self.index, self.clip_val.item(), self.level, self.stable, self.correlate, self.non_negative_only,
                    self.requires_grad))
        self.items = ['clip_val', 'level', 'stable', 'correlate', 'non_negative_only', 'warmup_choice', 'ema_decay',
                      'requires_grad']
        self.clip_val.requires_grad = self.enable and self.requires_grad

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

    def init_based_on_warmup(self, data=None):
        if (not self.enable and data is None) or not self.training:
            return

        iteration = self.iteration.item()
        with torch.no_grad():
            if hasattr(self, 'clip_val') and isinstance(self.clip_val, torch.Tensor):
                max_value = data.abs().max().item()
                if self.correlate > 0:
                    max_value = max_value * self.correlate

                if self.warmup_choice == 'MA':
                    self.clip_val.data = max_value + iteration * self.clip_val.data
                    self.clip_val.div_(iteration + 1)
                elif self.warmup_choice == 'EMA':
                    if iteration == 0:
                        self.clip_val.fill_(data.abs().max())
                    else:
                        self.clip_val.sub_((1 - self.ema_decay) * (self.clip_val - data.abs().max()))

                if iteration == (self.stable - 1):
                    self.verbose('update %s clip_val for index %d to %r' % (self.tag, self.index, self.clip_val.item()))
        self.iteration.add_(1)

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
    def forward(ctx, x, training, fp_forward, clip_val, level, non_negative_only, iteration, ema_decay, identifier="_"):
        def update_clip_val(input, clip_val, iteration, ema_decay, reduce_fn=lambda x: x.abs().max()):
            if iteration == 0:
                clip_val.fill_(reduce_fn(input))
            else:
                clip_val.sub_((1 - ema_decay) * (clip_val - reduce_fn(input)))
            iteration.add_(1)

        def save_for_backward(y, signed=True):
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

        if level == 0:
            y = x
            if training:
                setattr(ctx, 'input{}'.format(identifier), y)
        else:
            if training and not clip_val.requires_grad:
                update_clip_val(x.detach(), clip_val, iteration, ema_decay)

            clip_val = clip_val.abs().to(dtype=x.dtype)
            if non_negative_only:
                y = x / clip_val * (level - 1)
                y = torch.round(y)
                y = torch.clamp(y, min=0, max=level - 1)
                if training:
                    save_for_backward(y, signed=False)
                y = y / (level - 1) * clip_val
                # is_filtered = x >= clip_val
                max_clipped = x >= clip_val
            else:
                y = x / clip_val * (level // 2)
                y = torch.round(y)
                y = torch.clamp(y, min=-level // 2, max=level - level // 2 - 1)
                if training:
                    save_for_backward(y, signed=True)
                y = y / (level // 2) * clip_val
                # is_filtered = (x >= (clip_val * (level-level//2-1) / (level//2))) | (x <= -clip_val)
                max_clipped = x >= (clip_val * (level - level // 2 - 1) / (level // 2))
                min_clipped = x <= -clip_val

            if training:
                # setattr(ctx, 'is_filtered{}'.format(identifier), packbit.packbits_padded(is_filtered, dim=0))
                setattr(ctx, 'max_clipped{}'.format(identifier), packbit.packbits_padded(max_clipped, dim=0))
                if not non_negative_only:
                    setattr(ctx, 'min_clipped{}'.format(identifier), packbit.packbits_padded(min_clipped, dim=0))
                setattr(ctx, 'clip_val{}'.format(identifier), clip_val)


        if training:
            setattr(ctx, 'level{}'.format(identifier), level)
            setattr(ctx, 'non_negative_only{}'.format(identifier), non_negative_only)
        return x if fp_forward else y

    @staticmethod
    def restore(ctx, identifier="_"):
        def saved_tensors(input, level, dtype):
            if level in [65536, 256]:
                output = input.to(dtype=dtype)
            elif level == 16:
                output = packbit.unpackbits_padded(input, dim=0, mask=3).to(dtype=dtype)
            elif level == 4:
                output = packbit.unpackbits_padded(input, dim=0, mask=15).to(dtype=dtype)
            else:
                raise RuntimeError("un-supported quanitzation bit: {level}")
            return output

        input = getattr(ctx, 'input{}'.format(identifier))
        level = getattr(ctx, 'level{}'.format(identifier))
        non_negative_only = getattr(ctx, 'non_negative_only{}'.format(identifier))
        setattr(ctx, 'input{}'.format(identifier), None)
        if level == 0:
            y = input
        else:
            clip_val = getattr(ctx, 'clip_val{}'.format(identifier))
            if non_negative_only:
                y = saved_tensors(input, level, dtype=clip_val.dtype) / (level - 1) * clip_val
            else:
                y = saved_tensors(input, level, dtype=clip_val.dtype) / (level // 2) * clip_val
        return y

    @staticmethod
    def backward(ctx, grad_input, identifier="_"):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        level = getattr(ctx, 'level{}'.format(identifier))
        if level == 0:
            grad_clip = None
        else:
            clip_val = getattr(ctx, 'clip_val{}'.format(identifier))

            non_negative_only = getattr(ctx, 'non_negative_only{}'.format(identifier))

            if non_negative_only:
                max_clipped = getattr(ctx, 'max_clipped{}'.format(identifier))
                max_clipped = packbit.unpackbits_padded(max_clipped, dim=0).to(dtype=torch.bool)
                grad_clip = grad_input.masked_select(max_clipped).sum().reshape(clip_val.shape)
                grad_input.masked_fill_(max_clipped, 0.0)
                setattr(ctx, 'max_clipped{}'.format(identifier), None)
            else:
                max_clipped = getattr(ctx, 'max_clipped{}'.format(identifier))
                max_clipped = packbit.unpackbits_padded(max_clipped, dim=0).to(dtype=torch.bool)

                min_clipped = getattr(ctx, 'min_clipped{}'.format(identifier))
                min_clipped = packbit.unpackbits_padded(min_clipped, dim=0).to(dtype=torch.bool)

                grad_clip = grad_input.masked_select(max_clipped).sum() - grad_input.masked_select(min_clipped).sum()
                grad_clip = grad_clip.reshape(clip_val.shape)

                grad_input.masked_fill_(max_clipped, 0.0)
                grad_input.masked_fill_(min_clipped, 0.0)

                setattr(ctx, 'max_clipped{}'.format(identifier), None)
                setattr(ctx, 'min_clipped{}'.format(identifier), None)

            setattr(ctx, 'clip_val{}'.format(identifier), None)
        return grad_clip


class quantization(nn.Module, Quant):
    def __init__(self, memory_saving=False, args=None, logger=None, tag='fm'):
        super(quantization, self).__init__()
        Quant.__init__(self, memory_saving=memory_saving, args=args, logger=logger, tag=tag)

    def __repr__(self):
        return self.__str__()

