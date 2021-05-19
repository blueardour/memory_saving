

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

class Quant(object):
    def __init__(self, memory_saving=False, args=None, logger=None):
        self.memory_saving = memory_saving

        # quantizer
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
        self.string = 'ms.'
        self.repr = super(type(self), self).__repr__()

        if logger is None:
            if hasattr(args, 'logger'):
                self.logger = args.logger
            else:
                logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
                self.logger = logging.getLogger(logger_root + __name__)

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
        if hasattr(self, 'repr'):
            string = self.repr

        if self.memory_saving:
            if hasattr(self, 'string'):
                string = self.string + string
            string = string + "-clip_val({})-level({})-stable({})-correlate({})-non_negative_only({})".format(
                self.clip_val.item(), self.level, self.stable, self.correlate, self.non_negative_only)

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

