import memory_saving.cpp_extension.quantization as ext_quant
import time
import torch
import numpy as np

def check_quant():
    diff = 0.


    for i in range(200):
        data = torch.randn(128*197*197, 3).cuda()
        # temp = data.permute(0, 2, 1).reshape(-1, 3)

        mins, _ = data.min(0)
        maxs, _ = data.max(0)

        level = 255
        scale = maxs - mins + 2e-6
        cuda_scales = 255 / scale

        N, num_groups = data.shape

        output = ext_quant.pack_single_precision(data, cuda_scales, mins, 8, True, 128)
        dequant_out = ext_quant.unpack_single_precision(output, 8, cuda_scales, mins, N, num_groups,
                                                        128)
        cuda_quant_error = (dequant_out - data).norm()

        y = (data - mins) / scale * (level - 1)
        y = torch.round(y)
        y = torch.clamp(y, min=0, max=level - 1)
        torch_dequant_out = y / (level - 1) * scale + mins

        torch_quant_error = (torch_dequant_out - data).norm()
        cur_error = (cuda_quant_error - torch_quant_error).item()
        print(cur_error)
        diff += cur_error
    print('diff = ', diff/200)


def check_correctness():
    len1 = 128 * 197 * 197
    len2 = 128 * 64 * 197
    diff = 0.
    for i in range(100):
        data = torch.randn(128 * 197, 3, 197).cuda()
        temp = data.permute(0, 2, 1).reshape(-1, 3)

        mins, _ = temp.min(0)
        maxs, _ = temp.max(0)

        level = 255
        scale = maxs - mins + 2e-6

        v1_mins = mins[None, :, None].expand(128 * 197, 3, 1).contiguous()
        v1_maxs = maxs[None, :, None].expand(128 * 197, 3, 1).contiguous()

        # cuda_scales = 255 / (v1_maxs - v1_mins)
        cuda_scales = 255 / scale

        forward = 0
        N, num_groups, group_size = data.shape
        num_features = int(np.prod(data.shape[1:]))
        num_features = (num_features + (group_size - num_features % group_size) % group_size)

        output = ext_quant.pack_single_precision(data, cuda_scales, mins, 8, True)
        dequant_out = ext_quant.unpack_single_precision(output, 8, cuda_scales, mins, N, num_features // group_size,
                                                        group_size)

        cuda_quant_error = (dequant_out - data).norm()

        # torch

        # clip_val = maxs - mins
        y = (temp - mins) / scale * (level - 1)
        # y = torch.clamp(y, min=0)
        y = torch.clamp(y, min=0, max=level - 1)
        y = torch.round(y)
        # y = torch.clamp(y, min=0, max=level - 1)

        torch_dequant_out = y / (level - 1) * scale + mins
        # torch_dequant_out = torch_dequant_out.reshape(128 * 197, 197, 3).permute(0, 2, 1)
        torch_quant_error = (torch_dequant_out - temp).norm()
        diff += (cuda_quant_error - torch_quant_error).item()

        # print('torch quant error = ', torch_quant_error)
    print('diff = ', diff/100)

def cuda_quant(data, mins, scales):
    # cuda_scales = 255 / (maxs - mins + 2e-6)
    N, num_groups, group_size = data.shape
    num_features = int(np.prod(data.shape[1:]))
    num_features = (num_features + (group_size - num_features % group_size) % group_size)

    output = ext_quant.pack_single_precision(data, scales, mins, 8, True)
    # dequant_out = ext_quant.unpack_single_precision(output, 8, scales, mins, N, num_features // group_size, group_size)

def cuda_quant2(data, mins, scales):
    # cuda_scales = 255 / (maxs - mins + 2e-6)
    output = ext_quant.pack_single_precision(data, scales, mins, 8, True, 128)
    # print('..')

def torch_quant(data, mins, scales):
    # torch
    level = 255
    y = (data - mins) * scales
    y = torch.round(y)
    y = torch.clamp(y, min=0, max=level)

    # torch_dequant_out = y / scales + mins

def test_cuda():
    # Torch Forward: 9015.575 us
    cuda_data = torch.randn(197*197, 3, 128).cuda()
    torch_data = cuda_data.permute(0, 2, 1).reshape(-1, 3)

    mins, _ = torch_data.min(0)
    maxs, _ = torch_data.max(0)
    scales = 255 / (maxs - mins + 2e-6)
    output = ext_quant.pack_single_precision(cuda_data, scales, mins, 8, True)


if __name__ == '__main__':
    check_quant()
    # test_cuda()
    # # CUDA Forward: 61.892 us
    # # Torch Forward: 200.598 us
    # cuda_data = torch.randn(128 * 64, 3, 197).cuda()

    # # CUDA Forward: 55.924 us
    # # Torch Forward: 200.393 us
    # cuda_data = torch.randn(128 * 197, 3, 64).cuda()

    # # CUDA Forward: 165.276 us
    # # Torch Forward: 590.601 us
    # cuda_data = torch.randn(197 * 197, 3, 128).cuda()
    #
    # # CUDA Forward: 190.136 us
    # # Torch Forward: 606.257 us
    # cuda_data = torch.randn(128 * 197, 3, 197).cuda()

    # CUDA Forward: 2881.780 us
    # Torch Forward: 9015.575 us
    # cuda_data = torch.randn(197*197, 3, 256).cuda()
    # torch_data = cuda_data.permute(0, 2, 1).reshape(-1, 3)

    # CUDA Forward: 178.211 us
    # Torch Forward: 1148.423 us

    # CUDA Forward: 98.196 us
    # Torch Forward: 607.925 us

    # __float2int_rn
    # CUDA Forward: 92.401 us
    # Torch Forward: 584.516 us
    data = torch.randn(197 * 197 * 128, 3).cuda()

    mins, _ = data.min(0)
    maxs, _ = data.max(0)

    scales = 255 / (maxs - mins + 2e-6)

    forward = 0
    for _ in range(100000):
        start = time.time()
        cuda_quant2(data, mins, scales)
        forward += time.time() - start
    print('CUDA Forward: {:.3f} us'.format(forward * 1e6 / 1e5))

    forward = 0
    for _ in range(10000):
        start = time.time()
        torch_quant(data, mins, scales)
        forward += time.time() - start
    print('Torch Forward: {:.3f} us'.format(forward * 1e6 / 1e4))