import memory_saving.cpp_extension.quantization as ext_quant
import time
import torch
import numpy as np

def check_quant_element_v2():
    data = torch.zeros(4, 2).cuda()
    data = data.transpose(1, 0).contiguous()
    # data = torch.zeros(2, 4).cuda()
    N, C = data.shape
    data[0, :] = 2
    data[1, :] = 4
    # data[:, 0] = 2.
    # data[:, 1] = 4.
    # data[:, 2] = 8.
    # data[:, 3] = 16.
    # data[:, 4] = 32.
    print(data)
    # temp = data.permute(0, 2, 1).reshape(-1, 3)

    mins, _ = data.min(0)
    maxs, _ = data.max(0)

    # test_shift = torch.tensor([2., 4., 8., 16., 32]).cuda()
    test_shift = torch.tensor([2., 4.]).cuda()

    level = 255
    scale = maxs - mins + 2e-6
    cuda_scales = 255 / scale

    N, C = data.shape

    output = ext_quant.pack_single_precision(data, cuda_scales, test_shift, 8, True)
    print(output)
    dequant_out = ext_quant.unpack_single_precision(output, 8, cuda_scales, test_shift, N, C)
    print(dequant_out)
    print(torch.all(torch.eq(data, dequant_out)).item())
    # assert data == dequant_out

def check_quant_element():
    data = torch.zeros(2, 4, 64).cuda()
    B, N, C = data.shape
    data[:, :, 0] = 2.
    data[:, :, 1] = 4.
    data[:, :, 2] = 8.
    # data = torch.zeros(3, 4, 2).cuda()
    # data[0, :, :] = 2.
    # data[1, :, :] = 4.
    # data[2, :, :] = 8.
    print(data)
    # temp = data.permute(0, 2, 1).reshape(-1, 3)

    mins, _ = data.min(0)
    maxs, _ = data.max(0)
    print(mins)
    # maxs = torch.amax(data, (1))
    # mins = torch.amin(data, (1))

    test_shift = torch.tensor([2., 4., 8.]).cuda()

    level = 255
    scale = maxs - mins + 2e-6
    cuda_scales = 255 / scale

    # N, num_groups = data.shape

    output = ext_quant.pack_single_precision(data, cuda_scales, test_shift, 8, True, 2)
    print(output)
    dequant_out = ext_quant.unpack_single_precision(output, 8, cuda_scales, test_shift, B, N, C)
    print(dequant_out)
    print('..')


def check_quant_BN_C():
    diff = 0.
    for i in range(200):
        data = torch.randn(128 * 197, 192).cuda()
        # temp = data.permute(0, 2, 1).reshape(-1, 3)

        # mins, _ = data.min(0)
        # maxs, _ = data.max(0)
        mins = torch.amin(data, (0, 1))
        maxs = torch.amax(data, (0, 1))
        level = 255
        scale = maxs - mins + 2e-6
        cuda_scales = 255 / scale

        N, C = data.shape

        output = ext_quant.pack_single_precision(data, cuda_scales, mins, 8, True)
        dequant_out = ext_quant.unpack_single_precision(output, 8, cuda_scales, mins, N, C)
        cuda_quant_error = (dequant_out - data).norm()

        y = (data - mins) / scale * (level - 1)
        y = torch.round(y)
        y = torch.clamp(y, min=0, max=level - 1)
        torch_dequant_out = y / (level - 1) * scale + mins

        torch_quant_error = (torch_dequant_out - data).norm()
        cur_error = (cuda_quant_error - torch_quant_error).item()
        print(cur_error)
        diff += cur_error
    print('diff = ', diff / 200)


def check_quant():
    diff = 0.
    for i in range(200):
        B = 128
        N = 197
        C = 192
        B, H, N, D = 128, 3, 197, 64

        data = torch.randn(128, 192).cuda()
        data = data.permute(1, 0)

        # data = torch.randn(128, 197, 192).cuda()
        # data = data.reshape(128, 197, 3, 64).permute(0, 2, 1, 3)
        # # data = torch.randn(128, 3, 197, 64).cuda()
        # data = data.permute(1, 3, 0, 2).reshape(H * D, B * N)
        # data = data.permute(2, 1, 0).reshape(C, -1)
        # data = torch.randn(C, B*N).cuda()
        # temp = data.permute(0, 2, 1).reshape(-1, 3)

        # mins, _ = data.min(0)
        # maxs, _ = data.max(0)
        mins = torch.amin(data, (1))
        maxs = torch.amax(data, (1))
        level = 255
        scale = maxs - mins + 2e-6
        cuda_scales = 255 / scale

        N, C = data.shape

        output = ext_quant.pack_single_precision(data, cuda_scales, mins, 8, True)
        dequant_out = ext_quant.unpack_single_precision(output, 8, cuda_scales, mins, N, C)
        cuda_quant_error = (dequant_out - data).norm()

        torch_data = data.transpose(1,0)
        # noise = torch_data.new(torch_data.shape).uniform_(-0.5, 0.5)
        y = (torch_data - mins) * cuda_scales
        y = torch.round(y)
        y = torch.clamp(y, min=0, max=level - 1)
        torch_dequant_out = y / cuda_scales + mins

        torch_quant_error = (torch_dequant_out - torch_data).norm()
        cur_error = (cuda_quant_error - torch_quant_error).item()
        print(cur_error)
        diff += cur_error
    print('diff = ', diff / 200)


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
        y = torch.round(y)
        y = torch.clamp(y, min=0, max=level - 1)

        # y = torch.clamp(y, min=0, max=level - 1)

        torch_dequant_out = y / (level - 1) * scale + mins
        # torch_dequant_out = torch_dequant_out.reshape(128 * 197, 197, 3).permute(0, 2, 1)
        torch_quant_error = (torch_dequant_out - temp).norm()
        diff += (cuda_quant_error - torch_quant_error).item()

        # print('torch quant error = ', torch_quant_error)
    print('diff = ', diff / 100)


def cuda_quant(data, mins, scales):
    # cuda_scales = 255 / (maxs - mins + 2e-6)
    N, num_groups, group_size = data.shape
    num_features = int(np.prod(data.shape[1:]))
    num_features = (num_features + (group_size - num_features % group_size) % group_size)

    output = ext_quant.pack_single_precision(data, scales, mins, 8, True)
    # dequant_out = ext_quant.unpack_single_precision(output, 8, scales, mins, N, num_features // group_size, group_size)


def cuda_quant2(data, mins, scales, B, N, C):
    # cuda_scales = 255 / (maxs - mins + 2e-6)
    output = ext_quant.pack_single_precision(data, scales, mins, 8, True, B, N, C)
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
    cuda_data = torch.randn(197 * 197, 3, 128).cuda()
    torch_data = cuda_data.permute(0, 2, 1).reshape(-1, 3)

    mins, _ = torch_data.min(0)
    maxs, _ = torch_data.max(0)
    scales = 255 / (maxs - mins + 2e-6)
    output = ext_quant.pack_single_precision(cuda_data, scales, mins, 8, True)


def factorization(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = num // k
                break
    return factor

def get_threads(num):
    threads = 1024
    while (num % threads != 0 and threads > 1):
        threads /= 2
    return threads, num // threads

def compare(B, N, C):
    print('channel dim = ', C)
    threads, blocks = get_threads(B * N)
    print(threads, blocks)
    data = torch.randn(C, B * N).cuda()
    # data = torch.randn(192, 128 * 197).cuda()

    mins = torch.amin(data, (1))
    maxs = torch.amax(data, (1))

    scales = 255 / (maxs - mins + 2e-6)

    forward = 0
    for _ in range(10000):
        start = time.time()
        cuda_quant2(data, mins, scales, B, N, C)
        forward += time.time() - start
    print('CUDA Forward: {:.3f} us'.format(forward * 1e6 / 1e4))

    torch_forward = 0
    torch_data = data.transpose(1, 0)
    for _ in range(10000):
        start = time.time()
        torch_quant(torch_data, mins, scales)
        torch_forward += time.time() - start
    print('Torch Forward: {:.3f} us'.format(torch_forward * 1e6 / 1e4))
    print('speed up ratio = ', torch_forward/forward)
    print()


if __name__ == '__main__':

    # B = 2
    # N = 196
    # C = 192
    #
    # BN = B* N
    # threads, blocks = get_threads(BN)
    # print(threads, blocks)
    # # factor = 10
    # factors = factorization(BN)
    # print(factors)

    check_quant_element_v2()
    # check_quant()
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

    # channel wise quantization
    # data = torch.randn(128, 197, 192).cuda()
    # 192
    # CUDA Forward: 31.143 us
    # Torch Forward: 195.919 us

    # 192 with noise
    # CUDA Forward: 59.587 us
    # Torch Forward: 199.620 us

    # 384
    # CUDA Forward: 60.601 us
    # Torch Forward: 380.579 us

    # 384 with noise
    # CUDA Forward: 118.746 us
    # Torch Forward: 388.483 us

    # 768
    # CUDA Forward: 123.359 us
    # Torch Forward: 753.552 us

    # 768 with noise
    # CUDA Forward: 252.021 us
    # Torch Forward: 768.403 us

    # without noise
    # data = torch.randn(128, 3136, 96).cuda()
    # CUDA Forward: 260.118 us
    # Torch Forward: 1494.353 us

    # data = torch.randn(128, 784, 192).cuda()
    # CUDA Forward: 117.771 us
    # Torch Forward: 752.368 us

    # data = torch.randn(128, 196, 384).cuda()
    # CUDA Forward: 60.966 us
    # Torch Forward: 380.124 us

    # data = torch.randn(128, 49, 768).cuda()
    # CUDA Forward: 32.774 us
    # Torch Forward: 196.369 us
    # -= 2 5 times, batch size = 2

    # B = 128
    # N = 196
    # # C = 192
    # head_list = [3, 6, 12]
    # for head in head_list:
    #
    #     C = head * 64

    # swin arch
    # B = 128
    # seq_lens = [3136, 784, 196, 49]
    # head_dim = 32
    # head_list = {
    #     'tiny': [3, 6, 12, 24],
    #     'small': [3, 6, 12, 24],
    #     'base': [4, 8, 16, 32],
    #     'large': [6, 12, 24, 48],
    # }
    #
    # for model, heads in head_list.items():
    #     print(model)
    #     for i in range(4):
    #         N = seq_lens[i]
    #         C = heads[i] * head_dim
    #         print(B, N, C)
    #         compare(B, N, C)
    #

