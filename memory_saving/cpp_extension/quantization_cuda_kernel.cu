/*
 * Cuda kernels for quantization
 */

#include <torch/extension.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BLOCK_Y_DIM_MAX ((((int64_t)(1)) << 16) - 1)
#define fmax(a, b) ((a) > (b) ? (a): (b))
// #define fmin(a, b) ((a) < (b) ? (a): (b))

using torch::IntArrayRef;
using torch::Tensor;


// Pack float16/32 data into int8 bit stream
template<typename scalar_t, bool boundary_check>
__global__ void pack_single_precision_kernel(int32_t bits,
                                             const scalar_t* __restrict__ data,
                                             const scalar_t* __restrict__ scale,
                                             const scalar_t* __restrict__ shift,
                                             int8_t* __restrict__ packed
//                                              std::pair<uint64_t, uint64_t> seeds
                                            ) {
  const int64_t global_thread_id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

//   curandStatePhilox4_32_10_t state;
//   curand_init(seeds.first, global_thread_id, seeds.second, &state);
//   const float noise = curand_uniform(&state);

  const int64_t id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

//   previous
  uint8_t local_packed = 0;
  const int32_t val = __float2int_rn(fmaxf((data[id] - shift[threadIdx.x]), 0.0f));
//   const int32_t val = __float2int_rn(fmax((data[id] - shift[threadIdx.x]) * scale[threadIdx.x] + noise - 0.5, 0.0f));
//   const int32_t val = __float2int_rn(fmaxf((data[id] - shift[threadIdx.x]) * scale[threadIdx.x], 0.0f));
  local_packed |= val;
  packed[global_thread_id] = local_packed;
}

// Pack float16/32 data into int8 bit stream
Tensor pack_single_precision_cuda(Tensor data,
                                  Tensor scale,
                                  Tensor shift,
                                  int bits,
                                  bool stochastic) {
  int64_t B = data.size(0);
  int64_t N = data.size(1);
  int64_t C = data.size(2);

  // Compute total bits
  TORCH_CHECK(8 % bits == 0);

  int64_t total_bits = (int64_t)bits * (B * N * C);
  auto options = torch::TensorOptions().dtype(torch::kInt8).device(data.device());
  Tensor packed = torch::empty({(total_bits + 8) / 8,}, options);

//   // Random number generator
//   auto gen = at::check_generator<at::CUDAGeneratorImpl>(at::cuda::detail::getDefaultCUDAGenerator());
//   std::pair<uint64_t, uint64_t> rng_engine_inputs;
//   {
//     // See Note [Acquire lock when using random generators]
//     std::lock_guard<std::mutex> lock(gen->mutex_);
//     rng_engine_inputs = gen->philox_engine_inputs(C);
//   }
//   TORCH_CHECK(stochastic);

  dim3 block_dim(N, B, 1);
  dim3 thread_dim(C, 1, 1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
    pack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
      bits,
      data.data_ptr<scalar_t>(),
      scale.data_ptr<scalar_t>(), shift.data_ptr<scalar_t>(),
      packed.data_ptr<int8_t>()
//       rng_engine_inputs
      );
  }));
  // int64_t needed_blocks_y = N/group_size;
  // // Call pack kernels
  // for (int64_t block_idx_y_base = 0; block_idx_y_base < needed_blocks_y; block_idx_y_base += BLOCK_Y_DIM_MAX) {
  //   dim3 block_dim(N, std::min(needed_blocks_y - block_idx_y_base, BLOCK_Y_DIM_MAX), 1);
  //   dim3 thread_dim(group_size, 1, 1);

  //   AT_DISPATCH_FLOATING_TYPES_AND_HALF(data.scalar_type(), "pack_single_precision", ([&] {
  //     pack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
  //       bits,
  //       data.data_ptr<scalar_t>(),
  //       scale.data_ptr<scalar_t>(), shift.data_ptr<scalar_t>(),
  //       packed.data_ptr<int8_t>(),
  //       // rng_engine_inputs,
  //       N, num_groups, group_size, block_idx_y_base);
  //   }));
  // }

  return packed;
}

// Unpack int32 bit stream to float16/32 data
template<typename scalar_t, bool boundary_check>
__global__ void unpack_single_precision_kernel(int32_t bits,
                                               const int8_t* __restrict__ data,
                                               const scalar_t* __restrict__ scale,
                                               const scalar_t* __restrict__ shift,
                                               scalar_t* __restrict__ unpacked) {
  // const int64_t no = blockIdx.y + block_idx_y_base;
  // const int group_id = blockIdx.x;
  // const int d = threadIdx.x;
  // const int64_t global_thread_id = (no * num_groups + group_id) * group_size + d;
  const int64_t global_thread_id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

  uint8_t local_packed = data[global_thread_id];
  int mask = ((1 << bits) - 1);
  const int val = local_packed & mask;
  const int64_t id = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//   unpacked[id] = ((scalar_t)val) / scale[threadIdx.x] + shift[threadIdx.x];
  unpacked[id] = ((scalar_t)val) + shift[threadIdx.x];
}

// Unpack int32 bit stream to float16/32 data
Tensor unpack_single_precision_cuda(Tensor data,
                                    int bits,
                                    Tensor scale,
                                    Tensor shift,
                                    int64_t B,
                                    int64_t N,
                                    int64_t C) {
  auto options = torch::TensorOptions().dtype(scale.dtype()).device(data.device());
  Tensor unpacked = torch::empty({B, N, C}, options);

  // dim3 block_dim(num_groups, std::min(needed_blocks_y - block_idx_y_base, BLOCK_Y_DIM_MAX), 1);
  // dim3 thread_dim(group_size, 1, 1);
  dim3 block_dim(N, B, 1);
  dim3 thread_dim(C, 1, 1);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
    unpack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
      bits,
      data.data_ptr<int8_t>(),
      scale.data_ptr<scalar_t>(), shift.data_ptr<scalar_t>(),
      unpacked.data_ptr<scalar_t>()
      );
  }));

  // int64_t needed_blocks_y = N/group_size;
  // for (int64_t block_idx_y_base = 0; block_idx_y_base < needed_blocks_y; block_idx_y_base += BLOCK_Y_DIM_MAX) {
  //   dim3 block_dim(num_groups, std::min(needed_blocks_y - block_idx_y_base, BLOCK_Y_DIM_MAX), 1);
  //   dim3 thread_dim(group_size, 1, 1);

  //   AT_DISPATCH_FLOATING_TYPES_AND_HALF(scale.scalar_type(), "unpack_single_precision", ([&] {
  //     unpack_single_precision_kernel<scalar_t, false><<<block_dim, thread_dim>>>(
  //       bits,
  //       data.data_ptr<int8_t>(),
  //       scale.data_ptr<scalar_t>(), shift.data_ptr<scalar_t>(),
  //       unpacked.data_ptr<scalar_t>(),
  //       N, num_groups, group_size, block_idx_y_base);
  //   }));
  // }

  return unpacked;
}
