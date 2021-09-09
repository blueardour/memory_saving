/*
 * Cuda operators for quantization and packing
 */

#include <torch/extension.h>
#include <torch/torch.h>

#include "ext_common.h"

using torch::autograd::Function;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;
using torch::Tensor;
using torch::IntArrayRef;

Tensor pack_single_precision_cuda(
    Tensor data, Tensor scale, Tensor shift, int bits, bool stochastic);

Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor shift, int64_t B, int64_t N, int64_t C);

// Pack/Unpack single precision
Tensor pack_single_precision(Tensor data,
                             Tensor scale,
                             Tensor shift,
                             int bits,
                             bool stochastic) {

  return pack_single_precision_cuda(data, scale, shift, bits, stochastic);
}

Tensor unpack_single_precision(Tensor data,
                               int bits,
                               Tensor scale,
                               Tensor shift,
                               int64_t B,
                               int64_t N,
                               int64_t C) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);

  return unpack_single_precision_cuda(data, bits, scale, shift,
                                      B, N, C);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_single_precision", &pack_single_precision);
  m.def("unpack_single_precision", &unpack_single_precision);
}
