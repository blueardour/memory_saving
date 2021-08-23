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
    Tensor data, Tensor scale, Tensor shift, int bits, bool stochastic, int batch_size);

Tensor unpack_single_precision_cuda(
    Tensor data, int bits, Tensor scale, Tensor shift, int64_t N, int64_t num_groups, int64_t group_size);

// Pack/Unpack single precision
Tensor pack_single_precision(Tensor data,
                             Tensor scale,
                             Tensor shift,
                             int bits,
                             bool stochastic, 
                             int batch_size) {

  return pack_single_precision_cuda(data, scale, shift, bits, stochastic, batch_size);
}

Tensor unpack_single_precision(Tensor data,
                               int bits,
                               Tensor scale,
                               Tensor shift,
                               int64_t N,
                               int64_t num_groups,
                               int64_t group_size) {
  CHECK_CUDA_TENSOR_DIM_TYPE(data, 1, torch::kInt8);

  return unpack_single_precision_cuda(data, bits, scale, shift,
                                      N, num_groups, group_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_single_precision", &pack_single_precision);
  m.def("unpack_single_precision", &unpack_single_precision);
}
