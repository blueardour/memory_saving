
#include <torch/extension.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>

#include <array>

std::tuple<at::Tensor, at::Tensor> conv2d_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    c10::ArrayRef<long int> padding,
    c10::ArrayRef<long int> stride,
    c10::ArrayRef<long int> dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic,
    std::array<bool, 2> output_mask) {

    return at::cudnn_convolution_backward(
        input,
        grad_output,
        weight,
        padding,
        stride,
        dilation,
        groups,
        benchmark,
        deterministic,
        output_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_backward", &conv2d_backward, "2d convolution backward");
}

