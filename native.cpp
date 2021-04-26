
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
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 7
    bool allow_tf32,
#endif
    std::array<bool, 2> output_mask
    ) {

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
#if TORCH_VERSION_MAJOR >= 1 && TORCH_VERSION_MINOR >= 7
        allow_tf32,
#endif
        output_mask);
}

// output, save_mean, save_var, reserve
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> batch_norm_forward(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    bool training,
    double average_factor,
    double epsilon) {

    return at::cudnn_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        average_factor,
        epsilon
        );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> batch_norm_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    const at::Tensor& running_mean,
    const at::Tensor& running_var,
    const at::Tensor& save_mean,
    const at::Tensor& save_var,
    double epsilon,
    const at::Tensor& reserveSpace) {

    return at::cudnn_batch_norm_backward(
        input,
        grad_output,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_var,
        epsilon,
        reserveSpace);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_backward", &conv2d_backward, "2d convolution backward");
    m.def("batch_norm_forward", &batch_norm_forward, "batch norm forward");
    m.def("batch_norm_backward", &batch_norm_backward, "batch norm backward");
}

