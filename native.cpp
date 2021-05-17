
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

at::Tensor gelu_backward_cpu(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
    return at::native::gelu_backward_cpu(grad_output, input);
}

at::Tensor gelu_backward_cuda(
    const at::Tensor& grad_output,
    const at::Tensor& input) {
    return at::native::gelu_backward_cuda(grad_output, input);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cpu(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    int64_t M, int64_t N, double eps) {
    return at::native::layer_norm_cpu(input, weight, bias, M, N, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_forward_cuda(
    const at::Tensor & input,
    const at::Tensor & weight,
    const at::Tensor & bias,
    int64_t M, int64_t N, double eps) {
    return at::native::layer_norm_cuda(input, weight, bias, M, N, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_cpu(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const at::Tensor & weight,
    int64_t M, int64_t N, std::array<bool,3> output_mask) {
    return at::native::layer_norm_backward_cpu(grad_out, input, mean, rstd, weight, M, N, output_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> layer_norm_backward_cuda(
    const at::Tensor & grad_out,
    const at::Tensor & input,
    const at::Tensor & mean,
    const at::Tensor & rstd,
    const at::Tensor & weight,
    int64_t M, int64_t N, std::array<bool,3> output_mask) {
    return at::native::layer_norm_backward_cuda(grad_out, input, mean, rstd, weight, M, N, output_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv2d_backward", &conv2d_backward, "2d convolution backward");
    m.def("batch_norm_forward", &batch_norm_forward, "batch norm forward");
    m.def("batch_norm_backward", &batch_norm_backward, "batch norm backward");
    m.def("gelu_backward_cpu", &gelu_backward_cpu, "gelu backward (cpu version)");
    m.def("gelu_backward_cuda", &gelu_backward_cuda, "gelu backward (cuda version)");
    m.def("layer_norm_forward_cpu", &layer_norm_forward_cpu, "layer norm forward (cpu version)");
    m.def("layer_norm_backward_cpu", &layer_norm_backward_cpu, "layer norm backward (cpu version)");
    m.def("layer_norm_forward_cuda", &layer_norm_forward_cuda, "layer norm forward (cuda version)");
    m.def("layer_norm_backward_cuda", &layer_norm_backward_cuda, "layer norm backward (cuda version)");
}

