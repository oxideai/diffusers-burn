use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

pub fn upsample_nearest2d<B: Backend>(
    tensor: Tensor<B, 4>,
    height: usize,
    width: usize,
) -> Tensor<B, 4> {
    let [batch_size, channels, _height, _width] = tensor.dims();
    let tensor = tensor
        .reshape([batch_size, channels, height, 1, width, 1])
        .repeat(3, 2)
        .repeat(5, 2)
        .reshape([batch_size, channels, 2 * height, 2 * width]);

    tensor
}
