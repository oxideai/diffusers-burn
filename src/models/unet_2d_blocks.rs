//! 2D UNet Building Blocks
//!

use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{backend::Backend, module::avg_pool2d, Tensor},
};

use crate::utils::pad_with_zeros;

#[derive(Config)]
struct Downsample2DConfig {
    in_channels: usize,
    use_conv: bool,
    out_channels: usize,
    padding: usize,
}

#[derive(Module, Debug)]
struct Downsample2D<B: Backend> {
    conv: Option<nn::conv::Conv2d<B>>,
    padding: usize,
}

impl Downsample2DConfig {
    fn init<B: Backend>(&self) -> Downsample2D<B> {
        let conv = if self.use_conv {
            let conv = nn::conv::Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(nn::PaddingConfig2d::Explicit(self.padding, self.padding))
                .init();

            Some(conv)
        } else {
            None
        };

        Downsample2D {
            conv,
            padding: self.padding,
        }
    }
}

impl<B: Backend> Downsample2D<B> {
    fn pad_tensor(xs: Tensor<B, 4>, padding: usize) -> Tensor<B, 4> {
        if padding == 0 {
            let xs = pad_with_zeros(xs, 4 - 1, 0, 1);
            return pad_with_zeros(xs, 4 - 2, 0, 1);
        }

        return xs;
    }

    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.conv {
            None => avg_pool2d(xs, [2, 2], [2, 2], [0, 0], true),
            Some(conv) => conv.forward(Self::pad_tensor(xs, self.padding)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::Data;

    #[test]
    fn test_downsample_2d_no_conv() {
        let tensor: Tensor<TestBackend, 4> = Tensor::from_data(Data::from([
            [
                [[0.0351, 0.4179], [0.0137, 0.6947]],
                [[0.9526, 0.5386], [0.2856, 0.1839]],
                [[0.3215, 0.4595], [0.6777, 0.3946]],
                [[0.5221, 0.4230], [0.2774, 0.1069]],
            ],
            [
                [[0.8941, 0.8696], [0.5735, 0.8750]],
                [[0.6718, 0.4144], [0.1038, 0.2629]],
                [[0.7467, 0.9415], [0.5005, 0.6309]],
                [[0.6534, 0.2019], [0.3670, 0.8074]],
            ],
        ]));

        let downsample_2d = Downsample2DConfig::new(4, false, 4, 0).init();
        let output = downsample_2d.forward(tensor);

        output.into_data().assert_approx_eq(
            &Data::from([
                [[[0.2904]], [[0.4902]], [[0.4633]], [[0.3323]]],
                [[[0.8031]], [[0.3632]], [[0.7049]], [[0.5074]]],
            ]),
            3,
        );
    }

    #[test]
    fn test_pad_tensor_0() {
        let tensor: Tensor<TestBackend, 4> = Tensor::from_data(Data::from([
            [
                [[0.8600, 0.9473], [0.2543, 0.6181]],
                [[0.3889, 0.7722], [0.6736, 0.0454]],
                [[0.2809, 0.4672], [0.1632, 0.3959]],
                [[0.5317, 0.0831], [0.8353, 0.3654]],
            ],
            [
                [[0.6106, 0.4130], [0.7932, 0.8800]],
                [[0.8750, 0.1991], [0.7018, 0.7865]],
                [[0.7470, 0.2071], [0.2699, 0.4425]],
                [[0.7763, 0.0227], [0.6210, 0.0730]],
            ],
        ]));

        let output = Downsample2D::pad_tensor(tensor, 0);

        output.into_data().assert_approx_eq(
            &Data::from([
                [
                    [
                        [0.8600, 0.9473, 0.0000],
                        [0.2543, 0.6181, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.3889, 0.7722, 0.0000],
                        [0.6736, 0.0454, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.2809, 0.4672, 0.0000],
                        [0.1632, 0.3959, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.5317, 0.0831, 0.0000],
                        [0.8353, 0.3654, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                ],
                [
                    [
                        [0.6106, 0.4130, 0.0000],
                        [0.7932, 0.8800, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.8750, 0.1991, 0.0000],
                        [0.7018, 0.7865, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.7470, 0.2071, 0.0000],
                        [0.2699, 0.4425, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                    [
                        [0.7763, 0.0227, 0.0000],
                        [0.6210, 0.0730, 0.0000],
                        [0.0000, 0.0000, 0.0000],
                    ],
                ],
            ]),
            3,
        );
    }
}
