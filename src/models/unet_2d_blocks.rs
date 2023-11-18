use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::backend::Backend;
use burn::tensor::module::avg_pool2d;
use burn::tensor::Tensor;

#[derive(Module, Debug)]
struct Downsample2D<B: Backend> {
    conv: Option<Conv2d<B>>,
    padding: usize,
}

impl<B: Backend> Downsample2D<B> {
    fn new(
        in_channels: usize,
        use_conv: bool,
        out_channels: usize,
        padding: usize,
    ) -> Self {
        let conv = if use_conv {
            let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(padding, padding))
                .init();

            Some(conv)
        } else {
            None
        };
        Downsample2D { conv, padding }
    }

    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.conv {
            None => avg_pool2d(xs, [2, 2], [2, 2], [0, 0], true),
            Some(conv) => {
                if self.padding == 0 {
                    // TODO: Implement padding support
                    unimplemented!();
                    // let xs = xs;
                        // .pad_with_zeros(D::Minus1, 0, 1)?
                        // .pad_with_zeros(D::Minus2, 0, 1)?;
                    // conv.forward(&xs)
                } else {
                    conv.forward(xs)
                }
            }
        }
    }
}

// This does not support the conv-transpose mode.
#[derive(Module, Debug)]
struct Upsample2D<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> Upsample2D<B> {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        Self { conv }
    }

    fn forward(&self, xs: Tensor<B, 4>, size: Option<(usize, usize)>) -> Tensor<B, 4> {
        let xs = match size {
            None => {
                let [_bsize, _channels, height, width] = xs.dims();
                upsample_nearest2d(xs, 2 * height, 2 * width)
                // xs.upsample_nearest2d(2 * height, 2 * width)
            }
            Some((h, w)) => upsample_nearest2d(xs, h, w),
        };

        self.conv.forward(xs)
    }
}


pub fn upsample_nearest2d<B: Backend>(tensor: Tensor<B, 4>, height: usize, width: usize) -> Tensor<B, 4> {
    let [batch_size, channels, _height, _width] = tensor.dims();
    let tensor = tensor
        .reshape([batch_size, channels, height, 1, width, 1])
        .repeat(3, 2)
        .repeat(5, 2)
        .reshape([batch_size, channels, 2 * height, 2 * width]);

    tensor
}

#[derive(Debug, Clone, Copy)]
pub struct DownEncoderBlock2DConfig {
    pub num_layers: usize,
    pub resnet_eps: f64,
    pub resnet_groups: usize,
    pub output_scale_factor: f64,
    pub add_downsample: bool,
    pub downsample_padding: usize,
}

impl Default for DownEncoderBlock2DConfig {
    fn default() -> Self {
        Self {
            num_layers: 1,
            resnet_eps: 1e-6,
            resnet_groups: 32,
            output_scale_factor: 1.,
            add_downsample: true,
            downsample_padding: 1,
        }
    }
}

#[derive(Debug)]
pub struct DownEncoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<Downsample2D>,
    pub config: DownEncoderBlock2DConfig,
}