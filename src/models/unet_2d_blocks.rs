//! 2D UNet Building Blocks
//!

use burn::{
    config::Config,
    module::Module,
    nn,
    tensor::{backend::Backend, module::avg_pool2d, Tensor},
};

use crate::utils::{pad_with_zeros, upsample_nearest2d};

use super::{
    attention::{AttentionBlock, AttentionBlockConfig},
    resnet::{ResnetBlock2D, ResnetBlock2DConfig},
};

use alloc::vec;
use alloc::vec::Vec;

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

#[derive(Config)]
struct Upsample2DConfig {
    in_channels: usize,
    out_channels: usize,
}

// This does not support the conv-transpose mode.
#[derive(Module, Debug)]
struct Upsample2D<B: Backend> {
    conv: nn::conv::Conv2d<B>,
}

impl Upsample2DConfig {
    fn init<B: Backend>(&self) -> Upsample2D<B> {
        let conv = nn::conv::Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
            .with_padding(nn::PaddingConfig2d::Explicit(1, 1))
            .init();

        Upsample2D { conv }
    }
}

impl<B: Backend> Upsample2D<B> {
    fn forward(&self, xs: Tensor<B, 4>, size: Option<(usize, usize)>) -> Tensor<B, 4> {
        let xs = match size {
            None => {
                let [_bsize, _channels, height, width] = xs.dims();
                upsample_nearest2d(xs, 2 * height, 2 * width)
            }
            Some((h, w)) => upsample_nearest2d(xs, h, w),
        };

        self.conv.forward(xs)
    }
}

#[derive(Config, Debug)]
pub struct DownEncoderBlock2DConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    #[config(default = 1)]
    pub num_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    #[config(default = 32)]
    pub resnet_groups: usize,
    #[config(default = 1.)]
    pub output_scale_factor: f64,
    #[config(default = true)]
    pub add_downsample: bool,
    #[config(default = 1)]
    pub downsample_padding: usize,
}

#[derive(Module, Debug)]
pub struct DownEncoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    downsampler: Option<Downsample2D<B>>,
}

impl DownEncoderBlock2DConfig {
    pub fn init<B: Backend>(&self) -> DownEncoderBlock2D<B> {
        let resnets: Vec<_> = {
            (0..(self.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 {
                        self.in_channels
                    } else {
                        self.out_channels
                    };

                    let conv_cfg = ResnetBlock2DConfig::new(in_channels)
                        .with_out_channels(Some(self.out_channels))
                        .with_groups(self.resnet_groups)
                        .with_eps(self.resnet_eps)
                        .with_output_scale_factor(self.output_scale_factor);

                    conv_cfg.init()
                })
                .collect()
        };

        let downsampler = if self.add_downsample {
            let downsample_cfg = Downsample2DConfig {
                in_channels: self.out_channels,
                use_conv: true,
                out_channels: self.out_channels,
                padding: self.downsample_padding,
            };
            Some(downsample_cfg.init())
        } else {
            None
        };

        DownEncoderBlock2D {
            resnets,
            downsampler,
        }
    }
}

impl<B: Backend> DownEncoderBlock2D<B> {
    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut xs = xs.clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(xs, None)
        }
        match &self.downsampler {
            Some(downsampler) => downsampler.forward(xs),
            None => xs,
        }
    }
}

#[derive(Config, Debug)]
pub struct UpDecoderBlock2DConfig {
    pub out_channels: usize,
    #[config(default = 1)]
    pub num_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    #[config(default = 32)]
    pub resnet_groups: usize,
    #[config(default = 1.)]
    pub output_scale_factor: f64,
    #[config(default = true)]
    pub add_upsample: bool,
}

#[derive(Module, Debug)]
pub struct UpDecoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    upsampler: Option<Upsample2D<B>>,
}

impl UpDecoderBlock2DConfig {
    pub fn init<B: Backend>(&self, in_channels: usize) -> UpDecoderBlock2D<B> {
        let resnets: Vec<_> = {
            (0..(self.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 {
                        in_channels
                    } else {
                        self.out_channels
                    };

                    let conv_cfg = ResnetBlock2DConfig::new(in_channels)
                        .with_out_channels(Some(self.out_channels))
                        .with_groups(self.resnet_groups)
                        .with_eps(self.resnet_eps)
                        .with_output_scale_factor(self.output_scale_factor);

                    conv_cfg.init()
                })
                .collect()
        };

        let upsampler = if self.add_upsample {
            let upsample = Upsample2DConfig {
                in_channels: self.out_channels,
                out_channels: self.out_channels,
            };

            Some(upsample.init())
        } else {
            None
        };

        UpDecoderBlock2D { resnets, upsampler }
    }
}

impl<B: Backend> UpDecoderBlock2D<B> {
    fn forward(&self, xs: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut xs = xs.clone();
        for resnet in self.resnets.iter() {
            xs = resnet.forward(xs, None)
        }
        match &self.upsampler {
            Some(upsampler) => upsampler.forward(xs, None),
            None => xs,
        }
    }
}

#[derive(Config, Debug)]
pub struct UNetMidBlock2DConfig {
    in_channels: usize,
    temb_channels: Option<usize>,
    #[config(default = 1)]
    pub num_layers: usize,
    #[config(default = 1e-6)]
    pub resnet_eps: f64,
    pub resnet_groups: Option<usize>,
    pub attn_num_head_channels: Option<usize>,
    // attention_type "default"
    #[config(default = 1.)]
    pub output_scale_factor: f64,
}

#[derive(Module, Debug)]
struct AttentionResnetBlock2D<B: Backend> {
    attention_block: AttentionBlock<B>,
    resnet_block: ResnetBlock2D<B>,
}

#[derive(Module, Debug)]
pub struct UNetMidBlock2D<B: Backend> {
    resnet: ResnetBlock2D<B>,
    attn_resnets: Vec<AttentionResnetBlock2D<B>>,
}

impl UNetMidBlock2DConfig {
    pub fn init<B: Backend>(&self) -> UNetMidBlock2D<B> {
        let resnet_groups = self
            .resnet_groups
            .unwrap_or_else(|| usize::min(self.in_channels / 4, 32));

        let resnet = ResnetBlock2DConfig::new(self.in_channels)
            .with_eps(self.resnet_eps)
            .with_groups(resnet_groups)
            .with_output_scale_factor(self.output_scale_factor)
            .with_temb_channels(self.temb_channels)
            .init();

        let mut attn_resnets = vec![];
        for _index in 0..self.num_layers {
            let attention_block = AttentionBlockConfig::new(self.in_channels)
                .with_n_head_channels(self.attn_num_head_channels)
                .with_n_groups(resnet_groups)
                .with_rescale_output_factor(self.output_scale_factor)
                .with_eps(self.resnet_eps)
                .init();

            let resnet_block = ResnetBlock2DConfig::new(self.in_channels)
                .with_eps(self.resnet_eps)
                .with_groups(resnet_groups)
                .with_output_scale_factor(self.output_scale_factor)
                .with_temb_channels(self.temb_channels)
                .init();

            attn_resnets.push(AttentionResnetBlock2D {
                attention_block,
                resnet_block,
            })
        }

        UNetMidBlock2D {
            resnet,
            attn_resnets,
        }
    }
}

impl<B: Backend> UNetMidBlock2D<B> {
    pub fn forward(&self, xs: Tensor<B, 4>, temb: Option<Tensor<B, 2>>) -> Tensor<B, 4> {
        let mut xs = self.resnet.forward(xs, temb.clone());
        for block in self.attn_resnets.iter() {
            xs = block
                .resnet_block
                .forward(block.attention_block.forward(xs), temb.clone())
        }

        xs
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
