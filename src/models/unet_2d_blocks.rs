use crate::models::resnet::{ResnetBlock2D, ResnetBlock2DConfig};
use crate::utils::upsample::upsample_nearest2d;
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;
use burn::tensor::backend::Backend;
use burn::tensor::module::avg_pool2d;
use burn::tensor::Tensor;

#[derive(Config)]
struct Downsample2DConfig {
    in_channels: usize,
    use_conv: bool,
    out_channels: usize,
    padding: usize,
}

impl Downsample2DConfig {
    fn init<B: Backend>(&self) -> Downsample2D<B> {
        let conv = if self.use_conv {
            let conv = Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
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

#[derive(Module, Debug)]
struct Downsample2D<B: Backend> {
    conv: Option<Conv2d<B>>,
    padding: usize,
}

impl<B: Backend> Downsample2D<B> {
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

#[derive(Config)]
struct Upsample2DConfig {
    in_channels: usize,
    out_channels: usize,
}

impl Upsample2DConfig {
    fn init<B: Backend>(&self) -> Upsample2D<B> {
        let conv = Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        Upsample2D { conv }
    }
}

// This does not support the conv-transpose mode.
#[derive(Module, Debug)]
struct Upsample2D<B: Backend> {
    conv: Conv2d<B>,
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
    pub num_layers: usize,
    pub resnet_eps: f64,
    pub resnet_groups: usize,
    pub output_scale_factor: f64,
    pub add_downsample: bool,
    pub downsample_padding: usize,
}

impl DownEncoderBlock2DConfig {
    pub fn init<B: Backend>(&self) -> DownEncoderBlock2D<B> {
        let resnets: Vec<_> = {
            let conv_cfg = ResnetBlock2DConfig {
                out_channels: Some(self.out_channels),
                groups: self.resnet_groups,
                groups_out: None,
                eps: self.resnet_eps,
                temb_channels: None,
                use_in_shortcut: None,
                output_scale_factor: self.output_scale_factor,
            };
            (0..(self.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 {
                        self.in_channels
                    } else {
                        self.out_channels
                    };
                    conv_cfg.init(in_channels)
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

#[derive(Module, Debug)]
pub struct DownEncoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    downsampler: Option<Downsample2D<B>>,
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
    pub num_layers: usize,
    pub resnet_eps: f64,
    pub resnet_groups: usize,
    pub output_scale_factor: f64,
    pub add_upsample: bool,
}

impl UpDecoderBlock2DConfig {
    pub fn init<B: Backend>(&self, in_channels: usize) -> UpDecoderBlock2D<B> {
        let resnets: Vec<_> = {
            let conv_cfg = ResnetBlock2DConfig {
                out_channels: Some(self.out_channels),
                groups: self.resnet_groups,
                groups_out: None,
                eps: self.resnet_eps,
                temb_channels: None,
                use_in_shortcut: None,
                output_scale_factor: self.output_scale_factor,
            };
            (0..(self.num_layers))
                .map(|i| {
                    let in_channels = if i == 0 {
                        in_channels
                    } else {
                        self.out_channels
                    };
                    conv_cfg.init(in_channels)
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

#[derive(Module, Debug)]
pub struct UpDecoderBlock2D<B: Backend> {
    resnets: Vec<ResnetBlock2D<B>>,
    upsampler: Option<Upsample2D<B>>,
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
