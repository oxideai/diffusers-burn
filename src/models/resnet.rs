//! ResNet Building Blocks
//!
//! Some Residual Network blocks used in UNet models.
//!
//! Denoising Diffusion Implicit Models, K. He and al, 2015.
//! https://arxiv.org/abs/1512.03385

use crate::models::groupnorm::{GroupNorm, GroupNormConfig};
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::activation::silu;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration for a ResNet block.
#[derive(Config, Debug)]
pub struct ResnetBlock2DConfig {
    /// The number of output channels, defaults to the number of input channels.
    pub out_channels: Option<usize>,
    pub temb_channels: Option<usize>,
    /// The number of groups to use in group normalization.
    #[config(default = 32)]
    pub groups: usize,
    pub groups_out: Option<usize>,
    /// The epsilon to be used in the group normalization operations.
    #[config(default = 1e-6)]
    pub eps: f64,
    /// Whether to use a 2D convolution in the skip connection. When using None,
    /// such a convolution is used if the number of input channels is different from
    /// the number of output channels.
    pub use_in_shortcut: Option<bool>,
    // non_linearity: silu
    /// The final output is scaled by dividing by this value.
    #[config(default = 1.)]
    pub output_scale_factor: f64,
}

impl ResnetBlock2DConfig {
    pub fn init<B: Backend>(&self, in_channels: usize) -> ResnetBlock2D<B> {
        let out_channels = self.out_channels.unwrap_or(in_channels);
        let conv_cgf = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1));
        let norm1 = GroupNormConfig::new(self.groups, in_channels)
            .with_epsilon(self.eps)
            .init();
        let conv1 = conv_cgf.init();
        let groups_out = self.groups_out.unwrap_or(self.groups);
        let norm2 = GroupNormConfig::new(groups_out, out_channels)
            .with_epsilon(self.eps)
            .init();
        let conv2 = conv_cgf.init();
        let use_in_shortcut = self.use_in_shortcut.unwrap_or(in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = Conv2dConfig::new([in_channels, out_channels], [1, 1]);
            Some(conv_cfg.init())
        } else {
            None
        };
        let time_emb_proj = self.temb_channels.map(|temb_channels| {
            let linear_cfg = LinearConfig::new(temb_channels, out_channels);
            linear_cfg.init()
        });

        ResnetBlock2D {
            norm1,
            conv1,
            norm2,
            conv2,
            time_emb_proj,
            conv_shortcut,
            output_scale_factor: self.output_scale_factor,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResnetBlock2D<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    time_emb_proj: Option<Linear<B>>,
    conv_shortcut: Option<Conv2d<B>>,
    output_scale_factor: f64,
}

impl<B: Backend> ResnetBlock2D<B> {
    pub fn forward(&self, xs: Tensor<B, 4>, temb: Option<Tensor<B, 4>>) -> Tensor<B, 4> {
        let shortcut_xs = match &self.conv_shortcut {
            Some(conv_shortcut) => conv_shortcut.forward(xs.clone()),
            None => xs.clone(),
        };

        let xs = self.norm1.forward(xs.clone());
        let xs = self.conv1.forward(silu(xs));
        // TODO: time embedding
        // let xs = match (temb, &self.time_emb_proj) {
        //     (Some(temb), Some(time_emb_proj)) => time_emb_proj
        //         .forward(silu(temb))
        //         .unsqueeze_dim::<5>(4 - 1)
        //         .unsqueeze_dim::<6>(5 - 1)
        //         + xs.unsqueeze(),
        //     _ => xs,
        // };
        let xs = self.conv2.forward(silu(self.norm2.forward(xs)));

        (shortcut_xs + xs) / self.output_scale_factor
    }
}
