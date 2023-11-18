//! ResNet Building Blocks
//!
//! Some Residual Network blocks used in UNet models.
//!
//! Denoising Diffusion Implicit Models, K. He and al, 2015.
//! https://arxiv.org/abs/1512.03385

use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::backend::Backend;
use crate::models::groupnorm::{GroupNorm, GroupNormConfig};

/// Configuration for a ResNet block.
#[derive(Debug, Clone, Copy)]
pub struct ResnetBlock2DConfig {
    /// The number of output channels, defaults to the number of input channels.
    pub out_channels: Option<usize>,
    pub temb_channels: Option<usize>,
    /// The number of groups to use in group normalization.
    pub groups: usize,
    pub groups_out: Option<usize>,
    /// The epsilon to be used in the group normalization operations.
    pub eps: f64,
    /// Whether to use a 2D convolution in the skip connection. When using None,
    /// such a convolution is used if the number of input channels is different from
    /// the number of output channels.
    pub use_in_shortcut: Option<bool>,
    // non_linearity: silu
    /// The final output is scaled by dividing by this value.
    pub output_scale_factor: f64,
}

impl Default for ResnetBlock2DConfig {
    fn default() -> Self {
        Self {
            out_channels: None,
            temb_channels: Some(512),
            groups: 32,
            groups_out: None,
            eps: 1e-6,
            use_in_shortcut: None,
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct ResnetBlock2D<B: Backend> {
    norm1: GroupNorm<B>,
    conv1: Conv2d<B>,
    norm2: GroupNorm<B>,
    conv2: Conv2d<B>,
    time_emb_proj: Option<Linear<B>>,
    conv_shortcut: Option<Conv2d<B>>,
    config: ResnetBlock2DConfig,
}

impl<B: Backend> ResnetBlock2D<B> {
    pub fn new(
        in_channels: usize,
        config: ResnetBlock2DConfig,
    ) -> Self {
        let out_channels = config.out_channels.unwrap_or(in_channels);
        let conv_cgf = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1));
        let norm1 = GroupNormConfig::new(config.groups, in_channels)
            .with_epsilon(config.eps)
            .init();
        let conv1 = conv_cgf.init();
        let groups_out = config.groups_out.unwrap_or(config.groups);
        let norm2 = GroupNormConfig::new(groups_out, out_channels)
            .with_epsilon(config.eps)
            .init();
        let conv2 = conv_cgf.init();
        let use_in_shortcut = config.use_in_shortcut.unwrap_or(in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = Conv2dConfig::new([in_channels, out_channels], [1, 1]);
            Some(conv_cfg.init())
        } else {
            None
        };
        let time_emb_proj = config.temb_channels.map(|temb_channels| {
            let linear_cfg = LinearConfig::new(temb_channels, out_channels);
            // nn::linear(&vs / "time_emb_proj", temb_channels, out_channels, Default::default())
        });

        // let conv_shortcut = if use_in_shortcut {
        //     let conv_cfg = nn::ConvConfig { stride: 1, padding: 0, ..Default::default() };
        //     Some(nn::conv2d(&vs / "conv_shortcut", in_channels, out_channels, 1, conv_cfg))
        // } else {
        //     None
        // };
    }
}