use anyhow::Result;
use clap::Args;

use burn::{
    config::Config,
    module::{Module, Param},
    nn,
    tensor::{backend::Backend, Tensor, f16},
};

#[cfg(feature = "wgpu-backend")]
use burn_wgpu::{WgpuBackend, WgpuDevice, AutoGraphicsApi};
#[cfg(not(feature = "wgpu-backend"))]
use burn_tch::{TchBackend, TchDevice};

const GUIDANCE_SCALE: f64 = 7.5;

#[derive(Args, Debug)]
pub struct SampleArgs {
    /// The prompt to be used for image generation.
    #[arg(long, default_value = "Sand dunes with a solar eclipse in the sky")]
    prompt: String,

    #[arg(long, default_value = "")]
    negative_prompt: String,

    /// When set, use the CPU for the listed devices, can be 'all', 'unet', 'clip', etc.
    /// Multiple values can be set.
    #[arg(long)]
    cpu: Vec<String>,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<i64>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<i64>,

    /// The UNet weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    unet_weights: Option<String>,

    /// The CLIP weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    clip_weights: Option<String>,

    /// The VAE weight file, in .ot or .safetensors format.
    #[arg(long, value_name = "FILE")]
    vae_weights: Option<String>,

    #[arg(long, value_name = "FILE")]
    /// The file specifying the tokenizer to used for tokenization.
    tokenizer: Option<String>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    #[arg(long)]
    sliced_attention_size: Option<i64>,

    /// The number of steps to run the diffusion for.
    #[arg(long, default_value_t = 30)]
    n_steps: usize,

    /// The random seed to be used for the generation.
    #[arg(long, default_value_t = 32)]
    seed: i64,

    /// The number of samples to generate.
    #[arg(long, default_value_t = 1)]
    num_samples: i64,

    /// The name of the final image to generate.
    #[arg(long, value_name = "FILE", default_value = "sd_final.png")]
    final_image: String,

    /// Use autocast (disabled by default as it may use more memory in some cases).
    #[arg(long, action)]
    autocast: bool,

    #[arg(long, value_enum, default_value = "v2-1")]
    sd_version: StableDiffusionVersion,

    /// Generate intermediary images at each step.
    #[arg(long, action)]
    intermediary_images: bool,

    #[arg(long)]
    use_flash_attn: bool,

    // #[arg(long)]
    // use_f16: bool,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
}

pub fn handle_convert(args: &SampleArgs) -> Result<()> {
    #[cfg(feature = "wgpu-backend")] {
        type Backend = WgpuBackend<AutoGraphicsApi, f32, i32>;
        let device = WgpuDevice::BestAvailable;
    }
    #[cfg(not(feature = "wgpu-backend"))] {
        type Backend = TchBackend<f32>;
        let device = TchDevice::Cpu;
    }

    let sd_config = match args.sd_version {
        StableDiffusionVersion::V1_5 => {
            stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::Xl => {
            stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
        }
    };

    Ok(())
}
