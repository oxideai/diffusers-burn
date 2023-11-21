use crate::transformers::clip;

#[derive(Debug)]
pub struct StableDiffusionConfig {
    pub width: i64,
    pub height: i64,
    pub clip: clip::ClipConfig,
    // autoencoder: vae::AutoEncoderKLConfig,
    // unet: unet_2d::UNet2DConditionModelConfig,
    // scheduler: ddim::DDIMSchedulerConfig,
}
