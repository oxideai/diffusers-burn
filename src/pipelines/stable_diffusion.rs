use crate::schedulers::ddim;
use crate::transformers::clip;

#[derive(Clone, Debug)]
pub struct StableDiffusionConfig {
    pub width: i64,
    pub height: i64,
    pub clip: clip::Config,
    autoencoder: vae::AutoEncoderKLConfig,
    unet: unet_2d::UNet2DConditionModelConfig,
    scheduler: ddim::DDIMSchedulerConfig,
}
