use crate::transformers::clip;
use crate::transformers::clip::ClipConfig;
use burn::config::Config;
use burn::module::Module;
use burn::tensor::backend::Backend;

#[derive(Config, Debug)]
pub struct StableDiffusionConfig {
    width: i64,
    height: i64,
}

impl StableDiffusionConfig {
    pub fn init<B: Backend>(
        &self,
        clip_config: ClipConfig,
        device: &B::Device,
    ) -> StableDiffusion<B> {
        StableDiffusion {
            width: self.width,
            height: self.height,
            clip: clip_config.init_text_transformer(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct StableDiffusion<B: Backend> {
    width: i64,
    height: i64,
    clip: clip::ClipTextTransformer<B>,
    // autoencoder: vae::AutoEncoderKLConfig,
    // unet: unet_2d::UNet2DConditionModelConfig,
    // scheduler: ddim::DDIMSchedulerConfig,
}
