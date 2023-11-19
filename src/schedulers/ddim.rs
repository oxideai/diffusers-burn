use num_traits::ToPrimitive;
use std::marker::PhantomData;

use burn::{
    module::Module,
    tensor::{backend::Backend, Data, ElementConversion, Shape, Tensor},
};

use super::{BetaSchedule, PredictionType};

/// The configuration for the DDIM scheduler.
#[derive(Module, Debug, Clone)]
pub struct DDIMSchedulerConfig {
    /// The value of beta at the beginning of training.
    pub beta_start: f64,
    /// The value of beta at the end of training.
    pub beta_end: f64,
    /// How beta evolved during training.
    pub beta_schedule: BetaSchedule,
    /// The amount of noise to be added at each step.
    pub eta: f64,
    /// Adjust the indexes of the inference schedule by this value.
    pub steps_offset: usize,
    /// prediction type of the scheduler function
    pub prediction_type: PredictionType,
    /// number of diffusion steps used to train the model
    pub train_timesteps: usize,
}

impl Default for DDIMSchedulerConfig {
    fn default() -> Self {
        Self {
            beta_start: 0.00085f64,
            beta_end: 0.012f64,
            beta_schedule: BetaSchedule::ScaledLinear,
            eta: 0.,
            steps_offset: 1,
            prediction_type: PredictionType::Epsilon,
            train_timesteps: 1000,
        }
    }
}

#[derive(Module, Debug)]
pub struct DDIMScheduler<B: Backend> {
    timesteps: Vec<usize>,
    alphas_cumprod: Vec<f64>,
    step_ratio: usize,
    init_noise_sigma: f64,
    config: DDIMSchedulerConfig,
    __phantom: PhantomData<B>,
}

impl<B: Backend> DDIMScheduler<B> {
    pub fn new(device: &B::Device, inference_steps: usize, config: DDIMSchedulerConfig) -> Self {
        let step_ratio = config.train_timesteps / inference_steps;
        let timesteps = (0..inference_steps)
            .map(|s| s * step_ratio + config.steps_offset)
            .rev()
            .collect();
        let betas = match config.beta_schedule {
            BetaSchedule::Linear => linear_tensor::<B>(
                device,
                config.beta_start,
                config.beta_end,
                config.train_timesteps,
            ),
            BetaSchedule::ScaledLinear => scaled_linear_tensor::<B>(
                device,
                config.beta_start,
                config.beta_end,
                config.train_timesteps,
            ),
            BetaSchedule::SquaredcosCapV2 => {
                squared_cos_tensor::<B>(device, config.train_timesteps, 0.999)
            }
        };

        let betas_vec: Vec<B::FloatElem> = betas.to_data().value;
        let mut alphas_cumprod = Vec::with_capacity(betas_vec.len());

        for beta in &betas_vec {
            let alpha = 1.0 - beta.to_f64().expect("beta to be a float");
            alphas_cumprod.push(alpha * alphas_cumprod.last().copied().unwrap_or(1.0))
        }

        Self {
            timesteps,
            alphas_cumprod,
            step_ratio,
            init_noise_sigma: 1.0,
            config,
            __phantom: PhantomData,
        }
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps.as_slice()
    }
}

fn scaled_linear_tensor<B: Backend>(
    device: &B::Device,
    start: f64,
    end: f64,
    num_steps: usize,
) -> Tensor<B, 1> {
    linear_tensor(device, start.sqrt(), end.sqrt(), num_steps)
}

/// Creates a linear tensor (vector) with the values `start..end` evenly distributed
/// over `num_steps`
fn linear_tensor<B: Backend>(
    device: &B::Device,
    start: f64,
    end: f64,
    num_steps: usize,
) -> Tensor<B, 1> {
    let mut cur = start;
    let mut betas = Vec::with_capacity(num_steps);

    assert!(start < end);

    let step_size = (end - start) / num_steps as f64;

    assert!(step_size > 0.0);

    while cur < end {
        betas.push(cur.elem());
        cur += step_size;
    }
    Tensor::from_data_device(Data::new(betas, Shape::new([betas.len()])), device)
}

/// Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
/// `(1-beta)` over time from `t = [0,1]`.
///
/// Contains a function `alpha_bar` that takes an argument `t` and transforms it to the cumulative product of `(1-beta)`
/// up to that part of the diffusion process.
fn squared_cos_tensor<B: Backend>(
    device: &B::Device,
    num_diffusion_timesteps: usize,
    max_beta: f64,
) -> Tensor<B, 1> {
    let alpha_bar = |time_step: usize| {
        f64::cos((time_step as f64 + 0.008) / 1.008 * std::f64::consts::FRAC_PI_2).powi(2)
    };
    let mut betas = Vec::with_capacity(num_diffusion_timesteps);
    for i in 0..num_diffusion_timesteps {
        let t1 = i / num_diffusion_timesteps;
        let t2 = (i + 1) / num_diffusion_timesteps;
        betas.push((1.0 - alpha_bar(t2) / alpha_bar(t1)).min(max_beta).elem());
    }
    Tensor::from_data_device(Data::new(betas, Shape::new([betas.len()])), device)
}
