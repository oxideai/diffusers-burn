use num_traits::ToPrimitive;
use std::marker::PhantomData;

use burn::{
    module::Module,
    tensor::{backend::Backend, Data, Distribution, ElementConversion, Shape, Tensor},
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

    // Perform a backward step
    pub fn step<const D: usize>(
        &self,
        model_output: Tensor<B, D>,
        timestep: usize,
        sample: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let timestep = if timestep >= self.alphas_cumprod.len() {
            timestep - 1
        } else {
            timestep
        };
        let prev_timestep = if timestep > self.step_ratio {
            timestep - self.step_ratio
        } else {
            0
        };
        let alpha_prod_t = self.alphas_cumprod[timestep];
        let alpha_prod_t_prev = self.alphas_cumprod[prev_timestep];
        let beta_prod_t = 1. - alpha_prod_t;
        let beta_prod_t_prev = 1. - alpha_prod_t_prev;

        let (pred_original_sample, pred_epsilon) = match self.config.prediction_type {
            PredictionType::Epsilon => {
                let pred_original_sample = sample
                    .sub(model_output.clone().mul_scalar(beta_prod_t.sqrt()))
                    * (1. / alpha_prod_t.sqrt());
                (pred_original_sample, model_output)
            }
            PredictionType::VPrediction => {
                let pred_original_sample = sample.clone().mul_scalar(alpha_prod_t.sqrt())
                    - model_output.clone().mul_scalar(beta_prod_t.sqrt());
                let pred_epsilon = model_output.mul_scalar(alpha_prod_t.sqrt())
                    + sample.mul_scalar(beta_prod_t.sqrt());
                (pred_original_sample, pred_epsilon)
            }
            PredictionType::Sample => {
                let pred_epsilon = sample.sub(model_output.clone().mul_scalar(alpha_prod_t.sqrt()))
                    * (1. / beta_prod_t.sqrt());
                (model_output, pred_epsilon)
            }
        };

        let variance = (beta_prod_t_prev / beta_prod_t) * (1. - alpha_prod_t / alpha_prod_t_prev);
        let std_dev_t = self.config.eta * variance.sqrt();

        let pred_sample_direction =
            pred_epsilon.mul_scalar((1. - alpha_prod_t_prev - std_dev_t * std_dev_t).sqrt());
        let prev_sample =
            pred_original_sample.mul_scalar(alpha_prod_t_prev.sqrt()) + pred_sample_direction;

        if self.config.eta > 0. {
            prev_sample.clone()
                + Tensor::random_device(
                    prev_sample.shape(),
                    Distribution::Normal(0f64, std_dev_t as f64),
                    &prev_sample.device(),
                )
        } else {
            prev_sample
        }
    }

    pub fn add_noise<const D: usize>(
        &self,
        original: Tensor<B, D>,
        noise: Tensor<B, D>,
        timestep: usize,
    ) -> Tensor<B, D> {
        let timestep = if timestep >= self.alphas_cumprod.len() {
            timestep - 1
        } else {
            timestep
        };
        let sqrt_alpha_prod = self.alphas_cumprod[timestep].sqrt();
        let sqrt_one_minus_alpha_prod = (1.0 - self.alphas_cumprod[timestep]).sqrt();

        original.mul_scalar(sqrt_alpha_prod) + noise.mul_scalar(sqrt_one_minus_alpha_prod)
    }

    pub fn timesteps(&self) -> &[usize] {
        &self.timesteps.as_slice()
    }

    pub fn init_noise_sigma(&self) -> f64 {
        self.init_noise_sigma
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
    let dims = [betas.len()];

    Tensor::from_data_device(Data::new(betas, Shape::new(dims)), device)
}

/// Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
/// `(1-beta)` over time from `t = [0,1]`.
///
/// Contains a function `alpha_bar` that takes an argument `t` and transforms
/// it to the cumulative product of `(1-beta)` up to that part of the diffusion process.
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
    let dims = [betas.len()];

    Tensor::from_data_device(Data::new(betas, Shape::new(dims)), device)
}
