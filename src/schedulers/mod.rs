pub mod ddim;

/// This represents how beta ranges from its minimum value to the maximum
/// during training.
#[derive(Debug, Clone, Copy)]
pub enum BetaSchedule {
    /// Linear interpolation.
    Linear,
    /// Linear interpolation of the square root of beta.
    ScaledLinear,
    /// Glide cosine schedule
    SquaredcosCapV2,
}

/// prediction type of the scheduler function, one of `epsilon` (predicting
/// the noise of the diffusion process), `sample` (directly predicting the noisy sample`)
/// or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)
#[derive(Debug, Clone, Copy)]
pub enum PredictionType {
    Epsilon,
    VPrediction,
    Sample,
}
