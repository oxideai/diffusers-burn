use burn::config::Config;
use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Configuration to create a [GroupNorm](GroupNorm) layer.
#[derive(Config)]
pub struct GroupNormConfig {
    /// The number of groups to separate the channels into
    num_groups: usize,
    /// The number of channels expected in the input
    num_channels: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    epsilon: f64,
    /// A boolean value that when set to `true`, this module has learnable
    /// per-channel affine parameters initialized to ones (for weights)
    /// and zeros (for biases). Default: `true`
    #[config(default = true)]
    affine: bool,
}

/// Applies Group Normalization over a mini-batch of inputs.
///
/// `Y = groupnorm(X) * γ + β`
#[derive(Module, Debug)]
pub struct GroupNorm<B: Backend> {
    num_groups: usize,
    num_channels: usize,
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    epsilon: f64,
    affine: bool,
}

impl GroupNormConfig {
    /// Initialize a new [group norm](GroupNorm) module.
    pub fn init<B: Backend>(&self) -> GroupNorm<B> {
        assert_eq!(
            self.num_channels % self.num_groups,
            0,
            "The number of channels must be divisible by the number of groups"
        );

        let gamma = Tensor::ones([self.num_channels]).into();
        let beta = Tensor::zeros([self.num_channels]).into();

        GroupNorm {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            gamma,
            beta,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }

    /// Initialize a new [group norm](GroupNorm) module with a [record](GroupNormRecord).
    pub fn init_with<B: Backend>(&self, record: GroupNormRecord<B>) -> GroupNorm<B> {
        GroupNorm {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            gamma: record.gamma,
            beta: record.beta,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

impl<B: Backend> GroupNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let shape = input.shape();
        if shape.num_elements() <= 2 {
            panic!(
                "input rank for GroupNorm should be at least 3, but got {}",
                shape.num_elements()
            );
        }

        let batch_size = shape.dims[0];
        let num_channels = shape.dims[1];

        if num_channels != self.num_channels {
            panic!(
                "expected {} channels but got {}",
                self.num_channels, num_channels
            );
        }

        let hidden_size =
            shape.dims[D - 1..].iter().product::<usize>() * num_channels / self.num_groups;
        let input = input.reshape([batch_size, self.num_groups, hidden_size]);

        let mean = input.clone().sum_dim(D - 1) / hidden_size as f64;
        let var = input.clone().sqrt().sum_dim(D - 1) / hidden_size as f64;
        let input_normalized = input.sub(mean).div(var.sqrt().add_scalar(self.epsilon));

        if self.affine {
            let mut affine_shape = [1; D];
            affine_shape[1] = num_channels;

            input_normalized
                .reshape(shape)
                .mul(self.gamma.val().reshape(affine_shape))
                .add(self.beta.val().reshape(affine_shape))
        } else {
            input_normalized.reshape(shape)
        }
    }
}
