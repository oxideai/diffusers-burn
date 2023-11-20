//! Attention Based Building Blocks

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Config)]
pub struct GeGluConfig {
    dim_in: usize,
    dim_out: usize,
}

impl GeGluConfig {
    fn init<B: Backend>(&self) -> GeGlu<B> {
        let proj = LinearConfig::new(self.dim_in, self.dim_out * 2).init();
        GeGlu { proj }
    }
}

#[derive(Module, Debug)]
struct GeGlu<B: Backend> {
    proj: Linear<B>
}

impl<B: Backend> GeGlu<B> {
    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let projected = self.proj.forward(xs);
        let [n_batch, n_ctx, n_state] = projected.dims();

        let n_state_out = n_state / 2;

        let xs = projected
            .clone()
            .slice([0..n_batch, 0..n_ctx, 0..n_state_out]);
        let gate = projected.slice([0..n_batch, 0..n_ctx, n_state_out..n_state]);

        xs * gelu(gate)
    }
}

#[derive(Config)]
pub struct FeedForwardConfig {
    dim: usize,
    dim_out: Option<usize>,
    mult: usize,
}

impl FeedForwardConfig {
    fn init<B: Backend>(&self) -> FeedForward<B> {
        let inner_dim = self.dim * self.mult;
        let dim_out = self.dim_out.unwrap_or(self.dim);
        let proj_in = GeGluConfig::new(self.dim, inner_dim).init();
        let linear = LinearConfig::new(inner_dim, dim_out).init();
        FeedForward { proj_in, linear }
    }
}

#[derive(Module, Debug)]
struct FeedForward<B: Backend> {
    proj_in: GeGlu<B>,
    linear: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    fn forward(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let xs = self.proj_in.forward(xs);
        self.linear.forward(xs)
    }
}

#[derive(Config)]
pub struct CrossAttentionConfig {
    d_query: usize,
    d_context: Option<usize>,
    n_heads: usize,
    d_head: usize,
    slice_size: Option<usize>,
    use_flash_attn: bool,
}

#[derive(Module, Debug)]
struct CrossAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    n_heads: usize,
    scale: f64,
    slice_size: Option<usize>,
    use_flash_attn: bool,
}

impl CrossAttentionConfig {
    fn init<B: Backend>(&self) -> CrossAttention<B> {
        let linear = |in_dim: usize, out_dim: usize| {
            LinearConfig::new(in_dim, out_dim)
                .with_bias(false)
                .init()
        };

        let inner_dim = self.d_head * self.n_heads;
        let context_dim = self.d_context.unwrap_or(self.d_query);

        CrossAttention {
            query: linear(self.d_query, inner_dim),
            key: linear(context_dim, inner_dim),
            value: linear(context_dim, inner_dim),
            output: linear(inner_dim, self.d_query),
            n_heads: self.n_heads,
            scale: 1.0 / f64::sqrt(self.d_head as f64),
            slice_size: self.slice_size,
            use_flash_attn: self.use_flash_attn,
        }
    }
}


impl<B: Backend> CrossAttention<B> {
    fn reshape_heads_to_batch_dim(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = xs.dims();
        xs.reshape([batch_size, seq_len, self.n_heads, dim / self.n_heads])
            .swap_dims(1, 2)
            .reshape([batch_size * self.n_heads, seq_len, dim / self.n_heads])
    }

    fn reshape_batch_dim_to_heads(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = xs.dims();
        xs.reshape([batch_size / self.n_heads, self.n_heads, seq_len, dim])
            .swap_dims(1, 2)
            .reshape([batch_size / self.n_heads, seq_len, dim * self.n_heads])
    }
}

// trait TensorStack where Self: Sized {
//     fn stack<B: Backend, const D2: usize>(xs: &[Self], dim: usize) -> Tensor<B, D2>;
// }
//
// impl<B: Backend, const D: usize> TensorStack for Tensor<B, D> {
//     fn stack<C: Backend, const D2: usize>(xs: &[Self], dim: usize) -> Tensor<C, D2> {
//
//
//     }
// }