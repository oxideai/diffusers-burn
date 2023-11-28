//! Attention Based Building Blocks

use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::activation::{gelu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

#[derive(Config)]
pub struct GeGluConfig {
    /// The size of the input features.
    d_input: usize,
    /// The size of the output features.
    d_output: usize,
}

/// A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.
#[derive(Module, Debug)]
struct GeGlu<B: Backend> {
    proj: Linear<B>,
}

impl GeGluConfig {
    fn init<B: Backend>(&self) -> GeGlu<B> {
        let proj = LinearConfig::new(self.d_input, 2 * self.d_output).init();
        GeGlu { proj }
    }
}

impl<B: Backend> GeGlu<B> {
    fn forward<const D: usize>(&self, xs: Tensor<B, D>) -> Tensor<B, D> {
        let hidden_states_and_gate = self.proj.forward(xs).chunk(2, D - 1);
        hidden_states_and_gate[0].clone() * gelu(hidden_states_and_gate[1].clone())
    }
}

#[derive(Config)]
pub struct FeedForwardConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features. If not given defaults to `d_input`.
    d_output: Option<usize>,
    /// The multiplier to use for the hidden dimension.
    #[config(default = 4)]
    multiplier: usize,
    /// The dropout probability. Default: 0.0
    #[config(default = 0.)]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    geglu: GeGlu<B>,
    dropout: Dropout,
    linear_outer: Linear<B>,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self) -> FeedForward<B> {
        let inner_dim = self.d_input * self.multiplier;
        let dim_out = self.d_output.unwrap_or(self.d_input);

        FeedForward {
            geglu: GeGluConfig {
                d_input: self.d_input,
                d_output: inner_dim,
            }
            .init(),
            linear_outer: LinearConfig::new(inner_dim, dim_out).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> FeedForward<B> {
    pub fn forward<const D: usize>(&self, xs: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.geglu.forward(xs);
        let x = self.dropout.forward(x);
        self.linear_outer.forward(x)
    }
}

#[derive(Config)]
pub struct CrossAttentionConfig {
    /// The number of channels in the query.
    query_dim: usize,
    /// The number of channels in the context. If not given defaults to `query_dim`.
    context_dim: Option<usize>,
    /// The number of heads to use for the multi-head attention.
    #[config(default = 8)]
    n_heads: usize,
    /// The number of channels in each head.
    #[config(default = 64)]
    head_dim: usize,
    /// The size of the slices to use for the multi-head attention.
    slice_size: Option<usize>,
    #[config(default = 0.)]
    // The dropout probability.
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct CrossAttention<B: Backend> {
    query: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    n_heads: usize,
    scale: f64,
    slice_size: Option<usize>,
}

impl CrossAttentionConfig {
    pub fn init<B: Backend>(&self) -> CrossAttention<B> {
        let inner_dim = self.head_dim * self.n_heads;
        let context_dim = self.context_dim.unwrap_or(self.query_dim);
        let scale = 1. / (self.head_dim as f64).sqrt();

        CrossAttention {
            query: LinearConfig::new(self.query_dim, inner_dim)
                .with_bias(false)
                .init(),
            key: LinearConfig::new(context_dim, inner_dim)
                .with_bias(false)
                .init(),
            value: LinearConfig::new(context_dim, inner_dim)
                .with_bias(false)
                .init(),
            output: LinearConfig::new(inner_dim, self.query_dim).init(),
            n_heads: self.n_heads,
            scale,
            slice_size: self.slice_size,
        }
    }
}

impl<B: Backend> CrossAttention<B> {
    fn reshape_heads_to_batch_dim<const D: usize>(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = xs.dims();
        xs.reshape([batch_size, seq_len, self.n_heads, dim / self.n_heads])
            .swap_dims(1, 2)
            .reshape([batch_size * self.n_heads, seq_len, dim / self.n_heads])
    }

    fn reshape_batch_dim_to_heads(&self, xs: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, dim] = xs.dims();
        let output = xs
            .reshape([batch_size / self.n_heads, self.n_heads, seq_len, dim])
            .swap_dims(1, 2)
            .reshape([batch_size / self.n_heads, seq_len, dim * self.n_heads]);
        output
    }

    fn sliced_attention(
        &self,
        query: Tensor<B, 3>,
        key: Tensor<B, 3>,
        value: Tensor<B, 3>,
        slice_size: usize,
    ) -> Tensor<B, 3> {
        let batch_size_attention = query.clone().shape().dims[0];
        let mut hidden_states = Vec::with_capacity(batch_size_attention / slice_size);

        for i in 0..batch_size_attention / slice_size {
            let start_idx = i * slice_size;
            let end_idx = (i + 1) * slice_size;

            let xs = query
                .clone()
                .slice([start_idx..end_idx, 0..query.shape().dims[1]])
                .matmul(
                    key.clone()
                        .slice([start_idx..end_idx, 0..key.shape().dims[1]])
                        .swap_dims(3 - 1, 3 - 2)
                        * self.scale,
                );

            let xs = softmax(xs, 3 - 1).matmul(
                value
                    .clone()
                    .slice([start_idx..end_idx, 0..value.shape().dims[1]]),
            );

            hidden_states.push(xs);
        }

        let output = Tensor::cat(hidden_states, 0);
        self.reshape_batch_dim_to_heads(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{Param, ParamId};
    use burn::tensor::{Data, Distribution, Shape};

    #[test]
    fn test_geglu_tensor_shape_3() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let weight = Tensor::from_data(Data::from([
            [
                0.1221, 2.0378, -0.1171, 1.3004, -0.9630, -0.3108, -1.3376, -1.0593,
            ],
            [
                0.4669, -0.8146, 0.9965, -0.4659, 2.0444, -0.0709, -0.0147, 0.2135,
            ],
        ]));
        let bias = Tensor::from_data(Data::from([
            0.2867778149426027,
            0.6646517317105776,
            0.023946332404821136,
            -0.1395737454364393,
            0.05131041098737321,
            -0.4225726694675192,
            0.036411720220954735,
            0.01829268669677364,
        ]));

        let geglu = GeGlu {
            proj: Linear {
                weight: Param::new(ParamId::new(), weight),
                bias: Some(Param::new(ParamId::new(), bias)),
            },
        };

        let tensor: Tensor<TestBackend, 3> = Tensor::from_data(Data::from([
            [[1., 2.], [3., 4.], [5., 6.]],
            [[7., 8.], [9., 10.], [11., 12.]],
        ]));

        let output = geglu.forward(tensor);
        assert_eq!(output.shape(), Shape::from([2, 3, 4]));
        output.to_data().assert_approx_eq(
            &Data::from([
                [
                    [4.2632e0, -1.7927e-1, -2.3216e-1, -3.7916e-2],
                    [1.3460e1, -2.9266e-1, -2.1707e-4, -4.5595e-2],
                    [2.7750e1, -1.1442e-1, -2.5335e-13, -2.5403e-4],
                ],
                [
                    [4.7135e1, -1.7708e-2, -0.0000e0, -6.7097e-9],
                    [7.1616e1, -1.0652e-3, -0.0000e0, -0.0000e0],
                    [1.0119e2, -2.1943e-5, -0.0000e0, -0.0000e0],
                ],
            ]),
            2,
        );
    }

    #[test]
    fn test_geglu_tensor_shape_2() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let weight = Tensor::from_data(Data::from([
            [0.6054, 1.9322, 0.1445, 1.3004, -0.6853, -0.8947],
            [-0.3678, 0.4081, -1.9001, -1.5843, -0.9399, 0.1018],
        ]));
        let bias = Tensor::from_data(Data::from([
            0.3237631905393836,
            0.22052049807936902,
            -0.3196353346822061,
            -0.02244043444199162,
            -0.33600250665852865,
            0.5259391939301621,
        ]));

        let geglu = GeGlu {
            proj: Linear {
                weight: Param::new(ParamId::new(), weight),
                bias: Some(Param::new(ParamId::new(), bias)),
            },
        };

        let tensor: Tensor<TestBackend, 2> =
            Tensor::from_data(Data::from([[1., 2.], [3., 4.], [5., 6.]]));

        let output = geglu.forward(tensor);
        assert_eq!(output.shape(), Shape::from([3, 3]));
        output.to_data().assert_approx_eq(
            &Data::from([
                [-2.4192e-5, -3.3057e-2, 2.8535e-1],
                [-0.0000e0, -2.0983e-7, 5.2465e-1],
                [-0.0000e0, -0.0000e0, 1.2599e-2],
            ]),
            1,
        );
    }

    #[test]
    fn test_sliced_attention() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        // create tensor of size [2, 4, 2]
        let query: Tensor<TestBackend, 3> = Tensor::from_data(Data::from([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
            [[25.0, 26.0], [27.0, 28.0], [29.0, 30.0], [31.0, 32.0]],
        ]));
        let key: Tensor<TestBackend, 3> = Tensor::from_data(Data::from([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
            [[25.0, 26.0], [27.0, 28.0], [29.0, 30.0], [31.0, 32.0]],
        ]));
        let value: Tensor<TestBackend, 3> = Tensor::from_data(Data::from([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0], [13.0, 14.0], [15.0, 16.0]],
            [[17.0, 18.0], [19.0, 20.0], [21.0, 22.0], [23.0, 24.0]],
            [[25.0, 26.0], [27.0, 28.0], [29.0, 30.0], [31.0, 32.0]],
        ]));

        let cross_attention = CrossAttentionConfig {
            query_dim: 320,
            context_dim: None,
            n_heads: 2,
            head_dim: 40,
            slice_size: Some(2),
            dropout: 0.,
        }
        .init();

        let output = cross_attention.sliced_attention(query, key, value, 2);

        assert_eq!(output.shape(), Shape::from([2, 4, 4]));
        output.into_data().assert_approx_eq(
            &Data::from([
                [
                    [5.9201, 6.9201, 14.9951, 15.9951],
                    [6.7557, 7.7557, 14.9986, 15.9986],
                    [6.9363, 7.9363, 14.9996, 15.9996],
                    [6.9824, 7.9824, 14.9999, 15.9999],
                ],
                [
                    [23.0000, 24.0000, 31.0000, 32.0000],
                    [23.0000, 24.0000, 31.0000, 32.0000],
                    [23.0000, 24.0000, 31.0000, 32.0000],
                    [23.0000, 24.0000, 31.0000, 32.0000],
                ],
            ]),
            3,
        )
    }
}
