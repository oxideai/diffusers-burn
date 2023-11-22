//! Contrastive Language-Image Pre-Training
//!
//! Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! https://github.com/openai/CLIP

use std::cmp::max;
use std::f32::consts::SQRT_2;

use burn::config::Config;
use burn::tensor::activation::softmax;
use burn::tensor::{Data, ElementConversion, Shape};
use burn::{
    module::Module,
    nn,
    tensor::{
        activation::{gelu, sigmoid},
        backend::Backend,
        Int, Tensor,
    },
};

#[derive(Module, Debug, Clone, serde::Deserialize, serde::Serialize)]
pub enum Activation {
    Gelu,
    QuickGelu,
    GeluErf,
}

impl Activation {
    pub fn forward<B: Backend, const D: usize>(&self, xs: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::Gelu => gelu(xs),
            Activation::QuickGelu => xs.clone() * sigmoid(xs * 1.702f64),
            Activation::GeluErf => (xs.clone() * (Tensor::erf(xs / SQRT_2) + 1)) / 2,
        }
    }
}

#[derive(Config, Debug)]
pub struct ClipConfig {
    vocab_size: usize,
    embed_dim: usize,       // aka config.hidden_size
    activation: Activation, // aka config.hidden_act
    intermediate_size: usize,
    max_position_embeddings: usize,
    // The character to use for padding, use EOS when not set.
    pad_with: Option<String>,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[allow(dead_code)]
    projection_dim: usize,
}

impl ClipConfig {
    // The config details can be found in the "text_config" section of this json file:
    // https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
    pub fn v1_5() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    // https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/text_encoder/config.json
    pub fn v2_1() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1024,
            intermediate_size: 4096,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 23,
            num_attention_heads: 16,
            projection_dim: 512,
            activation: Activation::Gelu,
        }
    }

    // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder/config.json
    pub fn sdxl() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 768,
            intermediate_size: 3072,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 12,
            num_attention_heads: 12,
            projection_dim: 768,
            activation: Activation::QuickGelu,
        }
    }

    // https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/text_encoder_2/config.json
    pub fn sdxl2() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 1280,
            intermediate_size: 5120,
            max_position_embeddings: 77,
            pad_with: Some("!".to_string()),
            num_hidden_layers: 32,
            num_attention_heads: 20,
            projection_dim: 1280,
            activation: Activation::Gelu,
        }
    }

    fn init_text_embeddings<B: Backend>(&self) -> ClipTextEmbeddings<B> {
        let token_embedding = nn::EmbeddingConfig::new(self.vocab_size, self.embed_dim).init();
        let position_embedding =
            nn::EmbeddingConfig::new(self.max_position_embeddings, self.embed_dim).init();
        let position_ids = Tensor::arange(0..self.max_position_embeddings).unsqueeze();

        ClipTextEmbeddings {
            token_embedding,
            position_embedding,
            position_ids,
        }
    }

    fn init_attention<B: Backend>(&self) -> ClipAttention<B> {
        assert_eq!(
            self.embed_dim % self.num_attention_heads,
            0,
            "embed_dim {} must be a multiple of num_attention_heads {}",
            self.embed_dim,
            self.num_attention_heads
        );

        let embed_dim = self.embed_dim;
        let num_attention_heads = self.num_attention_heads;
        let k_proj = nn::LinearConfig::new(embed_dim, embed_dim)
            .with_bias(false)
            .init();
        let v_proj = nn::LinearConfig::new(embed_dim, embed_dim)
            .with_bias(false)
            .init();
        let q_proj = nn::LinearConfig::new(embed_dim, embed_dim)
            .with_bias(false)
            .init();
        let out_proj = nn::LinearConfig::new(embed_dim, embed_dim).init();
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);
        ClipAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        }
    }

    fn init_mlp<B: Backend>(&self) -> ClipMlp<B> {
        let fc1 = nn::LinearConfig::new(self.embed_dim, self.intermediate_size).init();
        let fc2 = nn::LinearConfig::new(self.intermediate_size, self.embed_dim).init();
        ClipMlp {
            fc1,
            fc2,
            activation: self.activation.clone(),
        }
    }

    fn init_encoder_layer<B: Backend>(&self) -> ClipEncoderLayer<B> {
        ClipEncoderLayer {
            self_attn: self.init_attention(),
            layer_norm1: nn::LayerNormConfig::new(self.embed_dim).init(),
            mlp: self.init_mlp(),
            layer_norm2: nn::LayerNormConfig::new(self.embed_dim).init(),
        }
    }

    fn init_encoder<B: Backend>(&self) -> ClipEncoder<B> {
        let mut layers: Vec<ClipEncoderLayer<B>> = Vec::new();
        for _index in 0..self.num_hidden_layers {
            let layer = self.init_encoder_layer();
            layers.push(layer)
        }
        ClipEncoder { layers }
    }

    pub fn init_text_transformer<B: Backend>(&self) -> ClipTextTransformer<B> {
        let embeddings = self.init_text_embeddings();
        let encoder = self.init_encoder();
        let final_layer_norm = nn::LayerNormConfig::new(self.embed_dim).init();
        ClipTextTransformer {
            embeddings,
            encoder,
            final_layer_norm,
        }
    }
}

// CLIP Text Model
// https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py
#[derive(Module, Debug)]
struct ClipTextEmbeddings<B: Backend> {
    token_embedding: nn::Embedding<B>,
    position_embedding: nn::Embedding<B>,
    position_ids: Tensor<B, 2, Int>,
}

impl<B: Backend> ClipTextEmbeddings<B> {
    fn forward(&self, xs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let token_embedding = self.token_embedding.forward(xs);
        let position_embedding = self.position_embedding.forward(self.position_ids.clone());
        token_embedding + position_embedding
    }
}

#[derive(Module, Debug)]
struct ClipAttention<B: Backend> {
    k_proj: nn::Linear<B>,
    v_proj: nn::Linear<B>,
    q_proj: nn::Linear<B>,
    out_proj: nn::Linear<B>,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl<B: Backend> ClipAttention<B> {
    fn shape(&self, xs: Tensor<B, 3>, seq_len: usize, bsz: usize) -> Tensor<B, 4> {
        xs.reshape([bsz, seq_len, self.num_attention_heads, self.head_dim])
            .swap_dims(1, 2)
    }

    fn forward(&self, xs: Tensor<B, 3>, causal_attention_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let [bsz, seq_len, embed_dim] = xs.dims();
        let query_states = self.q_proj.forward(xs.clone()) * self.scale;
        let proj_shape = [bsz * self.num_attention_heads, seq_len, self.head_dim];

        let query_states = self.shape(query_states, seq_len, bsz).reshape(proj_shape);
        let key_states = self
            .shape(self.k_proj.forward(xs.clone()), seq_len, bsz)
            .reshape(proj_shape);
        let value_states = self
            .shape(self.v_proj.forward(xs), seq_len, bsz)
            .reshape(proj_shape);
        let src_len = key_states.dims()[1];
        let attn_weights = query_states.matmul(key_states.swap_dims(1, 2));

        let attn_weights = attn_weights
            .reshape([bsz, self.num_attention_heads, seq_len, src_len])
            .add(causal_attention_mask);
        let attn_weights = attn_weights.reshape([bsz * self.num_attention_heads, seq_len, src_len]);
        let attn_weights = softmax(attn_weights, 3);

        let attn_output = attn_weights.matmul(value_states);
        let attn_output = attn_output
            .reshape([bsz, self.num_attention_heads, seq_len, self.head_dim])
            .swap_dims(1, 2)
            .reshape([bsz, seq_len, embed_dim]);

        self.out_proj.forward(attn_output)
    }
}

#[derive(Module, Debug)]
struct ClipMlp<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    activation: Activation,
}

impl<B: Backend> ClipMlp<B> {
    fn forward<const D: usize>(&self, xs: Tensor<B, D>) -> Tensor<B, D> {
        let xs = self.fc1.forward(xs);
        self.fc2.forward(self.activation.forward(xs))
    }
}

#[derive(Module, Debug)]
struct ClipEncoderLayer<B: Backend> {
    self_attn: ClipAttention<B>,
    layer_norm1: nn::LayerNorm<B>,
    mlp: ClipMlp<B>,
    layer_norm2: nn::LayerNorm<B>,
}

impl<B: Backend> ClipEncoderLayer<B> {
    fn forward(&self, xs: Tensor<B, 3>, causal_attention_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let residual = xs;
        let xs = self.layer_norm1.forward(residual.clone());
        let xs = self.self_attn.forward(xs, causal_attention_mask);
        let xs2 = xs.clone() + residual;

        let residual = xs2;
        let xs = self.layer_norm2.forward(xs.clone());
        let xs = self.mlp.forward(xs);
        xs + residual
    }
}

#[derive(Module, Debug)]
struct ClipEncoder<B: Backend> {
    layers: Vec<ClipEncoderLayer<B>>,
}

impl<B: Backend> ClipEncoder<B> {
    fn forward(&self, xs: Tensor<B, 3>, causal_attention_mask: Tensor<B, 4>) -> Tensor<B, 3> {
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(xs, causal_attention_mask.clone());
        }
        xs
    }
}

/// A CLIP transformer based model.
#[derive(Module, Debug)]
pub struct ClipTextTransformer<B: Backend> {
    embeddings: ClipTextEmbeddings<B>,
    encoder: ClipEncoder<B>,
    final_layer_norm: nn::LayerNorm<B>,
}

impl<B: Backend> ClipTextTransformer<B> {
    // https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
    fn build_causal_attention_mask(bsz: usize, seq_len: usize, device: &B::Device) -> Tensor<B, 4> {
        let mut mask_vec = Vec::with_capacity(seq_len * seq_len);
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_vec.push(f32::MIN.elem());
                } else {
                    mask_vec.push(0f32.elem());
                }
            }
        }

        let mask_data: Data<B::FloatElem, 2> = Data::new(mask_vec, Shape::new([seq_len, seq_len]));
        Tensor::from_data_device(mask_data, device)
            .unsqueeze::<3>()
            .repeat(0, bsz)
            .unsqueeze_dim(1)
    }

    fn forward(&self, xs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [bsz, seq_len] = xs.dims();
        let xs = self.embeddings.forward(xs);
        let causal_attention_mask = Self::build_causal_attention_mask(bsz, seq_len, &xs.device());
        let xs = self.encoder.forward(xs, causal_attention_mask);
        self.final_layer_norm.forward(xs)
    }
}

fn zero_lower_diagonal<B: Backend>(xs: Tensor<B, 3>) -> Tensor<B, 3> {
    let [m, n, _] = xs.dims();

    // build an upper-triangle matrix
    let upper_diag = (0..max(m, n))
        .map(Tensor::<B, 2, Int>::diagonal)
        .fold(Tensor::zeros([max(m, n); 2]), Tensor::add);

    upper_diag.reshape([m, n]).unsqueeze().float().mul(xs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Data, Shape};

    #[test]
    fn test_init_text_embeddings() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let clip_config = ClipConfig::v1_5();
        let text_embeddings: ClipTextEmbeddings<TestBackend> = clip_config.init_text_embeddings();

        assert_eq!(
            text_embeddings.position_ids.to_data(),
            Data::from([[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
                65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76
            ]])
        );

        assert_eq!(text_embeddings.position_ids.shape(), Shape::from([1, 77]));
    }

    #[test]
    fn test_clip_attention_shape() {
        type TestBackend = burn_ndarray::NdArray<f32>;

        let clip_config = ClipConfig::v1_5();
        let clip_attention: ClipAttention<TestBackend> = clip_config.init_attention();

        let xs = Tensor::<TestBackend, 3>::zeros([2, 77, 768]);
        let xs = clip_attention.shape(xs, 77, 2);

        // TODO: Test contiguous
        assert_eq!(xs.shape(), Shape::from([2, 12, 77, 64]));
    }

    #[test]
    fn build_causal_attention_mask() {
        type TestBackend = burn_ndarray::NdArray<f32>;
        let device = <TestBackend as Backend>::Device::default();

        let mask = ClipTextTransformer::<TestBackend>::build_causal_attention_mask(2, 4, &device);
        assert_eq!(mask.shape(), Shape::from([2, 1, 4, 4]));

        mask.to_data().assert_approx_eq(
            &Data::from([
                [[
                    [0.0000e0, f32::MIN, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, 0.0000e0],
                ]],
                [[
                    [0.0000e0, f32::MIN, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, f32::MIN, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, f32::MIN],
                    [0.0000e0, 0.0000e0, 0.0000e0, 0.0000e0],
                ]],
            ]),
            3,
        );
    }
}
