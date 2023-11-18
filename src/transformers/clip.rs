//! Contrastive Language-Image Pre-Training
//!
//! Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! https://github.com/openai/CLIP

use std::alloc::GlobalAlloc;
use std::cmp::max;

use burn::tensor::activation::softmax;
use burn::tensor::ops::TensorOps;
use burn::{
    module::Module,
    nn,
    tensor::{
        activation::{gelu, sigmoid},
        backend::Backend,
        Int, Tensor,
    },
};

#[derive(Module, Debug, Clone, Copy)]
pub enum Activation {
    QuickGelu,
    Gelu,
}

impl Activation {
    pub fn forward<B: Backend, const D: usize>(&self, xs: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::QuickGelu => xs.clone() * sigmoid(xs * 1.702f64),
            Activation::Gelu => gelu(xs),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
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

impl Config {
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

    pub fn ssd1b() -> Self {
        Self::sdxl()
    }

    pub fn ssd1b2() -> Self {
        Self::sdxl2()
    }
}

// CLIP Text Model
// https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py
#[derive(Module, Debug)]
struct ClipTextEmbeddings<B: Backend> {
    token_embedding: nn::Embedding<B>,
    position_embedding: nn::Embedding<B>,
    position_ids: Tensor<B, 2>,
}

impl<B: Backend> ClipTextEmbeddings<B> {
    fn new(device: &B::Device, c: &Config) -> Self {
        let token_embedding = nn::EmbeddingConfig::new(c.vocab_size, c.embed_dim).init();
        let position_embedding =
            nn::EmbeddingConfig::new(c.max_position_embeddings, c.embed_dim).init();
        let position_ids = Tensor::arange_device(0..c.max_position_embeddings, device)
            .unsqueeze()
            .float();

        ClipTextEmbeddings {
            token_embedding,
            position_embedding,
            position_ids,
        }
    }

    pub fn forward(&self, xs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let token_embedding = self.token_embedding.forward(xs);
        let position_embedding = self
            .position_embedding
            .forward(self.position_ids.to_owned().int());
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
    fn new(c: &Config) -> Self {
        assert_eq!(
            c.embed_dim % c.num_attention_heads,
            0,
            "embed_dim {} must be a multiple of num_attention_heads {}",
            c.embed_dim,
            c.num_attention_heads
        );

        let embed_dim = c.embed_dim;
        let num_attention_heads = c.num_attention_heads;
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

    fn shape(&self, xs: Tensor<B, 3>, seq_len: usize, bsz: usize) -> Tensor<B, 4> {
        xs.reshape([bsz, seq_len, self.num_attention_heads, self.head_dim])
            .swap_dims(1, 2)
        // .contiguous() // TODO: Figure out if this is needed or if we can abstract over memory
    }

    pub fn forward(&self, xs: Tensor<B, 3>, causal_attention_mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        let [bsz, seq_len, embed_dim] = xs.dims();
        let query_states = self.q_proj.forward(xs.clone()) * self.scale;
        let proj_shape = [bsz * self.num_attention_heads, seq_len, self.head_dim];

        let query_states = self
            .shape(query_states, seq_len, bsz)
            .reshape(proj_shape)
            .to_full_precision();
        let key_states = self
            .shape(self.k_proj.forward(xs.clone()), seq_len, bsz)
            .reshape(proj_shape)
            .to_full_precision();
        let value_states = self
            .shape(self.v_proj.forward(xs), seq_len, bsz)
            .reshape(proj_shape)
            .to_full_precision();
        let src_len = key_states.dims()[1];
        let attn_weights = query_states.matmul(key_states.swap_dims(1, 2));

        let attn_weights = attn_weights
            .reshape([bsz, self.num_attention_heads, seq_len, src_len])
            .add(
                causal_attention_mask
                    .to_owned()
                    .unsqueeze::<4>()
                    .to_full_precision(),
            );
        let attn_weights = attn_weights.reshape([bsz * self.num_attention_heads, seq_len, src_len]);
        let attn_weights = softmax(attn_weights, 3);

        let attn_output = attn_weights.matmul(value_states);
        let attn_output = attn_output
            .reshape([bsz, self.num_attention_heads, seq_len, self.head_dim])
            .swap_dims(1, 2)
            .reshape([bsz, seq_len, embed_dim]);

        self.out_proj
            .forward(Tensor::from_full_precision(attn_output))
    }
}

#[derive(Module, Debug)]
struct ClipMlp<B: Backend> {
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    activation: Activation,
}

impl<B: Backend> ClipMlp<B> {
    fn new(c: &Config) -> Self {
        let fc1 = nn::LinearConfig::new(c.embed_dim, c.intermediate_size).init();
        let fc2 = nn::LinearConfig::new(c.intermediate_size, c.embed_dim).init();
        ClipMlp {
            fc1,
            fc2,
            activation: c.activation,
        }
    }

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
    fn new(c: &Config) -> Self {
        let self_attn = ClipAttention::new(c);
        let layer_norm1 = nn::LayerNormConfig::new(c.embed_dim).init();
        let mlp = ClipMlp::new(c);
        let layer_norm2 = nn::LayerNormConfig::new(c.embed_dim).init();
        ClipEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        }
    }

    pub fn forward(&self, xs: Tensor<B, 3>, causal_attention_mask: &Tensor<B, 2>) -> Tensor<B, 3> {
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
    fn new(c: &Config) -> Self {
        let mut layers: Vec<ClipEncoderLayer<B>> = Vec::new();
        for _index in 0..c.num_hidden_layers {
            let layer = ClipEncoderLayer::new(c);
            layers.push(layer)
        }
        ClipEncoder { layers }
    }

    pub fn forward(&self, xs: Tensor<B, 3>, causal_attention_mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(xs, causal_attention_mask);
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
    pub fn new(device: &B::Device, c: &Config) -> Self {
        let embeddings = ClipTextEmbeddings::new(device, c);
        let encoder = ClipEncoder::new(c);
        let final_layer_norm = nn::LayerNormConfig::new(c.embed_dim).init();
        ClipTextTransformer {
            embeddings,
            encoder,
            final_layer_norm,
        }
    }

    // https://github.com/huggingface/transformers/blob/674f750a57431222fa2832503a108df3badf1564/src/transformers/models/clip/modeling_clip.py#L678
    fn build_causal_attention_mask(bsz: usize, seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
        let mask = Tensor::full_device([bsz, seq_len, seq_len], f32::MIN, device);
        let mask = zero_lower_diagonal(mask); // zero out the lower diagonal
        let mask = mask.unsqueeze(); // expand mask
        mask
    }

    fn forward(&self, xs: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [bsz, seq_len] = xs.dims();
        let xs = self.embeddings.forward(xs);
        let causal_attention_mask = Self::build_causal_attention_mask(bsz, seq_len, &xs.device());
        let xs = self.encoder.forward(xs, &causal_attention_mask);
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
