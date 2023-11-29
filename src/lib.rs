//! # Diffusion pipelines and models
//!
//! This is a Rust port of Hugging Face's [diffusers](https://github.com/huggingface/diffusers) Python api using [Burn](https://github.com/burn-rs/burn)

#![cfg_attr(not(feature = "std"), no_std)]

pub mod models;
pub mod pipelines;
pub mod schedulers;
pub mod transformers;
pub mod utils;

extern crate alloc;

#[cfg(all(test, not(feature = "wgpu"), not(feature = "torch")))]
pub type TestBackend = burn_ndarray::NdArray<f32>;

#[cfg(all(test, feature = "torch"))]
pub type TestBackend = burn_tch::LibTorch<f32>;

#[cfg(all(test, feature = "wgpu", not(target_os = "macos")))]
pub type TestBackend = burn_wgpu::Wgpu<burn_wgpu::Vulkan, f32, i32>;

#[cfg(all(test, feature = "wgpu", target_os = "macos"))]
pub type TestBackend = burn_wgpu::Wgpu<burn_wgpu::Metal, f32, i32>;
