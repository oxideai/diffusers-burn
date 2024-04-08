//! # Diffusion pipelines and models
//!
//! This is a Rust port of Hugging Face's [diffusers](https://github.com/huggingface/diffusers) Python api using [Burn](https://github.com/burn-rs/burn)

#![cfg_attr(not(feature = "std"), no_std)]

pub mod models;
pub mod pipelines;
pub mod transformers;
pub mod utils;

#[cfg(all(test, feature = "ndarray"))]
use burn::backend::ndarray;

#[cfg(all(test, feature = "torch"))]
use burn::backend::libtorch;

#[cfg(all(test, feature = "wgpu"))]
use burn::backend::wgpu;

extern crate alloc;

#[cfg(all(test, feature = "ndarray"))]
pub type TestBackend = ndarray::NdArray<f32>;

#[cfg(all(test, feature = "torch"))]
pub type TestBackend = libtorch::LibTorch<f32>;

#[cfg(all(test, feature = "wgpu", not(target_os = "macos")))]
pub type TestBackend = wgpu::Wgpu<wgpu::Vulkan, f32, i32>;

#[cfg(all(test, feature = "wgpu", target_os = "macos"))]
pub type TestBackend = wgpu::Wgpu<wgpu::Metal, f32, i32>;
