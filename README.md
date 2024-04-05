# diffusers-burn: A diffusers API in Rust/Burn

> **⚠️ This is still in development - contributors welcome!**

The `diffusers-burn` crate is a conversion of [diffusers-rs](https://github.com/LaurentMazare/diffusers-rs) using [burn](https://github.com/burn-rs/burn) rather than libtorch. This implementation supports Stable Diffusion v1.5, v2.1, as well as Stable Diffusion XL 1.0.

<div align="left" valign="middle">
<a href="https://runblaze.dev">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://www.runblaze.dev/logo_dark.png">
   <img align="right" src="https://www.runblaze.dev/logo_light.png" height="102px"/>
 </picture>
</a>

<br style="display: none;"/>

_[Blaze](https://runblaze.dev) supports this project by providing ultra-fast Apple Silicon macOS Github Action Runners. Apply the discount code `BURN50` at checkout to enjoy 50% off your first year._

</div>

## Feature Flags

This crate can be used without the standard library (`#![no_std]`) with `alloc` by disabling
the default `std` feature.

* `std` - enables the standard library. Enabled by default.
* `wgpu` - uses ndarray as the backend. Enabled by default when none specified and `std`.
* `ndarray` - uses ndarray as the backend.
* `ndarray-no-std` - uses ndarray-no-std as the backend. Enabled by default when none and `#![no_std]`.
* `ndarray-blas-accelerate` - uses ndarray with Accelerate framework (macOS only).
* `torch` - uses torch as the backend.

## Community

If you are excited about the project or want to contribute, don't hesitate to join our [Discord](https://discord.gg/UHtSgF6j5J)!
We try to be as welcoming as possible to everybody from any background. We're still building this out, but you can ask your questions there!

## Status

diffusers-burn is currently in active development, and is not yet complete.

## License

diffusers-burn is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
See [LICENSE-APACHE](./LICENSE-APACHE) and [LICENSE-MIT](./LICENSE-MIT) for details. Opening a pull
request is assumed to signal agreement with these licensing terms.
