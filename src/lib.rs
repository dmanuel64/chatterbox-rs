//! A Rust port of [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox)
//! text-to-speech / voice-cloning pipeline, running the pipeline's components as exported ONNX
//! graphs through the `ort` crate rather than reimplementing the original PyTorch models.
//!
//! The main entry point is [`ChatterboxTurbo`]: [`ChatterboxTurbo::load`] to load the default
//! (`f32`) model, then [`ChatterboxTurbo::generate`] (or one of its `_with_*` convenience
//! variants) to synthesize speech from text and a reference voice clip.

mod audio;
pub mod chatterbox_turbo;
pub mod config;
#[cfg(feature = "download")]
pub mod downloader;
mod models;

/// Convenience alias for [`chatterbox_turbo::Error`].
pub type ChatterboxError = chatterbox_turbo::Error;

pub use chatterbox_turbo::{
    ChatterboxTurbo, GenerateOptions, LoadOptions, ParalinguisticStrExt, ParalinguisticTag,
};
pub use models::{conditional_decoder, language_model, model, speech_encoder, token_embedder};
