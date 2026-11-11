#![cfg_attr(docsrs, feature(doc_cfg))]

//! A Rust port of [Resemble AI's
//! Chatterbox](https://github.com/resemble-ai/chatterbox) text-to-speech /
//! voice-cloning pipeline, running the pipeline's components as exported ONNX
//! graphs through the `ort` crate.
//!
//! The main entry point is [`ChatterboxTurbo`]: [`ChatterboxTurbo::load`] to
//! load the default (CPU) model, then [`ChatterboxTurbo::generate`] (or one of
//! its `_with_*` convenience variants) to synthesize speech from text and a
//! reference voice clip.
//!
//! The ONNX Runtime environment is managed for you (see
//! [`default_environment`]). To configure it yourself, build one with
//! [`ort::init`] (re-exported as [`ort`], so the versions always match) and use
//! the `*_with_environment` loaders.

mod audio;
pub mod chatterbox_turbo;
pub mod config;
#[cfg(feature = "download")]
pub mod downloader;
mod models;

pub type ChatterboxError = chatterbox_turbo::Error;

pub use chatterbox_turbo::{
    ChatterboxTurbo, GenerateOptions, LoadOptions, ParalinguisticStrExt, ParalinguisticTag,
    VoiceConditioning, default_environment,
};
/// The `ort` crate this crate is built against, for configuring an
/// [`ort::environment::Environment`] or [`ort::session::builder::SessionBuilder`]s
/// without depending on (and version-matching) `ort` yourself.
pub use ort;
pub use models::{conditional_decoder, language_model, model, speech_encoder, token_embedder};
