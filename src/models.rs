//! Chatterbox ONNX graphs. See [`model`] for the shared
//! [`model::Model`]/[`model::Metadata`] traits and precision bounds these modules build on.

pub mod conditional_decoder;
pub mod language_model;
pub mod model;
pub mod speech_encoder;
pub mod token_embedder;
