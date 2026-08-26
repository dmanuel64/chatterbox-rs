mod audio;
pub mod chatterbox_turbo;
pub mod config;
#[cfg(feature = "download")]
pub mod downloader;
mod models;

pub type ChatterboxError = chatterbox_turbo::Error;

pub use chatterbox_turbo::{
    ChatterboxTurbo, GenerateOptions, LoadOptions, ParalinguisticStrExt, ParalinguisticTag,
};
pub use models::model;
pub use ort::session::builder::{AutoDevicePolicy, GraphOptimizationLevel};
