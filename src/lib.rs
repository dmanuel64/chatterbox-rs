mod audio;
mod chatterbox_turbo;
pub mod config;
#[cfg(feature = "download")]
pub mod downloader;
mod models;

pub type ChatterboxError = chatterbox_turbo::Error;

pub use chatterbox_turbo::{ChatterboxTurbo, ParalinguisticStrExt, ParalinguisticTag};
pub use models::Variant;
pub use ort::session::builder::AutoDevicePolicy;
