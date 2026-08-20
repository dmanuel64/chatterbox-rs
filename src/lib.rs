pub mod config;
#[cfg(feature = "download")]
pub mod downloader;
mod onnx;
mod s3gen;
mod t3;
mod tokenizers;
mod voice_encoder;

use ort::device::Device;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub use onnx::Variant;

#[derive(Debug, Error)]
pub enum Error {
    #[error("An ONNX runtime error occurred: {0}")]
    Onnx(#[source] ort::Error),
}

pub type ChatterboxError = Error;

pub struct ChatterboxTts {
    tokenizer: Tokenizer,
    voice_encoder: VoiceEncoder,
    t3: T3Session,
    s3gen: S3Gen,
    sample_rate: u32,
}

pub struct GenerateOptions {
    pub exaggeration: f32,
    pub cfg_weight: f32,
    pub temperature: f32,
}

impl ChatterboxTts {
    pub fn load(device: Device) -> Result<Self, ChatterboxError> {
        todo!()
    }

    pub fn generate(
        &mut self,
        text: &str,
        reference_audio_path: impl AsRef<Path>,
        opts: GenerateOptions,
    ) -> Result<Vec<f32>, ChatterboxError> {
        todo!()
    }
}
