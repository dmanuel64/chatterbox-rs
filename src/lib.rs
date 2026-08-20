mod audio;
pub mod config;
mod s3gen;
mod t3;
mod tokenizer;
mod voice_encoder;
#[cfg(feature = "watermark")]
mod watermark;

use ort::device::Device;
use std::path::Path;
pub use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {}

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
