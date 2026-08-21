use std::path::Path;

use ort::device::Device;
use thiserror::Error;

use crate::onnx::{ConditionalDecoder, LanguageModel, SpeechEncoder, TokenEmbedder};

#[derive(Debug, Error)]
pub enum Error {
    #[error("An ONNX runtime error occurred: {0}")]
    Onnx(#[source] ort::Error),
}

pub struct ChatterboxTts {
    encoder: SpeechEncoder,
    embedder: TokenEmbedder,
    model: LanguageModel,
    decoder: ConditionalDecoder,
    pub sample_rate: u32,
}

pub struct GenerateOptions {
    pub exaggeration: f32,
    pub cfg_weight: f32,
    pub temperature: f32,
}

impl ChatterboxTts {
    pub(crate) fn new(
        encoder: SpeechEncoder,
        embedder: TokenEmbedder,
        model: LanguageModel,
        decoder: ConditionalDecoder,
        sample_rate: u32,
    ) -> Self {
        Self {
            encoder,
            embedder,
            model,
            decoder,
            sample_rate,
        }
    }
    pub fn load(device: Device) -> Result<Self, Error> {
        todo!()
    }

    #[cfg(feature = "download")]
    pub async fn download_and_load(device: Device) -> Result<Self, Error> {
        use crate::downloader;
        downloader::download_missing(models).await;
        Self::load(device)
    }

    pub fn generate(
        &mut self,
        text: &str,
        reference_audio_path: impl AsRef<Path>,
        opts: GenerateOptions,
    ) -> Result<Vec<f32>, Error> {
        todo!()
    }
}
