use std::path::PathBuf;

use crate::config;

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Variant {
    #[default]
    Fp32,
    Fp16,
    Int8,
    Q4,
    Q4Fp16,
}

impl Variant {
    fn filename_suffix(&self) -> &'static str {
        match self {
            Variant::Fp32 => "",
            Variant::Fp16 => "_fp16",
            Variant::Int8 => "_quantized",
            Variant::Q4 => "_q4",
            Variant::Q4Fp16 => "_q4f16",
        }
    }
}

pub trait Model {
    fn filename_prefix(&self) -> &'static str;
    fn variant(&self) -> Variant;

    fn filename(&self) -> String {
        format!(
            "{}{}",
            self.filename_prefix(),
            self.variant().filename_suffix()
        )
    }

    fn graph_file(&self) -> PathBuf {
        let onnx_dir = config::ONNX_DIR.read().expect("ONNX_DIR lock poisoned");
        onnx_dir.join(PathBuf::from(self.filename()).with_extension("onnx"))
    }

    fn weights_file(&self) -> PathBuf {
        let onnx_dir = config::ONNX_DIR.read().expect("ONNX_DIR lock poisoned");
        onnx_dir.join(PathBuf::from(self.filename()).with_extension("onnx_data"))
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SpeechEncoder {
    pub variant: Variant,
}

impl Model for SpeechEncoder {
    fn filename_prefix(&self) -> &'static str {
        "speech_encoder"
    }

    fn variant(&self) -> Variant {
        self.variant
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TokenEmbedder {
    pub variant: Variant,
}

impl Model for TokenEmbedder {
    fn filename_prefix(&self) -> &'static str {
        "embed_tokens"
    }

    fn variant(&self) -> Variant {
        self.variant
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LanguageModel {
    pub variant: Variant,
}

impl Model for LanguageModel {
    fn filename_prefix(&self) -> &'static str {
        "language_model"
    }

    fn variant(&self) -> Variant {
        self.variant
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConditionalDecoder {
    pub variant: Variant,
}

impl Model for ConditionalDecoder {
    fn filename_prefix(&self) -> &'static str {
        "conditional_decoder"
    }

    fn variant(&self) -> Variant {
        self.variant
    }
}
