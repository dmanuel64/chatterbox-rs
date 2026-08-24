use chatterbox_rs::{ChatterboxTurbo, GenerateOptions, Variant};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Clone a voice from a reference clip and synthesize new speech from text.
#[derive(Parser)]
struct Args {
    /// Path to the reference audio clip to clone the voice from
    reference_audio: PathBuf,

    /// Text to synthesize
    text: String,

    /// Path to write the generated WAV file to
    output: PathBuf,

    /// Model weight variant to use
    #[arg(long, value_enum, default_value_t = ModelVariant::Fp32)]
    variant: ModelVariant,

    /// Maximum number of speech tokens to generate
    #[arg(long, default_value_t = 1024)]
    max_new_tokens: u32,

    /// Repetition penalty applied to already-generated speech tokens
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f32,
}

#[derive(Clone, Copy, ValueEnum)]
enum ModelVariant {
    Fp32,
    Fp16,
    Int8,
    Q4,
    Q4Fp16,
}

impl From<ModelVariant> for Variant {
    fn from(value: ModelVariant) -> Self {
        match value {
            ModelVariant::Fp32 => Variant::Fp32,
            ModelVariant::Fp16 => Variant::Fp16,
            ModelVariant::Int8 => Variant::Int8,
            ModelVariant::Q4 => Variant::Q4,
            ModelVariant::Q4Fp16 => Variant::Q4Fp16,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let options = GenerateOptions {
        max_new_tokens: args.max_new_tokens.try_into()?,
        repetition_penalty: args.repetition_penalty.try_into()?,
    };

    let mut chatterbox = ChatterboxTurbo::load(args.variant.into())?;
    chatterbox.generate_with_files(&args.text, args.reference_audio, args.output, options)?;

    Ok(())
}
