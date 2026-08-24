use chatterbox_rs::{
    AutoDevicePolicy, ChatterboxTurbo, GenerateOptions, GraphOptimizationLevel, LoadOptions,
    Variant,
};
use clap::{Parser, ValueEnum};
use color_eyre::Result;
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

    /// Execution-provider device selection policy
    #[arg(long, value_enum, default_value_t = DevicePolicy::MaxPerformance)]
    device_policy: DevicePolicy,

    /// ONNX graph optimization level
    #[arg(long, value_enum, default_value_t = OptLevel::All)]
    graph_optimization_level: OptLevel,

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

#[derive(Clone, Copy, ValueEnum)]
enum DevicePolicy {
    Default,
    PreferCpu,
    PreferNpu,
    PreferGpu,
    MaxPerformance,
    MaxEfficiency,
    MinPower,
}

impl From<DevicePolicy> for AutoDevicePolicy {
    fn from(value: DevicePolicy) -> Self {
        match value {
            DevicePolicy::Default => AutoDevicePolicy::Default,
            DevicePolicy::PreferCpu => AutoDevicePolicy::PreferCPU,
            DevicePolicy::PreferNpu => AutoDevicePolicy::PreferNPU,
            DevicePolicy::PreferGpu => AutoDevicePolicy::PreferGPU,
            DevicePolicy::MaxPerformance => AutoDevicePolicy::MaxPerformance,
            DevicePolicy::MaxEfficiency => AutoDevicePolicy::MaxEfficiency,
            DevicePolicy::MinPower => AutoDevicePolicy::MinPower,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum OptLevel {
    Disable,
    Level1,
    Level2,
    Level3,
    All,
}

impl From<OptLevel> for GraphOptimizationLevel {
    fn from(value: OptLevel) -> Self {
        match value {
            OptLevel::Disable => GraphOptimizationLevel::Disable,
            OptLevel::Level1 => GraphOptimizationLevel::Level1,
            OptLevel::Level2 => GraphOptimizationLevel::Level2,
            OptLevel::Level3 => GraphOptimizationLevel::Level3,
            OptLevel::All => GraphOptimizationLevel::All,
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let env = ort::environment::Environment::current()?;
    println!("Discovered execution provider devices:");
    for device in env.devices() {
        let hw = device.hardware_device();
        println!(
            "  {} ({}) - {:?} [{}]",
            device.ep()?,
            device.ep_vendor()?,
            hw.ty(),
            hw.vendor()?
        );
    }

    let args = Args::parse();
    let variant: Variant = args.variant.into();

    #[cfg(feature = "download")]
    chatterbox_rs::downloader::download_missing(variant, false).await?;

    let options = GenerateOptions {
        max_new_tokens: args.max_new_tokens.try_into()?,
        repetition_penalty: args.repetition_penalty.try_into()?,
    };

    let mut chatterbox = ChatterboxTurbo::load_with_options(LoadOptions {
        device_policy: args.device_policy.into(),
        graph_optimization_level: args.graph_optimization_level.into(),
        speech_encoder: variant,
        token_embedder: variant,
        language_model: variant,
        conditional_decoder: variant,
        ..Default::default()
    })?;
    chatterbox.generate_with_files(&args.text, args.reference_audio, args.output, options)?;

    Ok(())
}
