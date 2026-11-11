use chatterbox_rs::{
    ChatterboxTurbo, GenerateOptions, LoadOptions, conditional_decoder,
    language_model::{self, KvCacheDevice},
    model, speech_encoder, token_embedder,
};
use clap::{Parser, ValueEnum};
use color_eyre::Result;
use half::f16;
use ort::{environment::Environment, session::Session};
use std::{path::PathBuf, time::Instant};

/// Clone a voice from a reference clip and synthesize new speech from text.
#[derive(Parser)]
struct Args {
    /// Path to the reference audio clip to clone the voice from
    reference_audio: PathBuf,

    /// Text to synthesize
    text: String,

    /// Path to write the generated WAV file to
    output: PathBuf,

    /// Language model weight variant to use. Defaults to q4 on CPU and q4-fp16
    /// with --cuda.
    ///
    /// fp32, int8 and q4 are CPU variants: they run on CUDA, but CUDA's
    /// attention kernel needs an fp16 graph, so with them attention falls back
    /// to the CPU on every token. With --cuda, use fp16 or q4-fp16. Avoid int8
    /// on CPU too: it can be several times slower per token than q4 or fp32.
    #[arg(long, value_enum)]
    variant: Option<ModelVariant>,

    /// Run speech_encoder/language_model/conditional_decoder on the CUDA
    /// execution provider. token_embedder is deliberately left off CUDA
    /// regardless of this flag: ONNX Runtime's CUDA kernel for its quantized
    /// embedding lookup (`GatherBlockQuantized`) throws `cudaErrorInvalidValue`
    /// at runtime for this graph, and it's cheap enough (a single embedding
    /// lookup per generated token) that running it on CPU costs nothing
    /// noticeable next to language_model's full forward pass in the same loop
    /// iteration.
    #[arg(long)]
    cuda: bool,

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

async fn run<L: model::Precision>(
    environment: &Environment,
    variant: model::Variant<L>,
    args: Args,
) -> Result<()> {
    #[cfg(feature = "download")]
    chatterbox_rs::downloader::download_missing_split(model::Variant::<f32>::FP32, variant, false)
        .await?;

    let options = GenerateOptions {
        max_new_tokens: args.max_new_tokens.try_into()?,
        repetition_penalty: args.repetition_penalty.try_into()?,
    };

    let load_start = Instant::now();
    let mut chatterbox = ChatterboxTurbo::load_with_environment_and_options(
        environment,
        LoadOptions {
            speech_encoder: speech_encoder::Metadata {
                variant: model::Variant::<f32>::FP32,
            },
            speech_encoder_session_builder: None,
            token_embedder: token_embedder::Metadata {
                variant: model::Variant::<f32>::FP32,
            },
            token_embedder_session_builder: Some(
                Session::builder(environment)?
                    .with_no_environment_execution_providers()
                    .map_err(ort::Error::<()>::from)?,
            ),
            language_model: language_model::Metadata { variant },
            language_model_session_builder: None,
            language_model_kv_device: if cfg!(feature = "cuda")
                && args.cuda
                && std::any::TypeId::of::<L>() == std::any::TypeId::of::<f16>()
            {
                KvCacheDevice::Cuda { device_id: 0 }
            } else {
                KvCacheDevice::Cpu
            },
            conditional_decoder: conditional_decoder::Metadata {
                variant: model::Variant::<f32>::FP32,
            },
            conditional_decoder_session_builder: None,
            sample_rate: 24000,
            num_kv_heads: 16,
            head_dim: 64,
        },
    )?;
    println!("Loaded in {:?}", load_start.elapsed());
    let generate_start = Instant::now();
    chatterbox.generate_with_files(&args.text, args.reference_audio, args.output, options)?;
    println!("Generated in {:?}", generate_start.elapsed());

    Ok(())
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    #[allow(unused_mut)]
    let mut env_builder = ort::init();
    #[cfg(feature = "cuda")]
    if args.cuda {
        env_builder = env_builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()]);
    }
    #[cfg(not(feature = "cuda"))]
    if args.cuda {
        eprintln!("warning: --cuda requires building with `--features cuda`; ignoring");
    }
    let env = env_builder.build()?;

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

    let variant = args.variant.unwrap_or(if args.cuda {
        ModelVariant::Q4Fp16
    } else {
        ModelVariant::Q4
    });
    if args.cuda && !matches!(variant, ModelVariant::Fp16 | ModelVariant::Q4Fp16) {
        eprintln!(
            "warning: fp32/int8/q4 are CPU variants; on CUDA their attention runs on the CPU \
             every token. Use --variant q4-fp16 or fp16 with --cuda."
        );
    }
    match variant {
        ModelVariant::Fp32 => run(&env, model::Variant::<f32>::FP32, args).await,
        ModelVariant::Int8 => run(&env, model::Variant::<f32>::INT8, args).await,
        ModelVariant::Q4 => run(&env, model::Variant::<f32>::Q4, args).await,
        ModelVariant::Fp16 => run(&env, model::Variant::<f16>::FP16, args).await,
        ModelVariant::Q4Fp16 => run(&env, model::Variant::<f16>::Q4_FP16, args).await,
    }
}
