# `chatterbox-rs`

A Rust port of [ResembleAI's
Chatterbox](https://github.com/resemble-ai/chatterbox) text-to-speech pipeline,
running its components as exported ONNX graphs through the
[`ort`](https://ort.pyke.io/) crate.

## Table of Contents

- [`chatterbox-rs`](#chatterbox-rs)
  - [Table of Contents](#table-of-contents)
  - [Models Support Status](#models-support-status)
  - [Features](#features)
  - [Quickstart](#quickstart)
    - [With `download` enabled](#with-download-enabled)
    - [Without `download`](#without-download)
    - [Examples](#examples)
  - [Choosing a variant (CPU vs. GPU)](#choosing-a-variant-cpu-vs-gpu)

## Models Support Status

- [x] Chatterbox-Turbo
- [ ] Chatterbox-Nano
- [ ] Chatterbox-Multilingual V3
- [ ] Single Language Pack
- [ ] Chatterbox *(Original)*

## Features

| Feature                    | Description                                                                    |
| -------------------------- | :----------------------------------------------------------------------------- |
| **`cuda`**                 | Enables CUDA support                                                           |
| **`mp3`**                  | Enables support for using `.mp3` reference files.                              |
| **`mp4`**                  | Enables support for using `.mp4` and `.aac` reference files.                   |
| **`common-audio-formats`** | Enables support for both `.mp3`, .`mp4`, and `.aac` reference files.           |
| **`all-audio-formats`**    | Enables support for all supported files in Symphonia as reference files.       |
| **`download`**             | Download the models                                                            |
| **`serde`**                | Support for serializing/deserializing the models                               |
| **`custom-variants`**      | Support for providing your own custom Chatterbox variants with mixed precision |

## Quickstart

Chatterbox-Turbo needs four ONNX graphs (`speech_encoder`, `embed_tokens`,
`language_model`, `conditional_decoder`) plus a `tokenizer.json`, all pulled
from the
[`ResembleAI/chatterbox-turbo-ONNX`](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX)
Hugging Face repo. There are two ways to get them onto disk.

### With `download` enabled

Add the `download` feature. This pulls in an async downloader, so you'll also
need an async runtime like `tokio`:

```bash
cargo add chatterbox-rs --features download
cargo add tokio --features rt,macros
```

```rust
use chatterbox_rs::{ChatterboxTurbo, GenerateOptions, model};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads whatever's missing for the default (CPU) variants into config::ONNX_DIR:
    // fp32 speech encoder/embedder/decoder and a q4 language model. Cheap to call on
    // every run (already-downloaded files are skipped).
    chatterbox_rs::downloader::download_missing_split(
        model::Variant::<f32>::FP32,
        model::Variant::<f32>::Q4,
        false,
    )
    .await?;

    let mut chatterbox = ChatterboxTurbo::load()?;
    chatterbox.generate_with_files(
        "Hello from chatterbox-rs!",
        "reference.wav",
        "output.wav",
        GenerateOptions::default(),
    )?;
    Ok(())
}
```

### Without `download`

Download the graphs yourself from the `onnx/` folder of
[`ResembleAI/chatterbox-turbo-ONNX`](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/tree/main/onnx)
(each graph is a `<name>.onnx` + `<name>.onnx_data` pair — both files are
required) along with `tokenizer.json` from the repo root. For the defaults
that's `speech_encoder`, `embed_tokens`, `language_model_q4`, and
`conditional_decoder`.

Point `config::ONNX_DIR` and `config::TOKENIZER_PATH` at wherever you put them
before loading:

```bash
cargo add chatterbox-rs
```

```rust
use chatterbox_rs::{ChatterboxTurbo, GenerateOptions, config};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    *config::ONNX_DIR.write().unwrap() = PathBuf::from("/path/to/onnx");
    *config::TOKENIZER_PATH.write().unwrap() = PathBuf::from("/path/to/tokenizer.json");

    let mut chatterbox = ChatterboxTurbo::load()?;
    chatterbox.generate_with_files(
        "Hello from chatterbox-rs!",
        "reference.wav",
        "output.wav",
        GenerateOptions::default(),
    )?;
    Ok(())
}
```

`ChatterboxTurbo::load` uses the CPU defaults (`q4` language model, `fp32` for
the other three components). With the `cuda` feature,
`ChatterboxTurbo::load_cuda(0)` uses the GPU preset instead (see [Choosing a
variant](#choosing-a-variant-cpu-vs-gpu)). See
[`ChatterboxTurbo::load_with_options`](src/chatterbox_turbo.rs) to load a
different `model::Variant` per component.

The ONNX Runtime environment is created and managed for you. If you need to
configure it (environment-wide execution providers, logging, thread pools),
build it with `chatterbox_rs::ort::init()` before loading and pass it to the
`*_with_environment` loaders, as `examples/clone_voice.rs` does. ONNX Runtime
allows only one environment per process, so do this before calling any other
loader.

### Examples

If you would like to quickly test out `chatterbox-rs`, you can run the existing
example in the repo:

```shell
# CPU (defaults to the q4 language model)
cargo run --release --example clone_voice --features download ./reference.wav "Hello from chatterbox-rs!" ./output.wav

# CUDA GPU (defaults to the q4-fp16 language model)
cargo run --release --example clone_voice --features download,cuda ./reference.wav "Hello from chatterbox-rs!" ./output.wav --cuda
```

## Choosing a variant (CPU vs. GPU)

The language model's variant decides which device it runs well on:

| Variant                            | Computes in | Use on                           |
| ---------------------------------- | ----------- | -------------------------------- |
| `q4`                               | fp32        | **CPU** (default)                |
| `fp32`                             | fp32        | CPU                              |
| `int8`                             | fp32        | Not recommended                  |
| `q4f16` (`q4-fp16` in the example) | fp16        | **CUDA** (default with `--cuda`) |
| `fp16`                             | fp16        | CUDA                             |

Set `LoadOptions::language_model_kv_device` to `KvCacheDevice::Cuda` only for an
fp16 language model on CUDA (`LoadOptions::cuda` does this for you); keep
`KvCacheDevice::Cpu` everywhere else.
