# `chatterbox-rs`

A Rust port of [ResembleAI's Chatterbox](https://github.com/resemble-ai/chatterbox) text-to-speech pipeline, running its components as exported ONNX graphs through the [`ort`](https://ort.pyke.io/) crate.

## Table of Contents

- [`chatterbox-rs`](#chatterbox-rs)
  - [Table of Contents](#table-of-contents)
  - [Models Support Status](#models-support-status)
  - [Features](#features)
  - [Quickstart](#quickstart)
    - [With `download` enabled](#with-download-enabled)
    - [Without `download`](#without-download)

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

Chatterbox-Turbo needs four ONNX graphs (`speech_encoder`, `embed_tokens`, `language_model`, `conditional_decoder`) plus a `tokenizer.json`, all pulled from the [`ResembleAI/chatterbox-turbo-ONNX`](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) Hugging Face repo. There are two ways to get them onto disk.

### With `download` enabled

Add the `download` feature. This pulls in an async downloader, so you'll also need an async runtime like `tokio`:

```bash
cargo add chatterbox-rs --features download
cargo add tokio --features rt,macros
```

```rust
use chatterbox_rs::{ChatterboxTurbo, GenerateOptions, model};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Downloads whatever's missing for the default (int8) variant into config::ONNX_DIR.
    // Cheap to call on every run (already-downloaded files are skipped)
    chatterbox_rs::downloader::download_missing(model::Variant::<f32>::INT8, false).await?;

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

Download the graphs yourself from the `onnx/` folder of [`ResembleAI/chatterbox-turbo-ONNX`](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/tree/main/onnx) (each graph is a `<name>.onnx` + `<name>.onnx_data` pair — both files are required) along with `tokenizer.json` from the repo root. For the default `int8` variant that's `speech_encoder_quantized`, `embed_tokens_quantized`, `language_model_quantized`, and `conditional_decoder_quantized`.

Point `config::ONNX_DIR` and `config::TOKENIZER_PATH` at wherever you put them before loading:

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

`ChatterboxTurbo::load()` always expects the `int8` variant at all four components; see [`ChatterboxTurbo::load_with_options`](src/chatterbox_turbo.rs) to load a different `model::Variant` per component (e.g. `fp16` for just the language model, as `examples/clone_voice.rs` does).
