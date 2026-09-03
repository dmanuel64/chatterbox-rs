# `chatterbox-rs`

Rust [s](https://github.com/resemble-ai/chatterbox) for [ResembleAI's Chatterbox](https://github.com/resemble-ai/chatterbox).

## Models Support Status

- [x] Chatterbox-Turbo
- [ ] Chatterbox-Nano
- [ ] Chatterbox-Multilingual V3
- [ ] Single Language Pack
- [ ] Chatterbox *(Original)*

## Features

| Feature                | Description                                                                    |
| ---------------------- | :----------------------------------------------------------------------------- |
| **`cuda`**             | Enables CUDA support                                                           |
| **`mp3`**              | Enables support for using `.mp3` reference files.                              |
| **`mp4`**              | Enables support for using `.mp4` and `.aac` reference files.                   |
| `common-audio-formats` | Enables support for both `.mp3`, .`mp4`, and `.aac` reference files.           |
| `all-audio-formats`    | Enables support for all supported files in Symphonia as reference files.       |
| `download`             | Download the models                                                            |
| `serde`                | Support for serializing/deserializing the models                               |
| `custom-variants`      | Support for providing your own custom Chatterbox variants with mixed precision |