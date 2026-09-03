//! Fetches ONNX graphs, weights, and the tokenizer from Hugging Face into [`config::ONNX_DIR`] /
//! [`config::TOKENIZER_PATH`]. Requires the `download` feature.

use futures::{StreamExt, TryStreamExt, stream};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use reqwest::Url;
use std::{
    io,
    path::{Path, PathBuf},
    str::FromStr,
    sync::LazyLock,
};
use thiserror::Error;
use tokio::{fs::File, io::AsyncWriteExt};

use crate::{
    config, model,
    models::{conditional_decoder, language_model, speech_encoder, token_embedder},
};
use num_traits::Float;

/// Errors that can occur while downloading model files.
#[derive(Debug, Error)]
pub enum Error {
    /// The HTTP request itself failed.
    #[error("failed to get remote file: {0}")]
    Reqwest(#[from] reqwest::Error),
    /// Writing the downloaded bytes to disk failed.
    #[error("failed to stream file: {source}")]
    Streaming { url: Url, source: io::Error },
    /// A repo owner, name, branch, or file path couldn't be turned into a valid URL segment.
    #[error("invalid {kind} name: '{name}'")]
    InvalidName {
        name: String,
        kind: &'static str,
        source: url::ParseError,
    },
    /// The download stopped short of the server-reported content length.
    #[error("incomplete download of {url}: expected {expected} bytes, got {actual}")]
    Incomplete {
        url: Url,
        expected: u64,
        actual: u64,
    },
}

fn download_progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{msg}\n{bar:40.cyan/blue} {bytes}/{total_bytes} ({bytes_per_sec}, {eta})",
    )
    .expect("valid progress bar template")
    .progress_chars("#>-")
}

async fn download_file(
    url: Url,
    dest: &Path,
    auth_token: Option<String>,
    force: bool,
    multi: &MultiProgress,
) -> Result<(), Error> {
    if force || !dest.exists() {
        let client = reqwest::Client::new();
        let mut request = client.get(url.clone());
        if let Some(t) = auth_token {
            request = request.bearer_auth(t);
        }
        let response = request.send().await?;
        let expected_len = response.content_length();
        let mut byte_stream = response.bytes_stream();
        if let Some(parent) = dest.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|source| Error::Streaming {
                    url: url.clone(),
                    source,
                })?;
        }
        let mut file = File::create(dest)
            .await
            .map_err(|source| Error::Streaming {
                url: url.clone(),
                source,
            })?;

        let pb = if *config::SHOW_DOWNLOAD_PROGRESS
            .read()
            .expect("SHOW_DOWNLOAD_PROGRESS lock poisoned")
        {
            let pb = multi.add(match expected_len {
                Some(len) => ProgressBar::new(len),
                None => ProgressBar::new_spinner(),
            });
            pb.set_style(download_progress_style());
            pb.set_message(dest.file_name().unwrap_or_default().to_string_lossy().into_owned());
            pb
        } else {
            ProgressBar::hidden()
        };

        let mut bytes_written: u64 = 0;
        let stream_result: Result<(), Error> = async {
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = chunk_result?;
                bytes_written += chunk.len() as u64;
                pb.set_position(bytes_written);
                file.write_all(&chunk)
                    .await
                    .map_err(|source| Error::Streaming {
                        url: url.clone(),
                        source,
                    })?;
            }
            Ok(())
        }
        .await;

        if let Err(err) = stream_result {
            pb.finish_and_clear();
            let _ = tokio::fs::remove_file(dest).await;
            return Err(err);
        }
        if let Some(expected) = expected_len
            && bytes_written != expected
        {
            pb.finish_and_clear();
            let _ = tokio::fs::remove_file(dest).await;
            return Err(Error::Incomplete {
                url,
                expected,
                actual: bytes_written,
            });
        }
        pb.finish_and_clear();
    }
    Ok(())
}

async fn download_hf_file(
    owner: &str,
    repo: &str,
    source: &str,
    branch: &str,
    dest: &Path,
    force: bool,
    multi: &MultiProgress,
) -> Result<(), Error> {
    static HF_URL: LazyLock<Url> = LazyLock::new(|| {
        Url::from_str("https://huggingface.co/").expect("URL to be parsed correctly")
    });

    let repo_url = HF_URL
        .join(&format!("{}/", owner.trim_end_matches("/")))
        .map_err(|source| Error::InvalidName {
            name: owner.to_string(),
            kind: "owner",
            source,
        })?
        .join(&format!("{}/", repo.trim_end_matches("/")))
        .map_err(|source| Error::InvalidName {
            name: repo.to_string(),
            kind: "repository",
            source,
        })?;
    let mut file_url = repo_url
        .join("resolve/")
        .expect("URL to be parsed correctly")
        .join(&format!("{}/", branch.trim_end_matches("/")))
        .map_err(|source| Error::InvalidName {
            name: branch.to_string(),
            kind: "branch",
            source,
        })?
        .join(source)
        .map_err(|source| Error::InvalidName {
            name: source.to_string(),
            kind: "file path",
            source,
        })?;
    file_url.set_query(Some("download=true"));
    let auth_token = config::HF_TOKEN
        .read()
        .expect("HF_TOKEN lock poisoned")
        .clone();
    download_file(file_url, dest, auth_token, force, multi).await
}

async fn download_chatterbot_file(
    source: &str,
    dest: &Path,
    force: bool,
    use_patched_models: bool,
    multi: &MultiProgress,
) -> Result<(), Error> {
    const CHATTERBOT_BRANCH: &str = "main";
    download_hf_file(
        if use_patched_models {
            "dmanuel99"
        } else {
            "ResembleAI"
        },
        "chatterbox-turbo-ONNX",
        source,
        CHATTERBOT_BRANCH,
        dest,
        force,
        multi,
    )
    .await
}

/// A single file to fetch: a source path relative to the Hugging Face repo, a local destination,
/// and whether to re-download it even if it already exists.
pub struct SourceDest {
    source: String,
    dest: PathBuf,
    force: bool,
}

async fn download_chatterbot_files(
    targets: &[SourceDest],
    use_patched_models: bool,
) -> Result<(), Error> {
    let multi = MultiProgress::new();
    stream::iter(targets)
        .map(|SourceDest { source, dest, force }| {
            let multi = &multi;
            async move {
                download_chatterbot_file(source, dest, *force, use_patched_models, multi).await
            }
        })
        .buffer_unordered(
            *config::MAX_CONCURRENT_DOWNLOADS
                .read()
                .expect("MAX_CONCURRENT_DOWNLOADS lock poisoned"),
        )
        .try_collect::<()>()
        .await
}

fn onnx_targets<F: Float + 'static>(
    files: &[Box<dyn model::Metadata<F>>],
    force: bool,
) -> Vec<SourceDest> {
    let mut targets = Vec::with_capacity(files.len() * 2);
    for m in files {
        let graph_dest = m.graph_file();
        let graph_source = format!(
            "onnx/{}",
            graph_dest
                .file_name()
                .expect("a filename to be present")
                .to_string_lossy()
        );
        let weights_dest = m.weights_file();
        let weights_source = format!(
            "onnx/{}",
            weights_dest
                .file_name()
                .expect("a filename to be present")
                .to_string_lossy()
        );
        targets.push(SourceDest {
            source: graph_source,
            dest: graph_dest,
            force,
        });
        targets.push(SourceDest {
            source: weights_source,
            dest: weights_dest,
            force,
        });
    }
    targets
}

async fn download_onnx_files<F: Float + 'static>(
    files: &[Box<dyn model::Metadata<F>>],
    force: bool,
    use_patched_models: bool,
) -> Result<(), Error> {
    let targets = onnx_targets(files, force);
    download_chatterbot_files(&targets, use_patched_models).await
}

/// Downloads all four ONNX graphs (and their weights) for `variant`, overwriting any files
/// already present when `force` is `true`.
pub async fn download_model<F: Float + 'static>(
    variant: model::Variant<F>,
    force: bool,
    use_patched_models: bool,
) -> Result<(), Error> {
    let encoder = speech_encoder::Metadata::<F> { variant };
    let embedder = token_embedder::Metadata::<F> { variant };
    let lm = language_model::Metadata::<F> { variant };
    let decoder = conditional_decoder::Metadata::<F> { variant };
    download_onnx_files(
        &[
            Box::new(encoder) as Box<dyn model::Metadata<F>>,
            Box::new(embedder),
            Box::new(lm),
            Box::new(decoder),
        ],
        force,
        use_patched_models,
    )
    .await?;
    Ok(())
}

fn tokenizer_target(force: bool) -> SourceDest {
    SourceDest {
        source: "tokenizer.json".to_string(),
        dest: config::TOKENIZER_PATH
            .read()
            .expect("TOKENIZER_PATH lock poisoned")
            .clone(),
        force,
    }
}

/// Downloads the tokenizer, overwriting it if it's already present and `force` is `true`.
pub async fn download_tokenizer(force: bool, use_patched_models: bool) -> Result<(), Error> {
    let targets = [tokenizer_target(force)];
    download_chatterbot_files(&targets, use_patched_models).await
}

/// Downloads `variant`'s ONNX graphs, weights, and the tokenizer, skipping any files that
/// already exist on disk.
pub async fn download_missing<F: Float + 'static>(
    variant: model::Variant<F>,
    use_patched_models: bool,
) -> Result<(), Error> {
    let encoder = speech_encoder::Metadata::<F> { variant };
    let embedder = token_embedder::Metadata::<F> { variant };
    let lm = language_model::Metadata::<F> { variant };
    let decoder = conditional_decoder::Metadata::<F> { variant };
    let mut targets: Vec<_> = onnx_targets(
        &[
            Box::new(encoder) as Box<dyn model::Metadata<F>>,
            Box::new(embedder),
            Box::new(lm),
            Box::new(decoder),
        ],
        false,
    );
    targets.push(tokenizer_target(false));
    download_chatterbot_files(&targets, use_patched_models).await?;
    Ok(())
}

/// Like [`download_missing`], but lets `language_model` use a different variant/precision from
/// `speech_encoder`/`token_embedder`/`conditional_decoder` — useful when only `language_model`'s
/// precision is meant to vary (e.g. because the other three are restricted to `f32` without the
/// `custom-variants` feature). Downloads each component's own files exactly once, rather than
/// calling [`download_missing`] twice and fetching both variants' copies of the fixed components.
pub async fn download_missing_split<S: Float + 'static, L: Float + 'static>(
    non_language_model_variant: model::Variant<S>,
    language_model_variant: model::Variant<L>,
    use_patched_models: bool,
) -> Result<(), Error> {
    let encoder = speech_encoder::Metadata::<S> {
        variant: non_language_model_variant,
    };
    let embedder = token_embedder::Metadata::<S> {
        variant: non_language_model_variant,
    };
    let decoder = conditional_decoder::Metadata::<S> {
        variant: non_language_model_variant,
    };
    let lm = language_model::Metadata::<L> {
        variant: language_model_variant,
    };
    let mut targets = onnx_targets(
        &[
            Box::new(encoder) as Box<dyn model::Metadata<S>>,
            Box::new(embedder),
            Box::new(decoder),
        ],
        false,
    );
    targets.extend(onnx_targets(
        &[Box::new(lm) as Box<dyn model::Metadata<L>>],
        false,
    ));
    targets.push(tokenizer_target(false));
    download_chatterbot_files(&targets, use_patched_models).await?;
    Ok(())
}
