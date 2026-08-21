use futures::{StreamExt, TryStreamExt, stream};
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
    ChatterboxTts, Variant, config,
    onnx::{ChatterboxOnnxFile, ConditionalDecoder, LanguageModel, SpeechEncoder, TokenEmbedder},
};

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to get remote file: {0}")]
    Reqwest(#[from] reqwest::Error),
    #[error("failed to stream file: {source}")]
    Streaming { url: Url, source: io::Error },
    #[error("invalid {kind} name: '{name}'")]
    InvalidName {
        name: String,
        kind: &'static str,
        source: url::ParseError,
    },
}

async fn download_file(
    url: Url,
    dest: &Path,
    auth_token: Option<String>,
    force: bool,
) -> Result<(), Error> {
    if force || !dest.exists() {
        let client = reqwest::Client::new();
        let mut request = client.get(url.clone());
        if let Some(t) = auth_token {
            request = request.bearer_auth(t);
        }
        let response = request.send().await?;
        let mut byte_stream = response.bytes_stream();
        let mut file = File::create(dest)
            .await
            .map_err(|source| Error::Streaming {
                url: url.clone(),
                source,
            })?;

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = chunk_result?;
            file.write_all(&chunk)
                .await
                .map_err(|source| Error::Streaming {
                    url: url.clone(),
                    source,
                })?;
        }
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
    download_file(file_url, dest, auth_token, force).await
}

async fn download_chatterbot_file(source: &str, dest: &Path, force: bool) -> Result<(), Error> {
    const CHATTERBOT_BRANCH: &str = "main";
    download_hf_file(
        "ResembleAI",
        "chatterbox-turbo-ONNX",
        source,
        CHATTERBOT_BRANCH,
        dest,
        force,
    )
    .await
}

pub struct SourceDest {
    source: String,
    dest: PathBuf,
    force: bool,
}

async fn download_chatterbot_files(targets: &[SourceDest]) -> Result<(), Error> {
    stream::iter(targets)
        .map(
            |SourceDest {
                 source,
                 dest,
                 force,
             }| async move { download_chatterbot_file(&source, &dest, *force).await },
        )
        .buffer_unordered(
            *config::MAX_CONCURRENT_DOWNLOADS
                .read()
                .expect("MAX_CONCURRENT_DOWNLOADS lock poisoned"),
        )
        .try_collect::<()>()
        .await
}

fn onnx_targets(files: &[Box<dyn ChatterboxOnnxFile>], force: bool) -> Vec<SourceDest> {
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

async fn download_onnx_files(
    files: &[Box<dyn ChatterboxOnnxFile>],
    force: bool,
) -> Result<(), Error> {
    let targets = onnx_targets(files, force);
    download_chatterbot_files(&targets).await
}

pub async fn download_model(variant: Variant, force: bool) -> Result<ChatterboxTts, Error> {
    let encoder = SpeechEncoder { variant };
    let embedder = TokenEmbedder { variant };
    let model = LanguageModel { variant };
    let decoder = ConditionalDecoder { variant };
    download_onnx_files(
        &[
            Box::new(encoder.clone()),
            Box::new(embedder.clone()),
            Box::new(model.clone()),
            Box::new(decoder.clone()),
        ],
        force,
    )
    .await?;
    Ok(ChatterboxTts::new(encoder, embedder, model, decoder, 0))
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

pub async fn download_tokenizer(force: bool) -> Result<(), Error> {
    let targets = [tokenizer_target(force)];
    download_chatterbot_files(&targets).await
}

pub async fn download_missing(variant: Variant) -> Result<ChatterboxTts, Error> {
    let encoder = SpeechEncoder { variant };
    let embedder = TokenEmbedder { variant };
    let model = LanguageModel { variant };
    let decoder = ConditionalDecoder { variant };
    let mut targets: Vec<_> = onnx_targets(
        &[
            Box::new(encoder.clone()),
            Box::new(embedder.clone()),
            Box::new(model.clone()),
            Box::new(decoder.clone()),
        ],
        false,
    );
    targets.push(tokenizer_target(false));
    download_chatterbot_files(&targets).await?;
    Ok(ChatterboxTts::new(encoder, embedder, model, decoder, 0))
}
