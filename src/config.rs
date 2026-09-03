//! Runtime configuration: where model files live on disk, and (with the `download` feature)
//! how downloads behave. All settings are `RwLock`-guarded statics, so they can be overridden
//! at runtime before models are loaded.

use directories::ProjectDirs;
#[cfg(feature = "download")]
use std::env;
use std::{
    path::{Path, PathBuf},
    sync::{LazyLock, RwLock},
};

static DIRS: LazyLock<ProjectDirs> = LazyLock::new(|| {
    ProjectDirs::from("io.github", "dmanuel64", "chatterbox-rs")
        .expect("a home directory to be found on the system")
});
static DATA_DIR: LazyLock<&Path> = LazyLock::new(|| DIRS.data_local_dir());

/// Where the `.onnx` graphs and `.onnx_data` weights are stored.
pub static ONNX_DIR: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(DATA_DIR.join("onnx")));
/// Path to the Chatterbox-Turbo tokenizer
pub static TOKENIZER_PATH: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(DATA_DIR.join("tokenizer.json")));

/// Maximum number of model artifacts to download in parallel
#[cfg(feature = "download")]
pub static MAX_CONCURRENT_DOWNLOADS: LazyLock<RwLock<usize>> = LazyLock::new(|| RwLock::new(4));
/// If download progress should be displayed via progress bar
#[cfg(feature = "download")]
pub static SHOW_DOWNLOAD_PROGRESS: LazyLock<RwLock<bool>> = LazyLock::new(|| RwLock::new(true));
/// HuggingFace authentication token. Populating this will make downloads more faster. Defaults to the environment's
/// `HF_TOKEN` variable.
#[cfg(feature = "download")]
pub static HF_TOKEN: LazyLock<RwLock<Option<String>>> =
    LazyLock::new(|| RwLock::new(env::var("HF_TOKEN").ok()));
