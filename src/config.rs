use directories::ProjectDirs;
#[cfg(feature = "download")]
use std::env;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    sync::{LazyLock, RwLock},
};

static DIRS: LazyLock<ProjectDirs> = LazyLock::new(|| {
    ProjectDirs::from("io.github", "dmanuel64", "chatterbox-rs")
        .expect("a home directory to be found on the system")
});
static DATA_DIR: LazyLock<&Path> = LazyLock::new(|| DIRS.data_local_dir());

pub static ONNX_DIR: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(DATA_DIR.join("onnx")));
pub static TOKENIZER_PATH: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(DATA_DIR.join("tokenizer.json")));

#[cfg(feature = "download")]
pub static MAX_CONCURRENT_DOWNLOADS: LazyLock<RwLock<usize>> = LazyLock::new(|| RwLock::new(4));
#[cfg(feature = "download")]
pub static SHOW_DOWNLOAD_PROGRESS: LazyLock<RwLock<bool>> = LazyLock::new(|| RwLock::new(true));
#[cfg(feature = "download")]
pub static HF_TOKEN: LazyLock<RwLock<Option<String>>> =
    LazyLock::new(|| RwLock::new(env::var("HF_TOKEN").ok()));
