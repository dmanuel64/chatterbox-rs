use std::{
    path::{Path, PathBuf},
    sync::{LazyLock, RwLock},
};

use directories::ProjectDirs;

static DIRS: LazyLock<ProjectDirs> = LazyLock::new(|| {
    ProjectDirs::from("io.github", "dmanuel64", "chatterbox-rs")
        .expect("a home directory to be found on the system")
});
static DATA_DIR: LazyLock<&Path> = LazyLock::new(|| DIRS.data_local_dir());

pub static MODELS_DIR: LazyLock<RwLock<PathBuf>> =
    LazyLock::new(|| RwLock::new(DATA_DIR.join("models")));
