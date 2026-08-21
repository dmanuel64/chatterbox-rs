use std::path::PathBuf;

use crate::config;

pub struct Tokenizer;

impl Tokenizer {
    const fn filename(&self) -> &'static str {
        "tokenizer.json"
    }

    fn file(&self) -> PathBuf {
        config::TOKENIZER_DIR
            .read()
            .expect("TOKENIZER_DIR lock poisoned")
            .join(self.filename())
    }
}
