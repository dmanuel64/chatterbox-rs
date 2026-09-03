//! [`ChatterboxTurbo`], the top-level TTS pipeline for Chatterbox-Turbo.

use crate::{
    config, model,
    model::{Precision, RestrictedPrecision},
    models::{conditional_decoder, language_model, speech_encoder, token_embedder},
};
use ndarray::{concatenate, prelude::*};
use num_traits::Float;
use ort::session::builder::SessionBuilder;
use std::{fmt::Display, fs, num::NonZero, path::Path};
use thiserror::Error;
use tokenizers::Tokenizer;
use typed_floats::tf32;

/// Errors that can occur while loading or running [`ChatterboxTurbo`].
#[derive(Debug, Error)]
pub enum Error {
    /// An `ort` error not already covered by [`Error::Model`].
    #[error("An ONNX runtime error occurred: {0}")]
    Onnx(#[from] ort::Error),
    /// Text tokenization failed.
    #[error("A tokenizer error occurred: {0}")]
    Tokenizer(#[from] tokenizers::Error),
    /// Loading or writing audio failed.
    #[error("An audio error occurred: {0}")]
    Audio(#[from] crate::audio::Error),
    /// A filesystem operation failed.
    #[error("An I/O error occurred: {0}")]
    Io(#[from] std::io::Error),
    /// One of the four ONNX components failed to load or run.
    #[error("A model error occurred: {0}")]
    Model(#[from] crate::models::model::Error),
}

/// Configures [`ChatterboxTurbo::load_with_options`]: which [`model::Variant`] each of the four
/// components loads, and an optional `ort` [`SessionBuilder`] per component (e.g. to select an
/// execution provider). `S`/`T`/`L`/`C` are the precisions of the speech encoder, token embedder,
/// language model, and conditional decoder respectively.
pub struct LoadOptions<S, T, L, C>
where
    S: Float,
    T: Float,
    L: Float,
    C: Float,
{
    pub speech_encoder: speech_encoder::Metadata<S>,
    pub speech_encoder_session_builder: Option<SessionBuilder>,
    pub token_embedder: token_embedder::Metadata<T>,
    pub token_embedder_session_builder: Option<SessionBuilder>,
    pub language_model: language_model::Metadata<L>,
    pub language_model_session_builder: Option<SessionBuilder>,
    pub conditional_decoder: conditional_decoder::Metadata<C>,
    pub conditional_decoder_session_builder: Option<SessionBuilder>,
    /// Sample rate (Hz) of the generated waveform.
    pub sample_rate: u32,
    /// Number of KV heads in the language model's attention, for sizing its KV cache.
    pub num_kv_heads: usize,
    /// Per-head dimension in the language model's attention, for sizing its KV cache.
    pub head_dim: usize,
}

impl Default for LoadOptions<f32, f32, f32, f32> {
    fn default() -> Self {
        Self {
            speech_encoder: speech_encoder::Metadata {
                variant: model::Variant::<f32>::INT8,
            },
            speech_encoder_session_builder: None,
            token_embedder: token_embedder::Metadata {
                variant: model::Variant::<f32>::INT8,
            },
            token_embedder_session_builder: None,
            language_model: language_model::Metadata {
                variant: model::Variant::<f32>::INT8,
            },
            language_model_session_builder: None,
            conditional_decoder: conditional_decoder::Metadata {
                variant: model::Variant::<f32>::INT8,
            },
            conditional_decoder_session_builder: None,
            sample_rate: 2400,
            num_kv_heads: 16,
            head_dim: 64,
        }
    }
}

/// The full TTS pipeline: text + a reference voice clip to generated speech. Owns one loaded
/// instance of each of the four ONNX components.
#[derive(Debug)]
pub struct ChatterboxTurbo<S, T, L, C>
where
    S: RestrictedPrecision,
    T: RestrictedPrecision,
    L: Precision,
    C: RestrictedPrecision,
{
    tokenizer: Tokenizer,
    pub speech_encoder: speech_encoder::Model<S>,
    pub token_embedder: token_embedder::Model<T>,
    pub language_model: language_model::Model<L>,
    pub conditional_decoder: conditional_decoder::Model<C>,
    pub sample_rate: u32,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl ChatterboxTurbo<f32, f32, f32, f32> {
    /// Loads all four components at [`f32`]/`int8` defaults (see [`LoadOptions::default`]).
    pub fn load() -> Result<Self, Error> {
        Self::load_with_options(LoadOptions::default())
    }
}

impl<S, T, L, C> ChatterboxTurbo<S, T, L, C>
where
    S: RestrictedPrecision,
    T: RestrictedPrecision,
    L: Precision,
    C: RestrictedPrecision,
{
    const START_SPEECH_TOKEN: i64 = 6561;
    const STOP_SPEECH_TOKEN: i64 = 6562;

    /// Loads all four components per `options`, using each component's own [`SessionBuilder`] when
    /// provided, or an `ort` default otherwise.
    pub fn load_with_options(options: LoadOptions<S, T, L, C>) -> Result<Self, Error> {
        let tokenizer = Tokenizer::from_file(
            config::TOKENIZER_PATH
                .read()
                .expect("TOKENIZER_PATH lock poisoned")
                .clone(),
        )?;
        Ok(Self {
            tokenizer,
            speech_encoder: if let Some(builder) = options.speech_encoder_session_builder {
                speech_encoder::Model::load_with_builder(options.speech_encoder, builder)?
            } else {
                speech_encoder::Model::load(options.speech_encoder)?
            },
            token_embedder: if let Some(builder) = options.token_embedder_session_builder {
                token_embedder::Model::load_with_builder(options.token_embedder, builder)?
            } else {
                token_embedder::Model::load(options.token_embedder)?
            },
            language_model: if let Some(builder) = options.language_model_session_builder {
                language_model::Model::load_with_builder(
                    options.language_model,
                    builder,
                    options.num_kv_heads,
                    options.head_dim,
                )?
            } else {
                language_model::Model::load(
                    options.language_model,
                    options.num_kv_heads,
                    options.head_dim,
                )?
            },
            conditional_decoder: if let Some(builder) = options.conditional_decoder_session_builder
            {
                conditional_decoder::Model::load_with_builder(options.conditional_decoder, builder)?
            } else {
                conditional_decoder::Model::load(options.conditional_decoder)?
            },
            sample_rate: options.sample_rate,
            num_kv_heads: options.num_kv_heads,
            head_dim: options.head_dim,
        })
    }

    fn prepare_audio_input(&self, reference_audio_bytes: Vec<u8>) -> Result<ArrayD<f32>, Error> {
        Ok(crate::audio::load(reference_audio_bytes, self.sample_rate)?)
    }

    fn prepare_text_input(&self, text: &str) -> Result<ArrayD<i64>, Error> {
        let encoding = self.tokenizer.encode(text, true)?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let seq_len = ids.len();
        Ok(Array2::from_shape_vec((1, seq_len), ids)
            .expect("ids length should match shape")
            .into_dyn())
    }

    /// Slices out the last position's logits, applies repetition penalty against every token
    /// generated so far, and greedily picks the highest-scoring next token.
    fn sample_next_token(
        generate_tokens: &ArrayRefD<i64>,
        logits: ArrayD<f32>,
        repetition_penalty: tf32::StrictlyPositiveFinite,
    ) -> ArrayD<i64> {
        let logits = logits.slice(s![.., -1, ..]).into_dyn();
        let processor = RepetitionPenaltyLogitsProcessor {
            penalty: repetition_penalty,
        };
        let next_token_logits = processor.process(generate_tokens, logits.to_owned());

        let last_axis = Axis(next_token_logits.ndim() - 1);
        next_token_logits
            .map_axis(last_axis, |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(idx, _)| idx as i64)
                    .expect("row should not be empty")
            })
            .insert_axis(last_axis)
    }

    /// Generates speech for `text`, using the voice from `reference_audio_bytes` (any format
    /// supported by the `audio` module). Returns mono [[`f32`]] samples at `self.sample_rate`.
    pub fn generate(
        &mut self,
        text: &str,
        reference_audio_bytes: Vec<u8>,
        options: GenerateOptions,
    ) -> Result<Vec<f32>, Error> {
        let audio_values = self.prepare_audio_input(reference_audio_bytes)?;
        let mut input_ids = self.prepare_text_input(text)?;

        let mut generate_tokens = array![[Self::START_SPEECH_TOKEN]].into_dyn();
        let mut attention_mask = Array2::default(Ix2::default());
        let mut position_ids: Array2<i64> = Array::default(Ix2::default());
        let mut batch_size = 0;
        let mut speaker_embeddings: Option<ArrayD<f32>> = None;
        let mut speaker_features: Option<ArrayD<f32>> = None;
        let mut prompt_token: Option<ArrayD<i64>> = None;

        for tokens_generated in 0..options.max_new_tokens.get() {
            // token_embedder's output is typed T; normalized to f32 immediately since that's
            // what language_model's inputs_embeds always needs
            let mut input_embeds =
                model::to_f32(self.token_embedder.embed_tokens(input_ids.clone())?);

            if tokens_generated == 0 {
                let encoding = self
                    .speech_encoder
                    .encode_reference_audio(model::from_f32(audio_values.clone()))?;
                prompt_token = Some(encoding.prompt_token);
                speaker_embeddings = Some(model::to_f32(encoding.speaker_embeddings));
                speaker_features = Some(model::to_f32(encoding.speaker_features));
                input_embeds = concatenate![
                    Axis(1),
                    model::to_f32(encoding.condition_embeddings),
                    input_embeds
                ];

                // Initialize cache and LLM inputs
                let &[b, seq_len, ..] = input_embeds.shape() else {
                    unreachable!("input_embeds should have at least 2 dimensions")
                };
                batch_size = b;
                self.language_model.init_past_key_values(batch_size);
                let (mask, ids) = language_model::Model::<L>::init_mask(batch_size, seq_len);
                attention_mask = mask;
                position_ids = ids;
            }

            let logits = self.language_model.step_language_model(
                input_embeds,
                &attention_mask,
                &position_ids,
            )?;
            input_ids =
                Self::sample_next_token(&generate_tokens, logits, options.repetition_penalty);
            generate_tokens = concatenate![Axis(1), generate_tokens, input_ids];
            if input_ids.iter().all(|&id| id == Self::STOP_SPEECH_TOKEN) {
                break;
            }

            // Update values for next generation loop
            attention_mask = concatenate![
                Axis(1),
                attention_mask,
                Array2::<i64>::ones(Ix2(batch_size, 1))
            ];
            position_ids = position_ids.slice(s![.., -1..]).to_owned() + 1;
        }
        // decode_audio unconditionally strips the last token, assuming it's the stop token.
        // If we hit max_new_tokens without ever sampling one, append it so a real generated
        // token doesn't get silently discarded instead.
        if !generate_tokens
            .slice(s![.., -1..])
            .iter()
            .all(|&id| id == Self::STOP_SPEECH_TOKEN)
        {
            generate_tokens = concatenate![
                Axis(1),
                generate_tokens,
                Array2::from_elem(Ix2(batch_size, 1), Self::STOP_SPEECH_TOKEN).into_dyn()
            ];
        }
        let wav = self.conditional_decoder.decode_audio(
            prompt_token.expect("prompt token to be cached"),
            &generate_tokens,
            model::from_f32(speaker_embeddings.expect("embeddings to be cached")),
            model::from_f32(speaker_features.expect("features to be cached")),
        )?;
        Ok(model::to_f32(wav).into_iter().collect())
    }

    /// Like [`Self::generate`], but reads the reference audio from a file path.
    pub fn generate_with_ref_file(
        &mut self,
        text: &str,
        reference_audio_path: impl AsRef<Path>,
        options: GenerateOptions,
    ) -> Result<Vec<f32>, Error> {
        let target_audio_bytes = fs::read(reference_audio_path)?;
        self.generate(text, target_audio_bytes, options)
    }

    /// Like [`Self::generate`], but writes the result to a WAV file instead of returning samples.
    pub fn generate_with_output(
        &mut self,
        text: &str,
        reference_audio_bytes: Vec<u8>,
        output_path: impl AsRef<Path>,
        options: GenerateOptions,
    ) -> Result<(), Error> {
        let generated_samples = self.generate(text, reference_audio_bytes, options)?;
        Ok(crate::audio::write(
            &generated_samples,
            self.sample_rate,
            output_path,
        )?)
    }

    /// Combines [`Self::generate_with_ref_file`] and [`Self::generate_with_output`]: reads the
    /// reference audio from a path and writes the result to a WAV file.
    pub fn generate_with_files(
        &mut self,
        text: &str,
        reference_audio_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        options: GenerateOptions,
    ) -> Result<(), Error> {
        let reference_audio_bytes = fs::read(reference_audio_path)?;
        self.generate_with_output(text, reference_audio_bytes, output_path, options)
    }
}

/// Generation parameters for [`ChatterboxTurbo::generate`].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GenerateOptions {
    /// Upper bound on generated speech tokens, before decoding stops even without a stop token.
    pub max_new_tokens: NonZero<u32>,
    /// Penalty applied to logits of tokens already generated, discouraging repetition.
    pub repetition_penalty: tf32::StrictlyPositiveFinite,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1024.try_into().unwrap(),
            repetition_penalty: 1.2.try_into().unwrap(),
        }
    }
}

struct RepetitionPenaltyLogitsProcessor {
    pub penalty: tf32::StrictlyPositiveFinite,
}

impl RepetitionPenaltyLogitsProcessor {
    pub fn process(&self, input_ids: &ArrayRefD<i64>, scores: ArrayD<f32>) -> ArrayD<f32> {
        let penalty: f32 = self.penalty.into();
        let mut scores_processed = scores.clone();

        for (orig_row, (mut proc_row, id_row)) in scores.axis_iter(Axis(0)).zip(
            scores_processed
                .axis_iter_mut(Axis(0))
                .zip(input_ids.axis_iter(Axis(0))),
        ) {
            for &id in id_row.iter() {
                let idx = id as usize;
                let val = orig_row[idx];
                proc_row[idx] = if val < 0.0 {
                    val * penalty
                } else {
                    val / penalty
                };
            }
        }
        scores_processed
    }
}

/// A bracketed tag (e.g. `[angry]`) Chatterbox recognizes when prepended to input text,
/// conditioning generation on that emotion or delivery style. See [`ParalinguisticStrExt`].
#[derive(Debug, Clone, Copy)]
pub enum ParalinguisticTag {
    /// `[angry]`
    Angry,
    /// `[fear]`
    Fear,
    /// `[surprised]`
    Surprised,
    /// `[whispering]`
    Whispering,
    /// `[advertisement]`
    Advertisement,
    /// `[dramatic]`
    Dramatic,
    /// `[narration]`
    Narration,
    /// `[crying]`
    Crying,
    /// `[happy]`
    Happy,
    /// `[sarcastic]`
    Sarcastic,
    /// `[clear throat]`
    ClearThroat,
    /// `[sigh]`
    Sigh,
    /// `[shush]`
    Shush,
    /// `[cough]`
    Cough,
    /// `[groan]`
    Groan,
    /// `[sniff]`
    Sniff,
    /// `[gasp]`
    Gasp,
    /// `[chuckle]`
    Chuckle,
    /// `[laugh]`
    Laugh,
}

impl ParalinguisticTag {
    /// The literal bracketed tag text, e.g. `"[angry]"`.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Angry => "[angry]",
            Self::Fear => "[fear]",
            Self::Surprised => "[surprised]",
            Self::Whispering => "[whispering]",
            Self::Advertisement => "[advertisement]",
            Self::Dramatic => "[dramatic]",
            Self::Narration => "[narration]",
            Self::Crying => "[crying]",
            Self::Happy => "[happy]",
            Self::Sarcastic => "[sarcastic]",
            Self::ClearThroat => "[clear throat]",
            Self::Sigh => "[sigh]",
            Self::Shush => "[shush]",
            Self::Cough => "[cough]",
            Self::Groan => "[groan]",
            Self::Sniff => "[sniff]",
            Self::Gasp => "[gasp]",
            Self::Chuckle => "[chuckle]",
            Self::Laugh => "[laugh]",
        }
    }
}

impl Display for ParalinguisticTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Extension trait for prepending [`ParalinguisticTag`]s to text before passing it to
/// [`ChatterboxTurbo::generate`].
pub trait ParalinguisticStrExt {
    /// Prepends `tag` to the beginning of the text, conditioning the whole utterance on it.
    fn with_tag(&self, tag: ParalinguisticTag) -> String {
        self.with_tags(std::iter::once(tag))
    }
    /// Prepends `tags` (in order) to the beginning of the text, conditioning the whole utterance
    /// on all of them.
    fn with_tags(&self, tags: impl IntoIterator<Item = ParalinguisticTag>) -> String;
}

impl ParalinguisticStrExt for str {
    fn with_tags(&self, tags: impl IntoIterator<Item = ParalinguisticTag>) -> String {
        let prefix = tags
            .into_iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        format!("{prefix} {self}")
    }
}
