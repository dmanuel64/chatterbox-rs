use crate::{
    Variant, config,
    models::{self, ConditionalDecoder, LanguageModel, Model, SpeechEncoder, TokenEmbedder},
};
use ndarray::{concatenate, prelude::*};
use ort::{
    device::Device,
    session::{
        Session,
        builder::{AutoDevicePolicy, SessionBuilder},
    },
    value::Tensor,
};
use std::{num::NonZero, path::Path};
use thiserror::Error;
use typed_floats::tf32;

#[derive(Debug, Error)]
pub enum Error {
    #[error("An ONNX runtime error occurred: {0}")]
    OnnxGeneric(#[from] ort::Error),
    #[error("An ONNX runtime error occurred: {0}")]
    OnnxSession(#[from] ort::Error<SessionBuilder>),
}

const SAMPLE_RATE: u32 = 24000;
const START_SPEECH_TOKEN: i64 = 6561;
const STOP_SPEECH_TOKEN: i64 = 6562;
const SILENCE_TOKEN: i64 = 4299;
const NUM_KV_HEADS: usize = 16;
const HEAD_DIM: usize = 64;

pub struct ChatterboxTurbo {
    speech_encoder: SpeechEncoder,
    token_embedder: TokenEmbedder,
    language_model: LanguageModel,
    conditional_decoder: ConditionalDecoder,
    speech_encoder_session: Session,
    token_embedder_session: Session,
    language_model_session: Session,
    conditional_decoder_session: Session,
}

pub struct GenerateOptions {
    pub max_new_tokens: NonZero<u32>,
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

pub struct LoadOptions {
    pub device_policy: AutoDevicePolicy,
    pub speech_encoder: Variant,
    pub token_embedder: Variant,
    pub language_model: Variant,
    pub conditional_decoder: Variant,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            device_policy: AutoDevicePolicy::MaxPerformance,
            speech_encoder: Variant::default(),
            token_embedder: Variant::default(),
            language_model: Variant::default(),
            conditional_decoder: Variant::default(),
        }
    }
}

impl ChatterboxTurbo {
    fn new(
        speech_encoder: SpeechEncoder,
        token_embedder: TokenEmbedder,
        language_model: LanguageModel,
        conditional_decoder: ConditionalDecoder,
        device_policy: AutoDevicePolicy,
    ) -> Result<Self, Error> {
        let speech_encoder_session = Session::builder()?
            .with_auto_device(device_policy)?
            .commit_from_file(speech_encoder.graph_file())?;
        let token_embedder_session = Session::builder()?
            .with_auto_device(device_policy)?
            .commit_from_file(token_embedder.graph_file())?;
        let language_model_session = Session::builder()?
            .with_auto_device(device_policy)?
            .commit_from_file(language_model.graph_file())?;
        let conditional_decoder_session = Session::builder()?
            .with_auto_device(device_policy)?
            .commit_from_file(conditional_decoder.graph_file())?;
        Ok(Self {
            speech_encoder,
            token_embedder,
            language_model,
            conditional_decoder,
            speech_encoder_session,
            token_embedder_session,
            language_model_session,
            conditional_decoder_session,
        })
    }

    pub fn load(variant: Variant) -> Result<Self, Error> {
        Self::load_with_options(LoadOptions {
            speech_encoder: variant,
            token_embedder: variant,
            language_model: variant,
            conditional_decoder: variant,
            ..Default::default()
        })
    }

    pub fn load_with_options(options: LoadOptions) -> Result<Self, Error> {
        Self::new(
            models::SpeechEncoder {
                variant: options.speech_encoder,
            },
            models::TokenEmbedder {
                variant: options.token_embedder,
            },
            models::LanguageModel {
                variant: options.language_model,
            },
            models::ConditionalDecoder {
                variant: options.conditional_decoder,
            },
            options.device_policy,
        )
    }

    fn prepare_audio_input(&self) -> ArrayD<f32> {}

    fn prepare_text_input(&self) -> ArrayD<i64> {}

    fn decode_audio(&self, generate_tokens: ArrayD<i64>) -> ArrayD<i64> {
        let speech_tokens = s![.., 1, -1];
        let silence_tokens = ArrayD::<i64>::from_elem((speech_tokens.shape()[0], 3), SILENCE_TOKEN);
    }

    pub fn generate(
        mut self,
        text: &str,
        reference_audio_path: impl AsRef<Path>,
        options: GenerateOptions,
    ) -> Result<Vec<f32>, Error> {
        let audio_values = self.prepare_audio_input();
        let input_ids = self.prepare_text_input();

        let repetition_penalty_processor = RepetitionPenaltyLogitsProcessor {
            penalty: options.repetition_penalty,
        };
        let generate_tokens = array![[START_SPEECH_TOKEN]];
        for tokens_generated in 0..options.max_new_tokens.get() {
            let mut input_embeds = self
                .token_embedder_session
                .run(ort::inputs![
                    "input_ids" => Tensor::from_array(input_ids.clone())?
                ])
                .expect("TODO create an error for this")
                .get("input_embeds")
                .expect("TODO create an error for this")
                .try_extract_array()
                .expect("TODO make an error for this");

            if tokens_generated == 0 {
                let ort_speech_encoder_input =
                    ort::inputs!["audio_values" => Tensor::from_array(audio_values.clone())?];
                let outputs = self.speech_encoder_session.run(ort_speech_encoder_input)?;
                let condition_embeddings = outputs
                    .get("cond_emb")
                    .expect("TODO create an error for this")
                    .try_extract_array()
                    .expect("TODO make an error for this");
                let prompt_token = outputs
                    .get("prompt_token")
                    .expect("TODO create an error for this");
                let speaker_embeddings = outputs
                    .get("speaker_embeddings")
                    .expect("TODO create an error for this");
                let speaker_features = outputs
                    .get("speaker_features")
                    .expect("TODO create an error for this");
                input_embeds = concatenate![Axis(1), condition_embeddings, input_embeds];

                // Initialize cache and LLM inputs
                let [batch_size, seq_len, ..] = input_embeds.shape();
                let past_key_values = None;

                let attention_mask = ArrayD::<i64>::ones((batch_size, seq_len));
            }
        }
        todo!()
    }
}

struct RepetitionPenaltyLogitsProcessor {
    pub penalty: tf32::StrictlyPositiveFinite,
}

impl RepetitionPenaltyLogitsProcessor {
    pub fn pass(&self, input_ids: ArrayD<u32>, scores: ArrayD<u32>) -> ArrayD<u32> {
        // let scores = ndarray::
    }
}
