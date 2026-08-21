use crate::{
    Variant,
    models::{self, ConditionalDecoder, LanguageModel, Model, SpeechEncoder, TokenEmbedder},
};
use ndarray::{concatenate, prelude::*};
use ort::{
    session::{
        Session,
        builder::{AutoDevicePolicy, SessionBuilder},
    },
    value::Tensor,
};
use std::{collections::HashMap, fs, num::NonZero, path::Path};
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

    fn prepare_audio_input(&self) -> ArrayD<f32> {
        todo!()
    }

    fn prepare_text_input(&self) -> ArrayD<i64> {
        todo!()
    }

    fn decode_audio(
        &mut self,
        prompt_token: i64,
        generate_tokens: ArrayD<i64>,
        speaker_embeddings: ArrayD<i64>,
        speaker_features: ArrayD<i64>,
    ) -> Result<ArrayD<i64>, Error> {
        let speech_tokens = generate_tokens.slice(s![.., 1..-1]).into_dyn();
        let silence_tokens =
            Array2::<i64>::from_elem(Ix2(speech_tokens.shape()[0], 3), SILENCE_TOKEN).into_dyn();
        let speech_tokens = concatenate![
            Axis(1),
            array![prompt_token].into_dyn(),
            speech_tokens,
            silence_tokens
        ];

        let output = self.conditional_decoder_session.run(ort::inputs![
            "speech_tokens" => Tensor::from_array(speech_tokens)?,
            "speaker_embeddings" => Tensor::from_array(speaker_embeddings)?,
            "speaker_features" => Tensor::from_array(speaker_features)?
        ])?;
        // TODO: they have it as squeeze(axis=0)
        let wav = output[0].try_extract_array()?.squeeze();
        Ok(wav.to_owned())
    }

    pub fn generate(
        &mut self,
        text: &str,
        reference_audio_bytes: Vec<i64>,
        options: GenerateOptions,
    ) -> Result<Vec<i64>, Error> {
        let audio_values = self.prepare_audio_input();
        let mut input_ids = self.prepare_text_input();

        let repetition_penalty_processor = RepetitionPenaltyLogitsProcessor {
            penalty: options.repetition_penalty,
        };
        let mut generate_tokens = array![[START_SPEECH_TOKEN]].into_dyn();
        let mut attention_mask = Array2::default(Ix2::default());

        let mut position_ids: Array2<i64> = Array::default(Ix2::default());
        let mut batch_size = 0;
        let mut past_key_values: Vec<(String, Array4<f32>)> = Vec::new();
        let mut speaker_embeddings: ArrayD<i64> = Array::default(Ix1::default()).into_dyn();
        let mut speaker_features: ArrayD<i64> = Array::default(Ix1::default()).into_dyn();
        let mut prompt_token: i64 = i64::default();

        for tokens_generated in 0..options.max_new_tokens.get() {
            let token_embedder_outputs = self.token_embedder_session.run(ort::inputs![
                "input_ids" => Tensor::from_array(input_ids.clone())?
            ])?;
            let mut input_embeds: ArrayD<f32> =
                token_embedder_outputs[0].try_extract_array()?.to_owned();

            if tokens_generated == 0 {
                let ort_speech_encoder_input =
                    ort::inputs!["audio_values" => Tensor::from_array(audio_values.clone())?];
                let speech_encoder_outputs =
                    self.speech_encoder_session.run(ort_speech_encoder_input)?;
                let condition_embeddings = speech_encoder_outputs[0].try_extract_array()?;
                prompt_token = speech_encoder_outputs[1].try_extract_scalar()?;
                speaker_embeddings = speech_encoder_outputs[2].try_extract_array()?.to_owned();
                speaker_features = speech_encoder_outputs[3].try_extract_array()?.to_owned();
                input_embeds = concatenate![Axis(1), condition_embeddings, input_embeds];

                // Initialize cache and LLM inputs
                let &[b, seq_len, ..] = input_embeds.shape() else {
                    unreachable!("input_embeds should have at least 2 dimensions")
                };
                batch_size = b;
                for input in self
                    .language_model_session
                    .inputs()
                    .iter()
                    .filter(|i| i.name() == "past_key_values")
                {
                    // TODO: dtype=np.float16 if i.type == 'tensor(float16)' else np.float32)
                    past_key_values.push((
                        input.name().to_string(),
                        Array4::<f32>::zeros(Ix4(batch_size, NUM_KV_HEADS, 0, HEAD_DIM)),
                    ));
                }
                attention_mask = Array::ones(Ix2(batch_size, seq_len));
                position_ids = Array::from_iter(0..seq_len as i64)
                    .broadcast((batch_size, seq_len))
                    .expect("broadcast should not fail")
                    .to_owned();
            }
            let mut language_model_inputs = ort::inputs![
                "input_embeds" => Tensor::from_array(input_embeds)?,
                "attention_mask" => Tensor::from_array(attention_mask.clone())?,
                "position_id" => Tensor::from_array(position_ids.clone())?,
            ];
            for (name, kv) in &past_key_values {
                language_model_inputs
                    .push((name.as_str().into(), Tensor::from_array(kv.clone())?.into()));
            }
            let language_model_outputs = self.language_model_session.run(language_model_inputs)?;
            let logits = &language_model_outputs[0].try_extract_array()?;
            let present_key_values: Vec<Array4<f32>> = language_model_outputs
                .values()
                .skip(1)
                .map(|v| {
                    v.try_extract_array::<f32>().map(|a| {
                        a.to_owned()
                            .into_dimensionality::<Ix4>()
                            .expect("KV cache tensor should be 4D")
                    })
                })
                .collect::<Result<_, _>>()?;

            let logits = logits.slice(s![.., -1, ..]).into_dyn();
            let next_token_logits =
                repetition_penalty_processor.process(generate_tokens.clone(), logits.to_owned());

            let last_axis = Axis(next_token_logits.ndim() - 1);
            input_ids = next_token_logits
                .map_axis(last_axis, |row| {
                    row.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(idx, _)| idx as i64)
                        .expect("row should not be empty")
                })
                .insert_axis(last_axis);
            generate_tokens = concatenate![Axis(1), generate_tokens, input_ids];
            if input_ids.iter().all(|&id| id == STOP_SPEECH_TOKEN) {
                break;
            }

            // Update values for next generation loop
            attention_mask = concatenate![
                Axis(1),
                attention_mask,
                Array2::<i64>::ones(Ix2(batch_size, 1))
            ];
            position_ids = position_ids.slice(s![.., -1..]).to_owned() + 1;
            for ((_, kv), new_kv) in past_key_values.iter_mut().zip(present_key_values) {
                *kv = new_kv;
            }
        }
        self.decode_audio(
            prompt_token,
            generate_tokens,
            speaker_embeddings,
            speaker_features,
        )
        .map(|a| a.into_iter().collect())
    }

    pub fn generate_with_ref_file(
        &mut self,
        text: &str,
        reference_audio_path: impl AsRef<Path>,
        options: GenerateOptions,
    ) -> Result<Vec<i64>, Error> {
        let target_audio_bytes = todo!();
        self.generate(text, target_audio_bytes, options)
    }

    pub fn generate_with_output(
        &mut self,
        text: &str,
        reference_audio_bytes: Vec<i64>,
        output_path: impl AsRef<Path>,
        options: GenerateOptions,
    ) -> Result<(), Error> {
        let generated_bytes = self.generate(text, reference_audio_bytes, options)?;
        todo!()
    }

    pub fn generate_with_files(
        &mut self,
        text: &str,
        reference_audio_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
        options: GenerateOptions,
    ) -> Result<(), Error> {
        let reference_audio_bytes = todo!();
        self.generate_with_output(text, reference_audio_bytes, output_path, options)
    }
}

struct RepetitionPenaltyLogitsProcessor {
    pub penalty: tf32::StrictlyPositiveFinite,
}

impl RepetitionPenaltyLogitsProcessor {
    pub fn process(&self, input_ids: ArrayD<i64>, scores: ArrayD<f32>) -> ArrayD<f32> {
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
