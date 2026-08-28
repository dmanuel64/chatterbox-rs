use ndarray::{ArrayD, ArrayRefD, Axis, Ix2, concatenate, prelude::*};
use num_traits::Float;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

use crate::models::model::{self, Metadata as BaseMetadata};

/// Speech-token id `conditional_decoder.onnx` expects appended as trailing silence padding after
/// the real generated speech tokens.
const SILENCE_TOKEN: i64 = 4299;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata<F: Float> {
    pub variant: model::Variant<F>,
}

impl<F: Float + 'static> model::Metadata<F> for Metadata<F> {
    fn filename_prefix(&self) -> &'static str {
        "conditional_decoder"
    }

    fn variant(&self) -> model::Variant<F> {
        self.variant
    }
}

#[derive(Debug)]
pub struct Model<F: Float> {
    metadata: Metadata<F>,
    session: Session,
}

impl<F: Float + 'static> model::Model<F> for Model<F> {
    fn session(&self) -> &Session {
        &self.session
    }

    fn metadata(&self) -> Box<dyn model::Metadata<F>> {
        Box::new(self.metadata)
    }
}

impl<F: Float + 'static> Model<F> {
    pub fn load(metadata: Metadata<F>) -> Result<Self, model::Error> {
        Self::load_with_builder(metadata, Session::builder()?)
    }

    pub fn load_with_builder(
        metadata: Metadata<F>,
        mut builder: SessionBuilder,
    ) -> Result<Self, model::Error> {
        Ok(Self {
            metadata,
            session: builder.commit_from_file(metadata.graph_file())?,
        })
    }

    /// `generate_tokens` is the full running sequence including the seed `START_SPEECH_TOKEN` and
    /// trailing `STOP_SPEECH_TOKEN` — both get sliced off here, leaving only the real generated
    /// speech tokens, which get prepended with `prompt_token` (the reference clip's own speech
    /// tokens) and padded with a few trailing silence tokens before decoding.
    pub(crate) fn decode_audio(
        &mut self,
        prompt_token: ArrayD<i64>,
        generate_tokens: &ArrayRefD<i64>,
        speaker_embeddings: ArrayD<f32>,
        speaker_features: ArrayD<f32>,
    ) -> Result<ArrayD<f32>, model::Error> {
        let speech_tokens = generate_tokens.slice(s![.., 1..-1]).into_dyn();
        let silence_tokens =
            Array2::<i64>::from_elem(Ix2(speech_tokens.shape()[0], 3), SILENCE_TOKEN).into_dyn();
        let speech_tokens = concatenate![Axis(1), prompt_token, speech_tokens, silence_tokens];

        let outputs = self.session.run(ort::inputs![
            "speech_tokens" => Tensor::from_array(speech_tokens)?,
            "speaker_embeddings" => Tensor::from_array(speaker_embeddings)?,
            "speaker_features" => Tensor::from_array(speaker_features)?
        ])?;
        // TODO: they have it as squeeze(axis=0)
        let wav = outputs[0].try_extract_array::<f32>()?.squeeze();
        Ok(wav.to_owned())
    }
}
