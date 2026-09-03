//! `conditional_decoder.onnx`: speech tokens + speaker embedding/features to waveform, in one
//! distilled step (no separate mel/vocoder split is exposed by the exported graph).

use ndarray::{ArrayD, ArrayRefD, Axis, Ix2, concatenate, prelude::*};
use num_traits::Float;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

use crate::models::model::{self, Metadata as BaseMetadata, RestrictedPrecision};

/// Speech-token id `conditional_decoder.onnx` expects appended as trailing silence padding after
/// the real generated speech tokens.
const SILENCE_TOKEN: i64 = 4299;

/// Identifies which `conditional_decoder` ONNX graph to load.
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

/// A loaded `conditional_decoder.onnx` session.
#[derive(Debug)]
pub struct Model<P: RestrictedPrecision> {
    metadata: Metadata<P>,
    session: Session,
}

impl<P: RestrictedPrecision> model::Model<P> for Model<P> {
    fn session(&self) -> &Session {
        &self.session
    }

    fn metadata(&self) -> Box<dyn model::Metadata<P>> {
        Box::new(self.metadata)
    }
}

impl<P: RestrictedPrecision> Model<P> {
    /// Loads the model with a default `ort` [`SessionBuilder`].
    pub fn load(metadata: Metadata<P>) -> Result<Self, model::Error> {
        Self::load_with_builder(metadata, Session::builder()?)
    }

    /// Loads the model with a caller-supplied [`SessionBuilder`] (e.g. to configure an execution
    /// provider).
    pub fn load_with_builder(
        metadata: Metadata<P>,
        mut builder: SessionBuilder,
    ) -> Result<Self, model::Error> {
        Ok(Self {
            metadata,
            session: builder.commit_from_file(metadata.graph_file())?,
        })
    }

    pub(crate) fn decode_audio(
        &mut self,
        prompt_token: ArrayD<i64>,
        generate_tokens: &ArrayRefD<i64>,
        speaker_embeddings: ArrayD<P>,
        speaker_features: ArrayD<P>,
    ) -> Result<ArrayD<P>, model::Error> {
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
        let wav = outputs[0].try_extract_array::<P>()?.squeeze();
        Ok(wav.to_owned())
    }
}
