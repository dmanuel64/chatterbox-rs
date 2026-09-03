//! `speech_encoder.onnx`: reference audio → conditioning embedding, prompt tokens, and speaker
//! embedding/features.

use ndarray::ArrayD;
use num_traits::Float;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

use crate::models::model::{self, Metadata as BaseMetadata, RestrictedPrecision};

/// The four outputs of `speech_encoder.onnx`, read positionally — the exported graph doesn't have
/// stable output names. The three floating-point outputs are typed `P`, matching whatever
/// [`RestrictedPrecision`] this model was loaded with (always `f32` for the official graphs, since
/// only `f32` implements [`RestrictedPrecision`] without the `custom-variants` feature).
pub(crate) struct ReferenceAudioEncoding<P> {
    /// T3 conditioning embedding, prepended to the text embedding stream.
    pub condition_embeddings: ArrayD<P>,
    /// The reference clip's own speech tokens, prepended to the generated ones in `decode_audio`.
    pub prompt_token: ArrayD<i64>,
    pub speaker_embeddings: ArrayD<P>,
    pub speaker_features: ArrayD<P>,
}

/// Identifies which `speech_encoder` ONNX graph to load.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata<F: Float> {
    pub variant: model::Variant<F>,
}

impl<F: Float + 'static> model::Metadata<F> for Metadata<F> {
    fn filename_prefix(&self) -> &'static str {
        "speech_encoder"
    }

    fn variant(&self) -> model::Variant<F> {
        self.variant
    }
}

/// A loaded `speech_encoder.onnx` session.
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
    /// Loads the model with a default `ort` session builder.
    pub fn load(metadata: Metadata<P>) -> Result<Self, model::Error> {
        Self::load_with_builder(metadata, Session::builder()?)
    }

    /// Loads the model with a caller-supplied session builder (e.g. to configure an execution
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

    pub(crate) fn encode_reference_audio(
        &mut self,
        audio_values: ArrayD<P>,
    ) -> Result<ReferenceAudioEncoding<P>, model::Error> {
        let outputs = self
            .session
            .run(ort::inputs!["audio_values" => Tensor::from_array(audio_values)?])?;
        Ok(ReferenceAudioEncoding {
            condition_embeddings: outputs[0].try_extract_array::<P>()?.into_owned(),
            prompt_token: outputs[1].try_extract_array::<i64>()?.into_owned(),
            speaker_embeddings: outputs[2].try_extract_array::<P>()?.into_owned(),
            speaker_features: outputs[3].try_extract_array::<P>()?.into_owned(),
        })
    }
}
