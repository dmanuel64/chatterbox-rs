use ndarray::ArrayD;
use num_traits::Float;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

use crate::models::model::{self, Metadata as BaseMetadata};

/// The four outputs of `speech_encoder.onnx`, read positionally — the exported graph doesn't have
/// stable output names. All four are `float32` regardless of variant (confirmed empirically: even
/// the `_fp16`/`_q4f16` files keep this graph's boundary entirely `float32`), so this struct is not
/// generic over `F` — `F` only selects which file gets loaded.
pub(crate) struct ReferenceAudioEncoding {
    /// T3 conditioning embedding, prepended to the text embedding stream.
    pub condition_embeddings: ArrayD<f32>,
    /// The reference clip's own speech tokens, prepended to the generated ones in `decode_audio`.
    pub prompt_token: ArrayD<i64>,
    pub speaker_embeddings: ArrayD<f32>,
    pub speaker_features: ArrayD<f32>,
}

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

    pub(crate) fn encode_reference_audio(
        &mut self,
        audio_values: ArrayD<f32>,
    ) -> Result<ReferenceAudioEncoding, model::Error> {
        let outputs = self
            .session
            .run(ort::inputs!["audio_values" => Tensor::from_array(audio_values)?])?;
        Ok(ReferenceAudioEncoding {
            condition_embeddings: outputs[0].try_extract_array::<f32>()?.into_owned(),
            prompt_token: outputs[1].try_extract_array::<i64>()?.into_owned(),
            speaker_embeddings: outputs[2].try_extract_array::<f32>()?.into_owned(),
            speaker_features: outputs[3].try_extract_array::<f32>()?.into_owned(),
        })
    }
}
