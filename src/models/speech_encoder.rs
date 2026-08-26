use num_traits::Float;
use ort::session::{Session, builder::SessionBuilder};

use crate::models::model::{self, Metadata as BaseMetadata};

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

impl<F: Float> Model<F> {
    pub fn load(metadata: Metadata<F>) -> Result<Self, model::Error> {
        Self::load_with_builder(metadata, Session::builder()?)
    }

    pub fn load_with_builder(
        metadata: Metadata<F>,
        builder: SessionBuilder,
    ) -> Result<Self, model::Error> {
        Ok(Self {
            metadata,
            session: builder.commit_from_file(metadata.graph_file())?,
        })
    }

    pub(crate) fn encode_reference_audio(&self) {
        // let ort_speech_encoder_input = ort::inputs![
        //     "audio_values" => float_input(
        //         audio_values.clone(),
        //         self.speech_encoder_audio_values_fp16
        //     )?
        // ];
        // let speech_encoder_outputs = self.speech_encoder_session.run(ort_speech_encoder_input)?;
        // let condition_embeddings = extract_f32_array(
        //     &speech_encoder_outputs[0],
        //     self.speech_encoder_condition_embeddings_fp16,
        // )?;
        // prompt_token = Some(speech_encoder_outputs[1].try_extract_array()?.to_owned());
        // speaker_embeddings = Some(extract_f32_array(
        //     &speech_encoder_outputs[2],
        //     self.speech_encoder_speaker_embeddings_fp16,
        // )?);
        // speaker_features = Some(extract_f32_array(
        //     &speech_encoder_outputs[3],
        //     self.speech_encoder_speaker_features_fp16,
        // )?);
    }
}
