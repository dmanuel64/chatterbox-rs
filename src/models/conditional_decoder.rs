use ort::session::{Session, builder::SessionBuilder};

use crate::models::model::{self, Metadata as BaseMetadata};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata {
    pub variant: model::Variant<f32>,
}

impl model::Metadata<f32> for Metadata {
    fn filename_prefix(&self) -> &'static str {
        "conditional_decoder"
    }

    fn variant(&self) -> model::Variant<f32> {
        self.variant
    }
}

#[derive(Debug)]
pub struct Model {
    metadata: Metadata,
    session: Session,
}

impl model::Model<f32> for Model {
    fn session(&self) -> &Session {
        &self.session
    }

    fn metadata(&self) -> Box<dyn model::Metadata<f32>> {
        Box::new(self.metadata)
    }
}

impl Model {
    pub fn load(metadata: Metadata) -> Result<Self, Error> {
        Self::load_with_builder(metadata, Session::builder()?)
    }

    pub fn load_with_builder(metadata: Metadata, builder: SessionBuilder) -> Result<Self, Error> {
        Ok(Self {
            metadata,
            session: builder.commit_from_file(metadata.graph_file())?,
        })
    }

    pub(crate) fn decode_audio(&mut self) {
        // let speech_tokens = generate_tokens.slice(s![.., 1..-1]).into_dyn();
        // let silence_tokens =
        //     Array2::<i64>::from_elem(Ix2(speech_tokens.shape()[0], 3), Self::SILENCE_TOKEN)
        //         .into_dyn();
        // let speech_tokens = concatenate![Axis(1), prompt_token, speech_tokens, silence_tokens];

        // let output = self.conditional_decoder_session.run(ort::inputs![
        //     "speech_tokens" => Tensor::from_array(speech_tokens)?,
        //     "speaker_embeddings" => float_input(
        //         speaker_embeddings,
        //         self.conditional_decoder_speaker_embeddings_fp16
        //     )?,
        //     "speaker_features" => float_input(
        //         speaker_features,
        //         self.conditional_decoder_speaker_features_fp16
        //     )?
        // ])?;
        // // TODO: they have it as squeeze(axis=0)
        // let wav = extract_f32_array(&output[0], self.conditional_decoder_wav_fp16)?.squeeze();
        // Ok(wav.to_owned())
    }
}
