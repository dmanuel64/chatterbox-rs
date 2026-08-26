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
