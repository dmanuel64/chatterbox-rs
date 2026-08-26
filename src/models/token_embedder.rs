use crate::models::model::{self, Metadata as BaseMetadata};
use ndarray::ArrayD;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata {
    pub variant: model::Variant<f32>,
}

impl model::Metadata<f32> for Metadata {
    fn filename_prefix(&self) -> &'static str {
        "embed_tokens"
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

    pub(crate) fn embed_tokens(&mut self, input_ids: ArrayD<f32>) -> Result<ArrayD<f32>, Error> {
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?
        ])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        Ok(output.into_owned())
    }
}
