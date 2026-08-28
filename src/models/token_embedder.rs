use crate::models::model::{self, Metadata as BaseMetadata};
use ndarray::ArrayD;
use num_traits::Float;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata<F: Float> {
    pub variant: model::Variant<F>,
}

impl<F: Float + 'static> model::Metadata<F> for Metadata<F> {
    fn filename_prefix(&self) -> &'static str {
        "embed_tokens"
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

    pub(crate) fn embed_tokens(
        &mut self,
        input_ids: ArrayD<i64>,
    ) -> Result<ArrayD<f32>, model::Error> {
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?
        ])?;
        let output = outputs[0].try_extract_array::<f32>()?;
        Ok(output.into_owned())
    }
}
