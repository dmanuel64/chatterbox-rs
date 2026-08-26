use crate::models::model;
use num_traits::Float;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata<F: Float> {
    pub variant: model::Variant<F>,
}

impl<F: Float + 'static> model::Metadata<F> for Metadata<F> {
    fn filename_prefix(&self) -> &'static str {
        "language_model"
    }

    fn variant(&self) -> model::Variant<F> {
        self.variant
    }
}

pub struct Model<F: Float> {
    metadata: Metadata<F>,
}

impl<F: Float + 'static> model::Model<F> for Model<F> {
    fn metadata(&self) -> Box<dyn model::Metadata<F>> {
        Box::new(self.metadata)
    }
}
