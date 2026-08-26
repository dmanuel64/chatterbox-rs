use crate::models::model;

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Metadata {
    pub variant: model::Variant<f32>,
}

impl model::Metadata<f32> for Metadata {
    fn filename_prefix(&self) -> &'static str {
        "speech_encoder"
    }

    fn variant(&self) -> model::Variant<f32> {
        self.variant
    }
}

#[derive(Debug)]
pub struct Model {
    metadata: Metadata,
}

impl model::Model<f32> for Model {
    fn metadata(&self) -> Box<dyn model::Metadata<f32>> {
        Box::new(self.metadata)
    }
}
