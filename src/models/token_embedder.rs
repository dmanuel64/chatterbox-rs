//! Chatterbox-Turbo's `embed_tokens.onnx`: token ids to embeddings, fed into
//! `language_model` as `inputs_embeds`.

use crate::models::model::{self, Metadata as BaseMetadata, RestrictedPrecision};
use ndarray::{Array2, ArrayD};
use num_traits::Float;
use ort::{
    environment::Environment,
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

/// Identifies which `embed_tokens` ONNX graph to load.
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

/// A loaded `embed_tokens.onnx` session.
#[derive(Debug)]
pub struct Model<P: RestrictedPrecision> {
    metadata: Metadata<P>,
    session: Session,
    speech_table: Option<Array2<f32>>,
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
    pub fn load(environment: &Environment, metadata: Metadata<P>) -> Result<Self, model::Error> {
        Self::load_with_builder(metadata, Session::builder(environment)?)
    }

    /// Loads the model with a caller-supplied [`SessionBuilder`] (e.g. to
    /// configure an execution provider).
    pub fn load_with_builder(
        metadata: Metadata<P>,
        mut builder: SessionBuilder,
    ) -> Result<Self, model::Error> {
        Ok(Self {
            metadata,
            session: builder.commit_from_file(metadata.graph_file())?,
            speech_table: None,
        })
    }

    pub(crate) fn embed_tokens(
        &mut self,
        input_ids: ArrayD<i64>,
    ) -> Result<ArrayD<P>, model::Error> {
        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(input_ids)?
        ])?;
        let output = outputs[0].try_extract_array::<P>()?;
        Ok(output.into_owned())
    }

    /// Embeds every id in `0..num_ids` once and keeps the results as an f32
    /// lookup table. Each id is its own length-1 sequence (shape `[num_ids, 1]`),
    /// exactly how generation calls this graph one token at a time.
    pub(crate) fn build_speech_table(&mut self, num_ids: usize) -> Result<(), model::Error> {
        let ids = Array2::from_shape_fn((num_ids, 1), |(i, _)| i as i64).into_dyn();
        let embeds = model::to_f32(self.embed_tokens(ids)?); // [num_ids, 1, hidden]
        let hidden = *embeds
            .shape()
            .last()
            .expect("embeddings should have a hidden dim");
        let table = embeds
            .into_shape_with_order((num_ids, hidden))
            .expect("one embedding row per id");
        self.speech_table = Some(table);
        Ok(())
    }

    /// Embeds speech tokens during generation: a table lookup when every id is in
    /// the table, falling back to running the graph otherwise.
    pub(crate) fn embed_speech_tokens(
        &mut self,
        input_ids: &ArrayD<i64>,
    ) -> Result<ArrayD<f32>, model::Error> {
        if let Some(table) = &self.speech_table
            && input_ids
                .iter()
                .all(|id| (0..table.nrows() as i64).contains(id))
        {
            let hidden = table.ncols();
            let mut out = Vec::with_capacity(input_ids.len() * hidden);
            for &id in input_ids {
                out.extend_from_slice(
                    table
                        .row(id as usize)
                        .as_slice()
                        .expect("table rows are contiguous"),
                );
            }
            let mut shape = input_ids.shape().to_vec();
            shape.push(hidden);
            return Ok(ArrayD::from_shape_vec(shape, out).expect("one row per id"));
        }
        Ok(model::to_f32(self.embed_tokens(input_ids.clone())?))
    }
}
