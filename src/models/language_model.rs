//! `language_model.onnx`: text embeddings + conditioning → speech tokens, generated
//! autoregressively via a KV cache.

use crate::models::model::{self, Metadata as BaseMetadata, Precision};
use ndarray::{Array, Array2, Array4, ArrayD, Ix4};
use num_traits::Float;
use ort::{
    session::{Session, builder::SessionBuilder},
    value::Tensor,
};

/// Identifies which `language_model` ONNX graph to load.
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

/// A loaded `language_model.onnx` session, holding its own KV cache between autoregressive steps.
#[derive(Debug)]
pub struct Model<F: Float> {
    metadata: Metadata<F>,
    session: Session,
    num_kv_heads: usize,
    head_dim: usize,
    // Native `F` — the KV cache is the only part of this graph confirmed to ever vary from
    // `float32` (depending on variant), so this is the one tensor pair actually worth being
    // generic over; everything else (`inputs_embeds`/`logits`) is hardcoded `f32` below.
    past_key_values: Vec<(String, Array4<F>)>,
}

impl<F: Float + 'static> model::Model<F> for Model<F> {
    fn session(&self) -> &Session {
        &self.session
    }

    fn metadata(&self) -> Box<dyn model::Metadata<F>> {
        Box::new(self.metadata)
    }
}

impl<P: Precision> Model<P> {
    /// Loads the model with a default `ort` session builder. `num_kv_heads`/`head_dim` describe
    /// the shape of this graph's KV cache tensors.
    pub fn load(
        metadata: Metadata<P>,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, model::Error> {
        Self::load_with_builder(metadata, Session::builder()?, num_kv_heads, head_dim)
    }

    /// Loads the model with a caller-supplied session builder (e.g. to configure an execution
    /// provider).
    pub fn load_with_builder(
        metadata: Metadata<P>,
        mut builder: SessionBuilder,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, model::Error> {
        Ok(Self {
            metadata,
            session: builder.commit_from_file(metadata.graph_file())?,
            num_kv_heads,
            head_dim,
            past_key_values: Vec::new(),
        })
    }

    /// Discovers this graph's `past_key_values.*` inputs and seeds each with a zero-length cache
    /// tensor. Done by inspecting `session.inputs()` rather than hardcoding a layer count, so this
    /// stays correct regardless of how many layers a given variant's graph actually has.
    pub(crate) fn init_past_key_values(&mut self, batch_size: usize) {
        self.past_key_values.clear();
        for input in self
            .session
            .inputs()
            .iter()
            .filter(|i| i.name().contains("past_key_values"))
        {
            self.past_key_values.push((
                input.name().to_string(),
                Array4::<P>::zeros(Ix4(batch_size, self.num_kv_heads, 0, self.head_dim)),
            ));
        }
    }

    /// Initializes `attention_mask` (all ones — nothing is padded/masked at the start of
    /// generation) and `position_ids` (`0..seq_len` broadcast across the batch) for the very first
    /// autoregressive step.
    pub(crate) fn init_mask(batch_size: usize, seq_len: usize) -> (Array2<i64>, Array2<i64>) {
        let attention_mask = Array2::ones((batch_size, seq_len));
        let position_ids = Array::from_iter(0..seq_len as i64)
            .broadcast((batch_size, seq_len))
            .expect("broadcast should not fail")
            .to_owned();
        (attention_mask, position_ids)
    }

    /// Runs one autoregressive step. `inputs_embeds`/`logits` are hardcoded `f32` regardless of
    /// `P` — confirmed empirically that this graph's KV cache is the only part of `language_model`
    /// that ever varies from `float32`, for every variant checked. The KV cache itself is updated
    /// in place from this step's `present_key_values` output, so callers don't need to manage it.
    pub(crate) fn step_language_model(
        &mut self,
        inputs_embeds: ArrayD<f32>,
        attention_mask: &Array2<i64>,
        position_ids: &Array2<i64>,
    ) -> Result<ArrayD<f32>, model::Error> {
        let mut inputs = ort::inputs![
            "inputs_embeds" => Tensor::from_array(inputs_embeds)?,
            "attention_mask" => Tensor::from_array(attention_mask.clone())?,
            "position_ids" => Tensor::from_array(position_ids.clone())?,
        ];
        for (name, kv) in &self.past_key_values {
            inputs.push((name.as_str().into(), Tensor::from_array(kv.clone())?.into()));
        }

        let outputs = self.session.run(inputs)?;
        let logits = outputs[0].try_extract_array::<f32>()?.into_owned();
        for ((_, kv), present) in self.past_key_values.iter_mut().zip(outputs.values().skip(1)) {
            *kv = present
                .try_extract_array::<P>()?
                .into_owned()
                .into_dimensionality::<Ix4>()
                .expect("KV cache tensor should be 4D");
        }
        Ok(logits)
    }
}
