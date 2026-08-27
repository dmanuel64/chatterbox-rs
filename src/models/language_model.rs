use crate::models::model::{self, Metadata as BaseMetadata};
use ndarray::{Array4, Ix4};
use num_traits::Float;
use ort::session::{Session, builder::SessionBuilder};

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

#[derive(Debug)]
pub struct Model<F: Float> {
    metadata: Metadata<F>,
    session: Session,
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
            past_key_values: Vec::new(),
        })
    }

    pub(crate) fn init_past_key_values(&mut self) {
        for input in self
            .session
            .inputs()
            .iter()
            .filter(|i| i.name().contains("past_key_values"))
        {
            self.past_key_values.push((
                input.name().to_string(),
                Array4::<F>::zeros(Ix4(batch_size, self.num_kv_heads, 0, self.head_dim)),
            ));
        }
    }

    pub(crate) fn init_mask(&self) {
        // attention_mask = Array::ones(Ix2(batch_size, seq_len));
        // position_ids = Array::from_iter(0..seq_len as i64)
        //     .broadcast((batch_size, seq_len))
        //     .expect("broadcast should not fail")
        //     .to_owned();
    }

    pub(crate) fn step_language_model(&self) {
        // let mut language_model_inputs = ort::inputs![
        //     "inputs_embeds" => float_input(input_embeds, self.language_model_inputs_embeds_fp16)?,
        //     "attention_mask" => Tensor::from_array(attention_mask.clone())?,
        //     "position_ids" => Tensor::from_array(position_ids.clone())?,
        // ];
        // for (name, fp16, kv) in &past_key_values {
        //     language_model_inputs.push((name.as_str().into(), float_input(kv.clone(), *fp16)?));
        // }
        // let language_model_outputs = self.language_model_session.run(language_model_inputs)?;
        // let logits =
        //     extract_f32_array(&language_model_outputs[0], self.language_model_logits_fp16)?;
        // let present_key_values: Vec<Array4<f32>> = language_model_outputs
        //     .values()
        //     .skip(1)
        //     .zip(past_key_values.iter())
        //     .map(|(v, (_, fp16, _))| {
        //         extract_f32_array(&v, *fp16).map(|a| {
        //             a.into_dimensionality::<Ix4>()
        //                 .expect("KV cache tensor should be 4D")
        //         })
        //     })
        //     .collect::<Result<_, _>>()?;
    }
}
