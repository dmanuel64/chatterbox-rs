//! Chatterbox-Turbo's `language_model.onnx`: text embeddings + conditioning to
//! speech tokens, generated autoregressively via a KV cache.

use crate::models::model::{self, Metadata as BaseMetadata, Precision};
use ndarray::{Array, Array2, Array4, ArrayD, Ix4};
use num_traits::Float;
use ort::{
    environment::Environment,
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, Session, builder::SessionBuilder},
    value::{DynValue, Tensor},
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

/// Where the language model's KV cache is kept between steps. Should match the
/// device that actually runs the graph's attention (`GroupQueryAttention`),
/// which reads and writes the cache.
///
/// Use [`KvCacheDevice::Cuda`] only with an fp16 graph (`fp16`/`q4f16`) on the
/// CUDA execution provider. ONNX Runtime's CUDA attention kernel only supports
/// fp16/bf16, so with an fp32 graph (`fp32`/`int8`/`q4`) attention falls back
/// to the CPU even on CUDA, and a GPU-resident cache would be copied to the CPU
/// and back on every step. Use [`KvCacheDevice::Cpu`] in every other case.
#[derive(Debug, Clone, Copy, Default)]
pub enum KvCacheDevice {
    #[default]
    Cpu,
    Cuda {
        device_id: i32,
    },
}

impl KvCacheDevice {
    fn memory_info(self) -> Result<MemoryInfo<'static>, ort::Error> {
        match self {
            Self::Cpu => Ok(MemoryInfo::default()),
            Self::Cuda { device_id } => MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id,
                AllocatorType::Device,
                MemoryType::Default,
            ),
        }
    }
}

#[derive(Debug)]
struct KvSlot {
    input_name: String,
    output_name: String,
    value: DynValue,
}

/// A loaded `language_model.onnx` session, holding its own KV cache between
/// autoregressive steps.
#[derive(Debug)]
pub struct Model<F: Float> {
    metadata: Metadata<F>,
    session: Session,
    binding: IoBinding,
    logits_name: String,
    kv_mem: MemoryInfo<'static>,
    host_mem: MemoryInfo<'static>,
    num_kv_heads: usize,
    head_dim: usize,
    past_key_values: Vec<KvSlot>,
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
    /// Loads the model with a default `ort` [`SessionBuilder`].
    /// `num_kv_heads`/`head_dim` describe the shape of this graph's KV cache
    /// tensors.
    pub fn load(
        environment: &Environment,
        kv_device: KvCacheDevice,
        metadata: Metadata<P>,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, model::Error> {
        Self::load_with_builder(
            kv_device,
            metadata,
            Session::builder(environment)?,
            num_kv_heads,
            head_dim,
        )
    }

    /// Loads the model with a caller-supplied [`SessionBuilder`] (e.g. to
    /// configure an execution provider).
    pub fn load_with_builder(
        kv_device: KvCacheDevice,
        metadata: Metadata<P>,
        mut builder: SessionBuilder,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self, model::Error> {
        let session = builder.commit_from_file(metadata.graph_file())?;
        let binding = session.create_binding()?;
        let logits_name = session.outputs()[0].name().to_string();
        Ok(Self {
            metadata,
            session,
            binding,
            logits_name,
            kv_mem: kv_device.memory_info()?,
            host_mem: MemoryInfo::default(),
            num_kv_heads,
            head_dim,
            past_key_values: Vec::new(),
        })
    }

    /// Discovers this graph's `past_key_values.*` inputs and seeds each with a
    /// zero-length cache tensor. Done by inspecting `session.inputs()` rather
    /// than hardcoding a layer count, so this stays correct regardless of how
    /// many layers a given variant's graph actually has.
    pub(crate) fn init_past_key_values(&mut self, batch_size: usize) -> Result<(), model::Error> {
        self.past_key_values.clear();
        for input in self
            .session
            .inputs()
            .iter()
            .filter(|i| i.name().contains("past_key_values"))
        {
            let input_name = input.name().to_string();
            let output_name = input_name.replacen("past_key_values", "present", 1);
            if !self
                .session
                .outputs()
                .iter()
                .any(|o| o.name() == output_name)
            {
                return Err(ort::Error::new(format!(
                    "no output `{output_name}` matching KV cache input `{input_name}`"
                ))
                .into());
            }
            let empty = Array4::<P>::zeros(Ix4(batch_size, self.num_kv_heads, 0, self.head_dim));
            self.past_key_values.push(KvSlot {
                input_name,
                output_name,
                value: Tensor::from_array(empty)?.into_dyn(),
            });
        }
        Ok(())
    }

    pub(crate) fn init_mask(batch_size: usize, seq_len: usize) -> (Array2<i64>, Array2<i64>) {
        let attention_mask = Array2::ones((batch_size, seq_len));
        let position_ids = Array::from_iter(0..seq_len as i64)
            .broadcast((batch_size, seq_len))
            .expect("broadcast should not fail")
            .to_owned();
        (attention_mask, position_ids)
    }

    pub(crate) fn step_language_model(
        &mut self,
        inputs_embeds: ArrayD<f32>,
        attention_mask: &Array2<i64>,
        position_ids: &Array2<i64>,
    ) -> Result<ArrayD<f32>, model::Error> {
        // Inputs: small per-step tensors are copied to the device here; the KV
        // cache is already there, so binding it copies nothing.
        self.binding
            .bind_input("inputs_embeds", &Tensor::from_array(inputs_embeds)?)?;
        self.binding.bind_input(
            "attention_mask",
            &Tensor::from_array(attention_mask.clone())?,
        )?;
        self.binding
            .bind_input("position_ids", &Tensor::from_array(position_ids.clone())?)?;
        for slot in &self.past_key_values {
            self.binding
                .bind_input(slot.input_name.as_str(), &slot.value)?;
        }

        // Outputs: rebind every step. Logits come back to the host for
        // sampling; the new cache stays on the KV device.
        self.binding.clear_outputs();
        self.binding
            .bind_output_to_device(self.logits_name.as_str(), &self.host_mem)?;
        for slot in &self.past_key_values {
            self.binding
                .bind_output_to_device(slot.output_name.as_str(), &self.kv_mem)?;
        }

        let mut outputs = self.session.run_binding(&self.binding)?;
        let logits = outputs[self.logits_name.as_str()]
            .try_extract_array::<f32>()?
            .into_owned();
        for slot in &mut self.past_key_values {
            slot.value = outputs
                .remove(&slot.output_name)
                .expect("output presence checked in init_past_key_values");
        }
        Ok(logits)
    }
}
