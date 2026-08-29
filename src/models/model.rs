use crate::config;
use half::f16;
use num_traits::{Float, FromPrimitive};
use ort::{session::Session, value::PrimitiveTensorElementType};
use std::{
    any::TypeId,
    fmt::{Debug, Display},
    marker::PhantomData,
    mem::size_of,
    path::PathBuf,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("incompatible model variant: TODO")]
    IncompatibleVariant { kind: Kind, float_type: TypeId },
    #[error("An ONNX runtime error occurred: {0}")]
    Onnx(#[from] ort::Error),
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Kind {
    Baseline,
    Quantized { weight_packing: WeightPacking },
}

impl Default for Kind {
    fn default() -> Self {
        Self::Quantized {
            weight_packing: WeightPacking::Bit8,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum WeightPacking {
    #[default]
    Bit8,
    Bit4,
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub struct Variant<F: Float> {
    kind: Kind,
    _phanton: PhantomData<F>,
}

impl Variant<f32> {
    pub const FP32: Self = Self::new_inner(Kind::Baseline);
    pub const INT8: Self = Self::new_inner(Kind::Quantized {
        weight_packing: WeightPacking::Bit8,
    });
    pub const Q4: Self = Self::new_inner(Kind::Quantized {
        weight_packing: WeightPacking::Bit4,
    });
}

impl Variant<f16> {
    pub const FP16: Variant<f16> = Variant::new_inner(Kind::Baseline);
    pub const Q4_FP16: Variant<f16> = Variant::new_inner(Kind::Quantized {
        weight_packing: WeightPacking::Bit4,
    });
}

impl<F: Float + 'static> Variant<F> {
    #[cfg(feature = "custom-variants")]
    pub const fn new(kind: Kind) -> Self {
        Self::new_inner(kind)
    }

    const fn new_inner(kind: Kind) -> Self {
        Self {
            kind,
            _phanton: PhantomData,
        }
    }

    fn filename_suffix(&self) -> String {
        let graph_size = size_of::<F>() * 8;
        let float_type = TypeId::of::<F>();
        match self.kind {
            Kind::Baseline => {
                if float_type != TypeId::of::<f32>() {
                    format!("_fp{graph_size}")
                } else {
                    String::new()
                }
            }
            Kind::Quantized { weight_packing } => {
                let mut o = if let WeightPacking::Bit8 = weight_packing {
                    String::from("_quantized")
                } else {
                    String::from("")
                };
                if float_type != TypeId::of::<f32>() {
                    o += &format!("_f{graph_size}");
                }
                o
            }
        }
    }
}

impl<F: Float + 'static> Display for Variant<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let graph_size = size_of::<F>() * 8;
        let out = match self.kind {
            Kind::Baseline => format!("fp{graph_size}"),
            Kind::Quantized { weight_packing } => {
                let mut o = if let WeightPacking::Bit8 = weight_packing {
                    String::from("int8")
                } else {
                    String::from("q4")
                };
                if TypeId::of::<F>() != TypeId::of::<f32>() {
                    o += &format!("_fp{graph_size}");
                }
                o
            }
        };
        write!(f, "{out}")
    }
}

pub trait Metadata<F: Float + 'static> {
    fn filename_prefix(&self) -> &'static str;
    fn variant(&self) -> Variant<F>;

    fn filename(&self) -> String {
        format!(
            "{}{}",
            self.filename_prefix(),
            self.variant().filename_suffix()
        )
    }

    fn graph_file(&self) -> PathBuf {
        let onnx_dir = config::ONNX_DIR
            .read()
            .expect("ONNX_DIR lock to not be poisoned");
        onnx_dir.join(PathBuf::from(self.filename()).with_extension("onnx"))
    }

    #[allow(unused)]
    fn weights_file(&self) -> PathBuf {
        let onnx_dir = config::ONNX_DIR
            .read()
            .expect("ONNX_DIR lock to not be poisoned");
        onnx_dir.join(PathBuf::from(self.filename()).with_extension("onnx_data"))
    }
}

pub trait Model<F: Float> {
    fn session(&self) -> &Session;
    fn metadata(&self) -> Box<dyn Metadata<F>>;
}

impl<F: Float + 'static> Display for dyn Model<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = self.metadata().variant();
        write!(f, "{variant}")
    }
}

// The float types ONNX Runtime can actually build a tensor from — every real model's generic
// bound needs at least this much, so it's pulled out once rather than spelled out at every site.
trait_set::trait_set! {
    pub trait Precision = Float + PrimitiveTensorElementType + Debug + 'static;
}

// A precision a model's own activation tensors (as opposed to its internally-quantized weights)
// can actually be constructed as. Applies to models that would otherwise have `f32` hardcoded —
// `speech_encoder`, `token_embedder`, `conditional_decoder` — since their official exported
// graphs only ever have `float32` tensors at the boundary; `language_model` isn't restricted by
// this (it's bounded by [`Precision`] directly), since its KV cache already needs real `float16`
// support unconditionally.
//
// Without `custom-variants`, only `f32` implements this — matching what the official graphs
// actually are. With `custom-variants`, this widens to any [`Precision`], same as `Variant::new`
// widens to accept any such type — so a hand-exported custom graph with activations in `f16`,
// `f64`, or any other supported width can be used directly.
#[cfg(feature = "custom-variants")]
trait_set::trait_set! {
    /// A precision a model's own activation tensors can actually be constructed as. See the
    /// module-level comment above this item for the full explanation.
    pub trait RestrictedPrecision = Precision + FromPrimitive;
}

#[cfg(not(feature = "custom-variants"))]
/// A precision a model's own activation tensors can actually be constructed as. See the
/// module-level comment above this item for the full explanation.
pub trait RestrictedPrecision: Precision + FromPrimitive {}
#[cfg(not(feature = "custom-variants"))]
impl RestrictedPrecision for f32 {}

/// Converts an array of any [`RestrictedPrecision`] to `f32`, for combining tensors from two
/// components whose own activation types may differ (e.g. `speech_encoder`'s `S` and
/// `token_embedder`'s `T`) before handing them to a component with a fixed, proven dtype
/// requirement of its own (`language_model`'s `inputs_embeds`, always `f32` regardless of variant).
pub fn to_f32<F: num_traits::ToPrimitive + Clone, D: ndarray::Dimension>(
    array: ndarray::Array<F, D>,
) -> ndarray::Array<f32, D> {
    array.mapv(|x| x.to_f32().expect("value should be representable as f32"))
}

/// The inverse of [`to_f32`] — converts `f32` back into whatever [`RestrictedPrecision`] a
/// downstream component's own tensors actually need.
pub fn from_f32<F: num_traits::FromPrimitive, D: ndarray::Dimension>(
    array: ndarray::Array<f32, D>,
) -> ndarray::Array<F, D> {
    array.mapv(|x| F::from_f32(x).expect("value should be representable as F"))
}
