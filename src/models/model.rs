//! Shared traits and types every ONNX model wrapper (`speech_encoder`, `token_embedder`,
//! `language_model`, `conditional_decoder`) is built on: [`Model`]/[`Metadata`] for locating and
//! describing a graph on disk, [`Variant`] for naming a graph's quantization/precision, and the
//! [`Precision`]/[`RestrictedPrecision`] bounds controlling which Rust float types a model can
//! actually be instantiated with.

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

/// Errors that can occur while loading or running a model.
#[derive(Debug, Error)]
pub enum Error {
    /// The requested `kind`/`float_type` combination has no matching exported graph.
    #[error("incompatible model variant: TODO")]
    IncompatibleVariant { kind: Kind, float_type: TypeId },
    /// The underlying `ort` session failed to load or run.
    #[error("An ONNX runtime error occurred: {0}")]
    Onnx(#[from] ort::Error),
}

/// Whether a graph's weights are quantized, and if so how.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Kind {
    /// Unquantized weights.
    Baseline,
    /// Weights quantized to `weight_packing`.
    Quantized {
        /// How the quantized weights are bit-packed.
        weight_packing: WeightPacking,
    },
}

impl Default for Kind {
    fn default() -> Self {
        Self::Quantized {
            weight_packing: WeightPacking::Bit8,
        }
    }
}

/// How a quantized graph's weights are bit-packed.
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum WeightPacking {
    /// 8-bit weights.
    #[default]
    Bit8,
    /// 4-bit weights.
    Bit4,
}

/// A named combination of quantization [`Kind`] and Rust float type, identifying one specific
/// exported ONNX graph (e.g. `int8` at `F = f32`, or `q4_fp16` at `F = f16`).
#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub struct Variant<F: Float> {
    kind: Kind,
    _phanton: PhantomData<F>,
}

impl Variant<f32> {
    /// Unquantized, [`f32`] weights.
    pub const FP32: Self = Self::new_inner(Kind::Baseline);
    /// 8-bit-quantized weights, [`f32`] activations.
    pub const INT8: Self = Self::new_inner(Kind::Quantized {
        weight_packing: WeightPacking::Bit8,
    });
    /// 4-bit-quantized weights, [`f32`] activations.
    pub const Q4: Self = Self::new_inner(Kind::Quantized {
        weight_packing: WeightPacking::Bit4,
    });
}

impl Variant<f16> {
    /// Unquantized weights, `f16` activations.
    pub const FP16: Variant<f16> = Variant::new_inner(Kind::Baseline);
    /// 4-bit-quantized weights, `f16` activations.
    pub const Q4_FP16: Variant<f16> = Variant::new_inner(Kind::Quantized {
        weight_packing: WeightPacking::Bit4,
    });
}

impl<F: Float + 'static> Variant<F> {
    /// Builds a variant for any [`Precision`] `F`. Only available with the `custom-variants`
    /// feature, since the official graphs are only ever proven to exist as [`f32`] or [`f16`]. Use
    /// the associated constants (e.g. [`Variant::<f32>::FP32`]) for those.
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

/// Identifies which on-disk ONNX graph a model wraps: a filename prefix (unique per component)
/// plus the [`Variant`] to load.
pub trait Metadata<F: Float + 'static> {
    /// The component's base filename, e.g. `"language_model"`.
    fn filename_prefix(&self) -> &'static str;
    /// The quantization/precision variant to load.
    fn variant(&self) -> Variant<F>;

    /// The full filename (no extension), combining [`Self::filename_prefix`] and the variant's
    /// suffix.
    fn filename(&self) -> String {
        format!(
            "{}{}",
            self.filename_prefix(),
            self.variant().filename_suffix()
        )
    }

    /// Path to this graph's `.onnx` file under [`config::ONNX_DIR`].
    fn graph_file(&self) -> PathBuf {
        let onnx_dir = config::ONNX_DIR
            .read()
            .expect("ONNX_DIR lock to not be poisoned");
        onnx_dir.join(PathBuf::from(self.filename()).with_extension("onnx"))
    }

    /// Path to this graph's external `.onnx_data` weights file under [`config::ONNX_DIR`].
    #[allow(unused)]
    fn weights_file(&self) -> PathBuf {
        let onnx_dir = config::ONNX_DIR
            .read()
            .expect("ONNX_DIR lock to not be poisoned");
        onnx_dir.join(PathBuf::from(self.filename()).with_extension("onnx_data"))
    }
}

/// A loaded ONNX model wrapper. Implemented by each component's own `Model<P>` type
/// (`speech_encoder::Model`, `token_embedder::Model`, `language_model::Model`,
/// `conditional_decoder::Model`).
pub trait Model<F: Float> {
    /// The underlying `ort` session.
    fn session(&self) -> &Session;
    /// The metadata this model was loaded with.
    fn metadata(&self) -> Box<dyn Metadata<F>>;
}

impl<F: Float + 'static> Display for dyn Model<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let variant = self.metadata().variant();
        write!(f, "{variant}")
    }
}

trait_set::trait_set! {
    /// Floating-point types `ort` can build a tensor from. The baseline bound for every model;
    /// [`RestrictedPrecision`] narrows this further for components proven to always be [`f32`].
    pub trait Precision = Float + PrimitiveTensorElementType + Debug + 'static;
}

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

/// Converts an array of any [`RestrictedPrecision`] to [`f32`], for combining tensors from two
/// components whose own activation types may differ (e.g. `speech_encoder`'s `S` and
/// `token_embedder`'s `T`) before handing them to a component with a fixed, proven dtype
/// requirement of its own (`language_model`'s `inputs_embeds`, always [`f32`] regardless of variant).
pub fn to_f32<F: num_traits::ToPrimitive + Clone, D: ndarray::Dimension>(
    array: ndarray::Array<F, D>,
) -> ndarray::Array<f32, D> {
    array.mapv(|x| x.to_f32().expect("value should be representable as f32"))
}

/// The inverse of [`to_f32`]: converts [`f32`] back into whatever [`RestrictedPrecision`] a
/// downstream component's own tensors actually need.
pub fn from_f32<F: num_traits::FromPrimitive, D: ndarray::Dimension>(
    array: ndarray::Array<f32, D>,
) -> ndarray::Array<F, D> {
    array.mapv(|x| F::from_f32(x).expect("value should be representable as F"))
}
