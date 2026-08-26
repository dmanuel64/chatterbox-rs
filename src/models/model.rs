use crate::config;
use half::f16;
use num_traits::Float;
use ort::session::Session;
use std::{
    any::{TypeId, type_name},
    fmt::Display,
    marker::PhantomData,
    path::PathBuf,
};
use thiserror::Error;

pub const FP32: Variant<f32> = Variant::new_inner(Kind::Baseline);
pub const FP16: Variant<f16> = Variant::new_inner(Kind::Baseline);
pub const INT8: Variant<f32> = Variant::new_inner(Kind::Quantized {
    weight_packing: WeightPacking::Bit8,
});
pub const Q4: Variant<f32> = Variant::new_inner(Kind::Quantized {
    weight_packing: WeightPacking::Bit4,
});
pub const Q4_FP16: Variant<f16> = Variant::new_inner(Kind::Quantized {
    weight_packing: WeightPacking::Bit4,
});

#[derive(Debug, Error)]
pub enum Error {
    #[error("incompatible model variant: TODO")]
    IncompatibleVariant { kind: Kind, float_type: TypeId },
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

impl<F: Float + 'static> Variant<F> {
    // TODO: refactor string to support custom variants
    // #[cfg(feature = "custom-variants")]
    // const fn new(kind: Kind) -> Self {
    //     Self::new_inner(kind)
    // }

    const fn new_inner(kind: Kind) -> Self {
        Self {
            kind,
            _phanton: PhantomData,
        }
    }

    fn filename_suffix(&self) -> &'static str {
        let is_half_precision = TypeId::of::<F>() == TypeId::of::<f16>();
        match self.kind {
            Kind::Baseline => {
                if is_half_precision {
                    "_fp16"
                } else {
                    ""
                }
            }
            Kind::Quantized { weight_packing } => {
                if let WeightPacking::Bit8 = weight_packing {
                    "_quantized"
                } else {
                    if is_half_precision { "_q4f16" } else { "_q4" }
                }
            }
        }
    }
}

impl<F: Float + 'static> Display for Variant<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let is_half_precision = TypeId::of::<F>() == TypeId::of::<f16>();
        let graph_size = &type_name::<F>()[2..];
        let out = match self.kind {
            Kind::Baseline => format!("fp{graph_size}"),
            Kind::Quantized { weight_packing } => {
                if let WeightPacking::Bit8 = weight_packing {
                    String::from("int8")
                } else {
                    let mut o = String::from("q4");
                    if is_half_precision {
                        o += &format!("_fp{graph_size}");
                    }
                    o
                }
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
