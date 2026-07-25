//! Shared host-equivalent numeric helpers for portable kernels.

use std::sync::Arc;

use sim_kernel::{Cx, NumberLiteral, Symbol, Value};
use sim_lib_numbers_tensor::{TensorExecError, number_literal_for_tensor_cell};

use crate::pipeline::WgpuKernelDType;

pub(crate) fn numeric_cell(cx: &mut Cx, value: &Value) -> Result<f32, TensorExecError> {
    let literal = number_literal_for_tensor_cell(value)
        .or_else(|| cx.number_value_ref(value.clone()).ok().flatten()?.literal)
        .ok_or_else(|| invalid("wgpu kernels expect numeric tensor cells"))?;
    parse_literal(literal)
}

pub(crate) fn number_value(
    cx: &mut Cx,
    domain: &Symbol,
    value: f32,
) -> Result<Value, TensorExecError> {
    cx.factory()
        .number_literal(domain.clone(), value.to_string())
        .map_err(TensorExecError::from)
}

pub(crate) fn round(dtype: WgpuKernelDType, value: f32) -> f32 {
    match dtype {
        WgpuKernelDType::F32 | WgpuKernelDType::Bf16WidenedToF32 => value,
        WgpuKernelDType::F16Native => half::f16::from_f32(value).to_f32(),
    }
}

pub(crate) fn invalid(message: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::InvalidRequest {
        message: message.into(),
    }
}

pub(crate) fn shape_error(message: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::Shape {
        message: message.into(),
    }
}

pub(crate) fn unsupported(operation: Symbol, reason: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::Unsupported {
        operation,
        reason: reason.into(),
    }
}

fn parse_literal(literal: NumberLiteral) -> Result<f32, TensorExecError> {
    if literal.domain == sim_lib_numbers_tensor::domains::i64() {
        literal
            .canonical
            .parse::<i64>()
            .map(|value| value as f32)
            .map_err(parse_error)
    } else if literal.domain == sim_lib_numbers_tensor::domains::rational() {
        parse_rational(&literal.canonical)
    } else {
        literal.canonical.parse::<f32>().map_err(parse_error)
    }
}

fn parse_rational(text: &str) -> Result<f32, TensorExecError> {
    let Some((numerator, denominator)) = text.split_once('/') else {
        return text.parse::<f32>().map_err(parse_error);
    };
    let numerator = numerator.trim().parse::<f32>().map_err(parse_error)?;
    let denominator = denominator.trim().parse::<f32>().map_err(parse_error)?;
    Ok(numerator / denominator)
}

fn parse_error(error: impl std::fmt::Display) -> TensorExecError {
    TensorExecError::Eval {
        message: Arc::from(format!("wgpu numeric parse failed: {error}")),
    }
}
