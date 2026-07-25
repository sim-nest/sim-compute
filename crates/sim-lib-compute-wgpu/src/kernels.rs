//! Portable element-wise kernel planning and host-equivalent execution.

use std::sync::Arc;

use sim_kernel::{Cx, NumberLiteral, Symbol, Value};
use sim_lib_numbers_tensor::{
    Tensor, TensorExecError, TensorRequest, add_op_symbol, bounded_element_count, cos_op_symbol,
    div_op_symbol, exp_op_symbol, mul_op_symbol, number_literal_for_tensor_cell, sin_op_symbol,
    sqrt_op_symbol, sub_op_symbol,
};

use crate::pipeline::{WgpuKernelDType, WgpuKernelOp};

/// Portable WGSL source used by validated element-wise pipelines.
pub const PORTABLE_ELEMENTWISE_WGSL: &str = r#"
struct KernelParams {
    op: u32,
    len: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: KernelParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) {
        return;
    }
    let x = left[i];
    let y = right[i];
    if (params.op == 0u) {
        out[i] = x + y;
    } else if (params.op == 1u) {
        out[i] = x - y;
    } else if (params.op == 2u) {
        out[i] = x * y;
    } else if (params.op == 3u) {
        out[i] = x / y;
    } else if (params.op == 4u) {
        out[i] = sqrt(x);
    } else if (params.op == 5u) {
        out[i] = exp(x);
    } else if (params.op == 6u) {
        out[i] = sin(x);
    } else {
        out[i] = cos(x);
    }
}
"#;

/// Returns the portable kernel operation for a tensor request.
pub fn kernel_op(symbol: &Symbol) -> Option<WgpuKernelOp> {
    if *symbol == add_op_symbol() {
        Some(WgpuKernelOp::Add)
    } else if *symbol == sub_op_symbol() {
        Some(WgpuKernelOp::Sub)
    } else if *symbol == mul_op_symbol() {
        Some(WgpuKernelOp::Mul)
    } else if *symbol == div_op_symbol() {
        Some(WgpuKernelOp::Div)
    } else if *symbol == sqrt_op_symbol() {
        Some(WgpuKernelOp::Sqrt)
    } else if *symbol == exp_op_symbol() {
        Some(WgpuKernelOp::Exp)
    } else if *symbol == sin_op_symbol() {
        Some(WgpuKernelOp::Sin)
    } else if *symbol == cos_op_symbol() {
        Some(WgpuKernelOp::Cos)
    } else {
        None
    }
}

/// Validates and evaluates one portable f32 kernel on host-equivalent values.
pub fn execute_portable_kernel(
    cx: &mut Cx,
    request: &TensorRequest,
    dtype: WgpuKernelDType,
) -> Result<Tensor, TensorExecError> {
    let op = kernel_op(&request.operation.symbol).ok_or_else(|| {
        unsupported(
            request.operation.symbol.clone(),
            "wgpu portable kernel does not support this operation",
        )
    })?;
    let shape = request.output.shape().to_vec();
    bounded_element_count(&shape).map_err(TensorExecError::from)?;
    let cells = if op.is_binary() {
        execute_binary(cx, op, request, dtype, &shape)?
    } else {
        execute_unary(cx, op, request, dtype, &shape)?
    };
    sim_lib_numbers_tensor::build_tensor_value(
        cx,
        shape,
        Some(request.output.dtype().clone()),
        cells,
    )
    .map_err(TensorExecError::from)?
    .object()
    .downcast_ref::<Tensor>()
    .cloned()
    .ok_or_else(|| invalid("wgpu kernel produced a non-tensor value"))
}

fn execute_binary(
    cx: &mut Cx,
    op: WgpuKernelOp,
    request: &TensorRequest,
    dtype: WgpuKernelDType,
    shape: &[usize],
) -> Result<Vec<Value>, TensorExecError> {
    let [left, right] = request.inputs.as_ref() else {
        return Err(invalid(
            "wgpu binary kernel expects exactly two tensor inputs",
        ));
    };
    let actual_shape = broadcast_shape(left.shape(), right.shape())?;
    if actual_shape != shape {
        return Err(shape_error(format!(
            "wgpu broadcast output shape {:?} did not match {:?}",
            actual_shape, shape
        )));
    }
    let mut out = Vec::with_capacity(bounded_element_count(shape).map_err(TensorExecError::from)?);
    for coord in Tensor::coordinates(shape) {
        let left = numeric_cell(cx, &select_cell(left, &coord, shape)?)?;
        let right = numeric_cell(cx, &select_cell(right, &coord, shape)?)?;
        let value = match op {
            WgpuKernelOp::Add => left + right,
            WgpuKernelOp::Sub => left - right,
            WgpuKernelOp::Mul => left * right,
            WgpuKernelOp::Div => left / right,
            _ => unreachable!("binary kernel op checked by caller"),
        };
        out.push(number_value(
            cx,
            request.output.dtype(),
            dtype.round(value),
        )?);
    }
    Ok(out)
}

fn execute_unary(
    cx: &mut Cx,
    op: WgpuKernelOp,
    request: &TensorRequest,
    dtype: WgpuKernelDType,
    shape: &[usize],
) -> Result<Vec<Value>, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(invalid(
            "wgpu unary kernel expects exactly one tensor input",
        ));
    };
    if tensor.shape() != shape {
        return Err(shape_error(format!(
            "wgpu unary output shape {:?} did not match {:?}",
            tensor.shape(),
            shape
        )));
    }
    let source = tensor.cells().map_err(TensorExecError::from)?;
    source
        .iter()
        .map(|cell| {
            let input = numeric_cell(cx, cell)?;
            let value = match op {
                WgpuKernelOp::Sqrt => input.sqrt(),
                WgpuKernelOp::Exp => input.exp(),
                WgpuKernelOp::Sin => input.sin(),
                WgpuKernelOp::Cos => input.cos(),
                _ => unreachable!("unary kernel op checked by caller"),
            };
            number_value(cx, request.output.dtype(), dtype.round(value))
        })
        .collect()
}

fn broadcast_shape(left: &[usize], right: &[usize]) -> Result<Vec<usize>, TensorExecError> {
    let rank = left.len().max(right.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let left_dim = *left.get(left.len().wrapping_sub(rank - axis)).unwrap_or(&1);
        let right_dim = *right
            .get(right.len().wrapping_sub(rank - axis))
            .unwrap_or(&1);
        if left_dim == right_dim {
            out.push(left_dim);
        } else if left_dim == 1 {
            out.push(right_dim);
        } else if right_dim == 1 {
            out.push(left_dim);
        } else {
            return Err(invalid(format!(
                "cannot broadcast tensor shapes {left:?} and {right:?}"
            )));
        }
    }
    Ok(out)
}

fn select_cell(
    tensor: &Tensor,
    coord: &[usize],
    result_shape: &[usize],
) -> Result<Value, TensorExecError> {
    let rank_gap = result_shape.len().saturating_sub(tensor.shape().len());
    let mut local = Vec::with_capacity(tensor.shape().len());
    for (axis, dim) in tensor.shape().iter().enumerate() {
        let result_axis = axis + rank_gap;
        let coord_value = coord
            .get(result_axis)
            .copied()
            .ok_or_else(|| invalid("wgpu broadcast axis mismatch"))?;
        local.push(if *dim == 1 { 0 } else { coord_value });
    }
    Tensor::flat_offset(tensor.shape(), &local)
        .and_then(|flat| tensor.cell(flat))
        .map_err(TensorExecError::from)
}

fn numeric_cell(cx: &mut Cx, value: &Value) -> Result<f32, TensorExecError> {
    let literal = number_literal_for_tensor_cell(value)
        .or_else(|| cx.number_value_ref(value.clone()).ok().flatten()?.literal)
        .ok_or_else(|| invalid("wgpu kernels expect numeric tensor cells"))?;
    parse_literal(literal)
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

fn number_value(cx: &mut Cx, domain: &Symbol, value: f32) -> Result<Value, TensorExecError> {
    cx.factory()
        .number_literal(domain.clone(), value.to_string())
        .map_err(TensorExecError::from)
}

fn parse_error(error: impl std::fmt::Display) -> TensorExecError {
    TensorExecError::Eval {
        message: Arc::from(format!("wgpu numeric parse failed: {error}")),
    }
}

fn invalid(message: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::InvalidRequest {
        message: message.into(),
    }
}

fn shape_error(message: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::Shape {
        message: message.into(),
    }
}

fn unsupported(operation: Symbol, reason: impl Into<Arc<str>>) -> TensorExecError {
    TensorExecError::Unsupported {
        operation,
        reason: reason.into(),
    }
}

impl WgpuKernelDType {
    fn round(self, value: f32) -> f32 {
        match self {
            Self::F32 | Self::Bf16WidenedToF32 => value,
            Self::F16Native => half::f16::from_f32(value).to_f32(),
        }
    }
}
