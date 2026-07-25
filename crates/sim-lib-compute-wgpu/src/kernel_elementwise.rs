//! Host-equivalent element-wise portable kernel execution.

use sim_kernel::{Cx, Value};
use sim_lib_numbers_tensor::{Tensor, TensorExecError, TensorRequest, bounded_element_count};

use crate::{
    kernel_support::{invalid, number_value, numeric_cell, round, shape_error},
    pipeline::{WgpuKernelDType, WgpuKernelOp},
};

pub(crate) fn execute_binary(
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
            round(dtype, value),
        )?);
    }
    Ok(out)
}

pub(crate) fn execute_unary(
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
    tensor
        .cells()
        .map_err(TensorExecError::from)?
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
            number_value(cx, request.output.dtype(), round(dtype, value))
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
