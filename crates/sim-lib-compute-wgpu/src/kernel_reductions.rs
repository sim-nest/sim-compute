//! Host-equivalent fixed-tree reductions for portable wgpu kernels.

use sim_kernel::{Cx, Value};
use sim_lib_numbers_tensor::{TensorExecError, TensorRequest};

use crate::{
    kernel_support::{invalid, number_value, numeric_cell, round, shape_error},
    pipeline::{WgpuKernelDType, WgpuKernelOp},
};

pub(crate) fn execute_reduction(
    cx: &mut Cx,
    op: WgpuKernelOp,
    request: &TensorRequest,
    dtype: WgpuKernelDType,
) -> Result<Vec<Value>, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(invalid("wgpu reduction expects exactly one tensor input"));
    };
    let cells = tensor.cells().map_err(TensorExecError::from)?;
    if !request.output.shape().is_empty() {
        return Err(shape_error("wgpu reduction output must be scalar"));
    }
    if cells.is_empty() && matches!(op, WgpuKernelOp::Min | WgpuKernelOp::Max) {
        return Err(invalid("wgpu min/max reductions require at least one cell"));
    }
    let value = match op {
        WgpuKernelOp::Sum => fixed_tree_sum(cx, cells.iter(), dtype)?,
        WgpuKernelOp::Min => fixed_tree_min_max(cx, cells.iter(), false)?,
        WgpuKernelOp::Max => fixed_tree_min_max(cx, cells.iter(), true)?,
        WgpuKernelOp::Norm => {
            let squares = cells
                .iter()
                .map(|cell| numeric_cell(cx, cell).map(|value| round(dtype, value * value)))
                .collect::<Result<Vec<_>, _>>()?;
            fixed_tree_sum_values(squares, dtype).sqrt()
        }
        _ => unreachable!("reduction op checked by caller"),
    };
    Ok(vec![number_value(
        cx,
        request.output.dtype(),
        round(dtype, value),
    )?])
}

pub(crate) fn fixed_tree_sum_values(mut values: Vec<f32>, dtype: WgpuKernelDType) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        for pair in values.chunks(2) {
            next.push(if let [left, right] = pair {
                round(dtype, *left + *right)
            } else {
                pair[0]
            });
        }
        values = next;
    }
    values[0]
}

fn fixed_tree_sum<'a>(
    cx: &mut Cx,
    cells: impl Iterator<Item = &'a Value>,
    dtype: WgpuKernelDType,
) -> Result<f32, TensorExecError> {
    let values = cells
        .map(|cell| numeric_cell(cx, cell).map(|value| round(dtype, value)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(fixed_tree_sum_values(values, dtype))
}

fn fixed_tree_min_max<'a>(
    cx: &mut Cx,
    cells: impl Iterator<Item = &'a Value>,
    max: bool,
) -> Result<f32, TensorExecError> {
    let mut values = cells
        .map(|cell| numeric_cell(cx, cell))
        .collect::<Result<Vec<_>, _>>()?;
    while values.len() > 1 {
        let mut next = Vec::with_capacity(values.len().div_ceil(2));
        for pair in values.chunks(2) {
            next.push(if let [left, right] = pair {
                if max {
                    left.max(*right)
                } else {
                    left.min(*right)
                }
            } else {
                pair[0]
            });
        }
        values = next;
    }
    values
        .into_iter()
        .next()
        .ok_or_else(|| invalid("wgpu min/max reductions require at least one cell"))
}
