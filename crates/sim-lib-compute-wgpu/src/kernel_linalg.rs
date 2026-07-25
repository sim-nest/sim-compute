//! Host-equivalent transpose, dot, and matmul for portable wgpu kernels.

use sim_kernel::{Cx, Value};
use sim_lib_numbers_tensor::{Tensor, TensorExecError, TensorRequest};

use crate::{
    kernel_reductions::fixed_tree_sum_values,
    kernel_support::{invalid, number_value, numeric_cell, round, shape_error},
    pipeline::WgpuKernelDType,
};

struct ProductPlan<'a> {
    left: &'a Tensor,
    right: &'a Tensor,
    left_start: usize,
    right_start: usize,
    count: usize,
    left_stride: usize,
    right_stride: usize,
}

pub(crate) fn execute_transpose(request: &TensorRequest) -> Result<Vec<Value>, TensorExecError> {
    let [tensor] = request.inputs.as_ref() else {
        return Err(invalid("wgpu transpose expects exactly one tensor input"));
    };
    let [rows, cols] = tensor.shape() else {
        return Err(invalid("wgpu transpose expects rank-2 input"));
    };
    if request.output.shape() != [*cols, *rows] {
        return Err(shape_error("wgpu transpose output shape mismatch"));
    }
    let mut out = Vec::with_capacity(tensor.len());
    for col in 0..*cols {
        for row in 0..*rows {
            out.push(
                tensor
                    .cell(row * cols + col)
                    .map_err(TensorExecError::from)?,
            );
        }
    }
    Ok(out)
}

pub(crate) fn execute_dot(
    cx: &mut Cx,
    request: &TensorRequest,
    dtype: WgpuKernelDType,
) -> Result<Vec<Value>, TensorExecError> {
    let [left, right] = request.inputs.as_ref() else {
        return Err(invalid("wgpu dot expects exactly two tensor inputs"));
    };
    if left.shape().len() != 1 || left.shape() != right.shape() {
        return Err(invalid("wgpu dot expects matching rank-1 inputs"));
    }
    if !request.output.shape().is_empty() {
        return Err(shape_error("wgpu dot output must be scalar"));
    }
    let value = fixed_tree_products(
        cx,
        ProductPlan {
            left,
            right,
            left_start: 0,
            right_start: 0,
            count: left.shape()[0],
            left_stride: 1,
            right_stride: 1,
        },
        dtype,
    )?;
    Ok(vec![number_value(
        cx,
        request.output.dtype(),
        round(dtype, value),
    )?])
}

pub(crate) fn execute_matmul(
    cx: &mut Cx,
    request: &TensorRequest,
    dtype: WgpuKernelDType,
) -> Result<Vec<Value>, TensorExecError> {
    let [left, right] = request.inputs.as_ref() else {
        return Err(invalid("wgpu matmul expects exactly two tensor inputs"));
    };
    match (left.shape(), right.shape()) {
        ([n], [m]) if n == m => matmul_scalar(cx, request, left, right, *n, dtype),
        ([rows, inner_left], [inner_right, cols]) if inner_left == inner_right => {
            matmul_matrix(cx, request, left, right, (*rows, *inner_left, *cols), dtype)
        }
        ([rows, inner_left], [inner_right]) if inner_left == inner_right => {
            matmul_matrix_vector(cx, request, left, right, (*rows, *inner_left), dtype)
        }
        ([inner_left], [inner_right, cols]) if inner_left == inner_right => {
            matmul_vector_matrix(cx, request, left, right, (*inner_left, *cols), dtype)
        }
        _ => Err(invalid(
            "wgpu matmul supports rank-1 and rank-2 tensors with matching inner dimensions",
        )),
    }
}

fn matmul_scalar(
    cx: &mut Cx,
    request: &TensorRequest,
    left: &Tensor,
    right: &Tensor,
    count: usize,
    dtype: WgpuKernelDType,
) -> Result<Vec<Value>, TensorExecError> {
    if !request.output.shape().is_empty() {
        return Err(shape_error("wgpu vector dot output must be scalar"));
    }
    let value = fixed_tree_products(
        cx,
        ProductPlan {
            left,
            right,
            left_start: 0,
            right_start: 0,
            count,
            left_stride: 1,
            right_stride: 1,
        },
        dtype,
    )?;
    Ok(vec![number_value(
        cx,
        request.output.dtype(),
        round(dtype, value),
    )?])
}

fn matmul_matrix(
    cx: &mut Cx,
    request: &TensorRequest,
    left: &Tensor,
    right: &Tensor,
    dims: (usize, usize, usize),
    dtype: WgpuKernelDType,
) -> Result<Vec<Value>, TensorExecError> {
    let (rows, inner, cols) = dims;
    if request.output.shape() != [rows, cols] {
        return Err(shape_error("wgpu matmul output shape mismatch"));
    }
    let mut out = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let value = fixed_tree_products(
                cx,
                ProductPlan {
                    left,
                    right,
                    left_start: row * inner,
                    right_start: col,
                    count: inner,
                    left_stride: 1,
                    right_stride: cols,
                },
                dtype,
            )?;
            out.push(number_value(
                cx,
                request.output.dtype(),
                round(dtype, value),
            )?);
        }
    }
    Ok(out)
}

fn matmul_matrix_vector(
    cx: &mut Cx,
    request: &TensorRequest,
    left: &Tensor,
    right: &Tensor,
    dims: (usize, usize),
    dtype: WgpuKernelDType,
) -> Result<Vec<Value>, TensorExecError> {
    let (rows, inner) = dims;
    if request.output.shape() != [rows] {
        return Err(shape_error("wgpu matrix-vector output shape mismatch"));
    }
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let value = fixed_tree_products(
            cx,
            ProductPlan {
                left,
                right,
                left_start: row * inner,
                right_start: 0,
                count: inner,
                left_stride: 1,
                right_stride: 1,
            },
            dtype,
        )?;
        out.push(number_value(
            cx,
            request.output.dtype(),
            round(dtype, value),
        )?);
    }
    Ok(out)
}

fn matmul_vector_matrix(
    cx: &mut Cx,
    request: &TensorRequest,
    left: &Tensor,
    right: &Tensor,
    dims: (usize, usize),
    dtype: WgpuKernelDType,
) -> Result<Vec<Value>, TensorExecError> {
    let (inner, cols) = dims;
    if request.output.shape() != [cols] {
        return Err(shape_error("wgpu vector-matrix output shape mismatch"));
    }
    let mut out = Vec::with_capacity(cols);
    for col in 0..cols {
        let value = fixed_tree_products(
            cx,
            ProductPlan {
                left,
                right,
                left_start: 0,
                right_start: col,
                count: inner,
                left_stride: 1,
                right_stride: cols,
            },
            dtype,
        )?;
        out.push(number_value(
            cx,
            request.output.dtype(),
            round(dtype, value),
        )?);
    }
    Ok(out)
}

fn fixed_tree_products(
    cx: &mut Cx,
    plan: ProductPlan<'_>,
    dtype: WgpuKernelDType,
) -> Result<f32, TensorExecError> {
    let mut products = Vec::with_capacity(plan.count);
    for index in 0..plan.count {
        let left = numeric_cell(
            cx,
            &plan
                .left
                .cell(plan.left_start + index * plan.left_stride)
                .map_err(TensorExecError::from)?,
        )?;
        let right = numeric_cell(
            cx,
            &plan
                .right
                .cell(plan.right_start + index * plan.right_stride)
                .map_err(TensorExecError::from)?,
        )?;
        products.push(round(dtype, left * right));
    }
    Ok(fixed_tree_sum_values(products, dtype))
}
