//! Portable kernel planning and host-equivalent execution.

use sim_kernel::{Cx, Symbol};
use sim_lib_numbers_tensor::{
    Tensor, TensorExecError, TensorRequest, add_op_symbol, bounded_element_count, cos_op_symbol,
    div_op_symbol, dot_op_symbol, exp_op_symbol, matmul_exec_op_symbol, max_op_symbol,
    min_op_symbol, mul_op_symbol, neg_op_symbol, norm_op_symbol, sin_op_symbol, sqrt_op_symbol,
    sub_op_symbol, sum_op_symbol, transpose_exec_op_symbol,
};

use crate::{
    kernel_elementwise::{execute_binary, execute_unary},
    kernel_linalg::{execute_dot, execute_matmul, execute_transpose},
    kernel_reductions::execute_reduction,
    kernel_support::unsupported,
    pipeline::{WgpuKernelDType, WgpuKernelOp},
};

pub use crate::kernel_wgsl::{PORTABLE_LINALG_WGSL, PORTABLE_REDUCTION_WGSL};

/// Returns the WGSL source for a real-device dispatch operation.
pub(crate) fn kernel_wgsl_for_op(op: WgpuKernelOp) -> &'static str {
    if op.is_reduction() {
        PORTABLE_REDUCTION_WGSL
    } else if op.is_linalg() {
        PORTABLE_LINALG_WGSL
    } else {
        crate::kernel_wgsl::POINTWISE_DISPATCH_WGSL
    }
}

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
    } else if *symbol == neg_op_symbol() {
        Some(WgpuKernelOp::Neg)
    } else if *symbol == sqrt_op_symbol() {
        Some(WgpuKernelOp::Sqrt)
    } else if *symbol == exp_op_symbol() {
        Some(WgpuKernelOp::Exp)
    } else if *symbol == Symbol::qualified("tensor", "op/log") {
        Some(WgpuKernelOp::Log)
    } else if *symbol == sin_op_symbol() {
        Some(WgpuKernelOp::Sin)
    } else if *symbol == cos_op_symbol() {
        Some(WgpuKernelOp::Cos)
    } else if *symbol == sum_op_symbol() {
        Some(WgpuKernelOp::Sum)
    } else if *symbol == min_op_symbol() {
        Some(WgpuKernelOp::Min)
    } else if *symbol == max_op_symbol() {
        Some(WgpuKernelOp::Max)
    } else if *symbol == norm_op_symbol() {
        Some(WgpuKernelOp::Norm)
    } else if *symbol == transpose_exec_op_symbol() {
        Some(WgpuKernelOp::Transpose)
    } else if *symbol == dot_op_symbol() {
        Some(WgpuKernelOp::Dot)
    } else if *symbol == matmul_exec_op_symbol() {
        Some(WgpuKernelOp::Matmul)
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
    let cells = if op.is_reduction() {
        execute_reduction(cx, op, request, dtype)?
    } else if op == WgpuKernelOp::Transpose {
        execute_transpose(request)?
    } else if op == WgpuKernelOp::Dot {
        execute_dot(cx, request, dtype)?
    } else if op == WgpuKernelOp::Matmul {
        execute_matmul(cx, request, dtype)?
    } else if op.is_binary() {
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
    .ok_or_else(|| crate::kernel_support::invalid("wgpu kernel produced a non-tensor value"))
}
