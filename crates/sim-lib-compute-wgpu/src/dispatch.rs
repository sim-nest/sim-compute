//! Real wgpu dispatch for retained device contexts.

use std::sync::Arc;

use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{Tensor, TensorExecError, TensorRequest, bounded_element_count};
use wgpu::util::DeviceExt;

use crate::{
    WgpuKernelDType, WgpuTensorExecutor,
    kernel_support::{invalid, number_value, numeric_cell, round, shape_error},
};

pub(crate) fn execute_pointwise_dispatch(
    executor: &WgpuTensorExecutor,
    cx: &mut sim_kernel::Cx,
    request: &TensorRequest,
    op: crate::WgpuKernelOp,
    dtype: WgpuKernelDType,
    bytes: u64,
) -> std::result::Result<(Tensor, Symbol), TensorExecError> {
    let Some(context) = &executor.context else {
        return Err(invalid("wgpu device context is unavailable"));
    };
    let shape = request.output.shape();
    let len = bounded_element_count(shape).map_err(TensorExecError::from)?;
    let left_values = expanded_left(cx, request, op, shape)?;
    let right_values = expanded_right(cx, request, op, shape)?;
    let left_bytes = f32_bytes(&left_values);
    let right_bytes = f32_bytes(&right_values);
    let output_size = bytes.max(4);
    let params = pointwise_params_bytes(op, len)?;
    let left = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-left"),
            contents: &left_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let right = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-right"),
            contents: &right_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-compute-wgpu-output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-compute-wgpu-readback"),
        size: output_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let params = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let pipeline = {
        let mut state = executor.state.lock().expect("wgpu executor state poisoned");
        state.pipelines.get_or_insert_compiled(
            &context.device,
            &executor.probe,
            op,
            dtype,
            shape.len(),
        )
    };
    let layout = pipeline.pipeline.get_bind_group_layout(0);
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-compute-wgpu-pointwise-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: left.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: right.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-compute-wgpu-pointwise-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim-compute-wgpu-pointwise-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups(len)?, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    context.queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| invalid(format!("wgpu device poll failed: {err}")))?;
    receiver
        .recv()
        .map_err(|err| invalid(format!("wgpu readback callback failed: {err}")))?
        .map_err(|err| invalid(format!("wgpu readback map failed: {err}")))?;
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .map_err(|err| invalid(format!("wgpu readback range failed: {err}")))?
        .to_vec();
    readback.unmap();
    let cells = mapped
        .chunks_exact(4)
        .take(len)
        .map(|bytes| {
            let value = f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            number_value(cx, request.output.dtype(), round(dtype, value))
        })
        .collect::<std::result::Result<Arc<[_]>, _>>()?;
    let tensor = sim_lib_numbers_tensor::build_tensor_value(
        cx,
        shape.to_vec(),
        Some(request.output.dtype().clone()),
        cells.to_vec(),
    )
    .map_err(TensorExecError::from)?
    .object()
    .downcast_ref::<Tensor>()
    .cloned()
    .ok_or_else(|| invalid("wgpu dispatch produced a non-tensor value"))?;
    Ok((tensor, pipeline.record.symbol))
}

pub(crate) fn is_pointwise_dispatch(op: crate::WgpuKernelOp) -> bool {
    matches!(
        op,
        crate::WgpuKernelOp::Add
            | crate::WgpuKernelOp::Sub
            | crate::WgpuKernelOp::Mul
            | crate::WgpuKernelOp::Div
            | crate::WgpuKernelOp::Neg
            | crate::WgpuKernelOp::Sqrt
            | crate::WgpuKernelOp::Exp
            | crate::WgpuKernelOp::Log
            | crate::WgpuKernelOp::Sin
            | crate::WgpuKernelOp::Cos
    )
}

fn expanded_left(
    cx: &mut sim_kernel::Cx,
    request: &TensorRequest,
    op: crate::WgpuKernelOp,
    shape: &[usize],
) -> std::result::Result<Vec<f32>, TensorExecError> {
    if op.is_binary() {
        let [left, right] = request.inputs.as_ref() else {
            return Err(invalid(
                "wgpu binary dispatch expects exactly two tensor inputs",
            ));
        };
        let actual_shape = broadcast_shape(left.shape(), right.shape())?;
        if actual_shape != shape {
            return Err(shape_error(format!(
                "wgpu broadcast output shape {:?} did not match {:?}",
                actual_shape, shape
            )));
        }
        expanded_tensor(cx, left, shape)
    } else {
        let [tensor] = request.inputs.as_ref() else {
            return Err(invalid(
                "wgpu unary dispatch expects exactly one tensor input",
            ));
        };
        if tensor.shape() != shape {
            return Err(shape_error(format!(
                "wgpu unary output shape {:?} did not match {:?}",
                tensor.shape(),
                shape
            )));
        }
        expanded_tensor(cx, tensor, shape)
    }
}

fn expanded_right(
    cx: &mut sim_kernel::Cx,
    request: &TensorRequest,
    op: crate::WgpuKernelOp,
    shape: &[usize],
) -> std::result::Result<Vec<f32>, TensorExecError> {
    if op.is_binary() {
        let [_, right] = request.inputs.as_ref() else {
            return Err(invalid(
                "wgpu binary dispatch expects exactly two tensor inputs",
            ));
        };
        expanded_tensor(cx, right, shape)
    } else {
        Ok(vec![
            0.0;
            bounded_element_count(shape)
                .map_err(TensorExecError::from)?
        ])
    }
}

fn expanded_tensor(
    cx: &mut sim_kernel::Cx,
    tensor: &Tensor,
    result_shape: &[usize],
) -> std::result::Result<Vec<f32>, TensorExecError> {
    let mut out =
        Vec::with_capacity(bounded_element_count(result_shape).map_err(TensorExecError::from)?);
    for coord in Tensor::coordinates(result_shape) {
        let cell = select_cell(tensor, &coord, result_shape)?;
        out.push(numeric_cell(cx, &cell)?);
    }
    Ok(out)
}

fn broadcast_shape(
    left: &[usize],
    right: &[usize],
) -> std::result::Result<Vec<usize>, TensorExecError> {
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
) -> std::result::Result<sim_kernel::Value, TensorExecError> {
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

pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

pub(crate) fn tensor_f32_values(
    cx: &mut sim_kernel::Cx,
    tensor: &Tensor,
    dtype: WgpuKernelDType,
) -> std::result::Result<Vec<f32>, TensorExecError> {
    tensor
        .cells()
        .map_err(TensorExecError::from)?
        .iter()
        .map(|cell| numeric_cell(cx, cell).map(|value| round(dtype, value)))
        .collect::<std::result::Result<Vec<_>, _>>()
}

pub(crate) fn scalar_tensor(
    cx: &mut sim_kernel::Cx,
    request: &TensorRequest,
    value: f32,
) -> std::result::Result<Tensor, TensorExecError> {
    let cell = number_value(cx, request.output.dtype(), value)?;
    sim_lib_numbers_tensor::build_tensor_value(
        cx,
        Vec::new(),
        Some(request.output.dtype().clone()),
        vec![cell],
    )
    .map_err(TensorExecError::from)?
    .object()
    .downcast_ref::<Tensor>()
    .cloned()
    .ok_or_else(|| invalid("wgpu dispatch produced a non-tensor value"))
}

pub(crate) fn compiled_pipeline(
    executor: &WgpuTensorExecutor,
    context: &crate::site::WgpuExecutionContext,
    op: crate::WgpuKernelOp,
    dtype: WgpuKernelDType,
    rank: usize,
) -> crate::pipeline::WgpuCompiledPipeline {
    let mut state = executor.state.lock().expect("wgpu executor state poisoned");
    state
        .pipelines
        .get_or_insert_compiled(&context.device, &executor.probe, op, dtype, rank)
}

pub(crate) fn pipeline_symbol(
    executor: &WgpuTensorExecutor,
    context: &crate::site::WgpuExecutionContext,
    op: crate::WgpuKernelOp,
    dtype: WgpuKernelDType,
    rank: usize,
) -> Symbol {
    compiled_pipeline(executor, context, op, dtype, rank)
        .record
        .symbol
}

pub(crate) fn readback_buffer(device: &wgpu::Device, size: u64, family: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("sim-compute-wgpu-{family}-readback")),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

pub(crate) fn read_f32s(
    context: &crate::site::WgpuExecutionContext,
    readback: &wgpu::Buffer,
    len: usize,
) -> std::result::Result<Vec<f32>, TensorExecError> {
    let (sender, receiver) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
    context
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|err| invalid(format!("wgpu device poll failed: {err}")))?;
    receiver
        .recv()
        .map_err(|err| invalid(format!("wgpu readback callback failed: {err}")))?
        .map_err(|err| invalid(format!("wgpu readback map failed: {err}")))?;
    let mapped = readback
        .slice(..)
        .get_mapped_range()
        .map_err(|err| invalid(format!("wgpu readback range failed: {err}")))?
        .to_vec();
    readback.unmap();
    Ok(mapped
        .chunks_exact(4)
        .take(len)
        .map(|bytes| f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect())
}

pub(crate) fn buffer_size(len: usize) -> std::result::Result<u64, TensorExecError> {
    u64::try_from(len)
        .ok()
        .and_then(|cells| cells.checked_mul(4))
        .map(|bytes| bytes.max(4))
        .ok_or_else(|| invalid("wgpu buffer byte count overflowed"))
}

pub(crate) fn check_storage_buffer_limit(
    executor: &WgpuTensorExecutor,
    cells: usize,
    label: &str,
) -> std::result::Result<(), TensorExecError> {
    let bytes = buffer_size(cells)?;
    let limit = executor
        .probe
        .adapter
        .granted_limits
        .max_storage_buffer_binding_size
        .max(4);
    if bytes > limit {
        return Err(invalid(format!(
            "{label} requires {bytes} bytes, exceeding storage buffer binding limit {limit}"
        )));
    }
    Ok(())
}

pub(crate) fn u32_count(value: usize, label: &str) -> std::result::Result<u32, TensorExecError> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} exceeds u32")))
}

fn pointwise_params_bytes(
    op: crate::WgpuKernelOp,
    len: usize,
) -> std::result::Result<[u8; 8], TensorExecError> {
    let op = match op {
        crate::WgpuKernelOp::Add => 0_u32,
        crate::WgpuKernelOp::Sub => 1,
        crate::WgpuKernelOp::Mul => 2,
        crate::WgpuKernelOp::Div => 3,
        crate::WgpuKernelOp::Neg => 4,
        crate::WgpuKernelOp::Sqrt => 5,
        crate::WgpuKernelOp::Exp => 6,
        crate::WgpuKernelOp::Log => 7,
        crate::WgpuKernelOp::Sin => 8,
        crate::WgpuKernelOp::Cos => 9,
        _ => return Err(invalid("wgpu pointwise params received a non-pointwise op")),
    };
    let len = u32::try_from(len).map_err(|_| invalid("wgpu tensor length exceeds u32"))?;
    let mut bytes = [0_u8; 8];
    bytes[..4].copy_from_slice(&op.to_ne_bytes());
    bytes[4..].copy_from_slice(&len.to_ne_bytes());
    Ok(bytes)
}

fn workgroups(len: usize) -> std::result::Result<u32, TensorExecError> {
    let len = u32::try_from(len).map_err(|_| invalid("wgpu tensor length exceeds u32"))?;
    Ok(len.div_ceil(64).max(1))
}
