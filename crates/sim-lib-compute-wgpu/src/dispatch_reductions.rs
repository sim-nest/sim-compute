//! Real wgpu reduction dispatch for retained device contexts.

use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{Tensor, TensorExecError, TensorRequest};
use wgpu::util::DeviceExt;

use crate::{
    WgpuKernelDType, WgpuTensorExecutor,
    dispatch::{
        buffer_size, check_storage_buffer_limit, compiled_pipeline, f32_bytes, pipeline_symbol,
        read_f32s, readback_buffer, scalar_tensor, tensor_f32_values, u32_count,
    },
    kernel_reductions::fixed_tree_sum_values,
    kernel_support::{invalid, round, shape_error},
};

pub(crate) fn execute_reduction_dispatch(
    executor: &WgpuTensorExecutor,
    cx: &mut sim_kernel::Cx,
    request: &TensorRequest,
    op: crate::WgpuKernelOp,
    dtype: WgpuKernelDType,
) -> std::result::Result<(Tensor, Symbol), TensorExecError> {
    let Some(context) = &executor.context else {
        return Err(invalid("wgpu device context is unavailable"));
    };
    let [tensor] = request.inputs.as_ref() else {
        return Err(invalid(
            "wgpu reduction dispatch expects exactly one tensor input",
        ));
    };
    if !request.output.shape().is_empty() {
        return Err(shape_error("wgpu reduction output must be scalar"));
    }
    let len = tensor.len();
    if len == 0 {
        if matches!(op, crate::WgpuKernelOp::Min | crate::WgpuKernelOp::Max) {
            return Err(invalid("wgpu min/max reductions require at least one cell"));
        }
        let tensor = scalar_tensor(cx, request, 0.0)?;
        let symbol = pipeline_symbol(executor, context, op, dtype, 0);
        return Ok((tensor, symbol));
    }
    let input_values = tensor_f32_values(cx, tensor, dtype)?;
    let partial_count = len.div_ceil(512);
    check_storage_buffer_limit(executor, input_values.len(), "wgpu reduction input")?;
    check_storage_buffer_limit(executor, partial_count, "wgpu reduction partial output")?;
    let output_size = buffer_size(partial_count)?;
    let input = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-reduction-input"),
            contents: &f32_bytes(&input_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let partials = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-compute-wgpu-reduction-partials"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = readback_buffer(&context.device, output_size, "reduction");
    let params = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-reduction-params"),
            contents: &reduction_params_bytes(op, len)?,
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let pipeline = compiled_pipeline(executor, context, op, dtype, 0);
    let layout = pipeline.pipeline.get_bind_group_layout(0);
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-compute-wgpu-reduction-bind-group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: partials.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sim-compute-wgpu-reduction-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim-compute-wgpu-reduction-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(
            u32_count(partial_count, "wgpu reduction partial count")?,
            1,
            1,
        );
    }
    encoder.copy_buffer_to_buffer(&partials, 0, &readback, 0, output_size);
    context.queue.submit([encoder.finish()]);
    let partials = read_f32s(context, &readback, partial_count)?;
    let value = match op {
        crate::WgpuKernelOp::Sum => fixed_tree_sum_values(partials, dtype),
        crate::WgpuKernelOp::Norm => fixed_tree_sum_values(partials, dtype).sqrt(),
        crate::WgpuKernelOp::Min => fixed_tree_min_max(partials, false)?,
        crate::WgpuKernelOp::Max => fixed_tree_min_max(partials, true)?,
        _ => {
            return Err(invalid(
                "wgpu reduction dispatch received a non-reduction op",
            ));
        }
    };
    Ok((
        scalar_tensor(cx, request, round(dtype, value))?,
        pipeline.record.symbol,
    ))
}

fn fixed_tree_min_max(
    mut values: Vec<f32>,
    max: bool,
) -> std::result::Result<f32, TensorExecError> {
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

fn reduction_params_bytes(
    op: crate::WgpuKernelOp,
    len: usize,
) -> std::result::Result<[u8; 8], TensorExecError> {
    let op = match op {
        crate::WgpuKernelOp::Sum => 0_u32,
        crate::WgpuKernelOp::Min => 1,
        crate::WgpuKernelOp::Max => 2,
        crate::WgpuKernelOp::Norm => 3,
        _ => return Err(invalid("wgpu reduction params received a non-reduction op")),
    };
    let len = u32_count(len, "wgpu reduction length")?;
    let mut bytes = [0_u8; 8];
    bytes[..4].copy_from_slice(&op.to_ne_bytes());
    bytes[4..].copy_from_slice(&len.to_ne_bytes());
    Ok(bytes)
}
