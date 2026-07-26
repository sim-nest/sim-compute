//! Real wgpu linalg dispatch for retained device contexts.

use std::sync::Arc;

use sim_kernel::Symbol;
use sim_lib_numbers_tensor::{Tensor, TensorExecError, TensorRequest, bounded_element_count};
use wgpu::util::DeviceExt;

use crate::{
    WgpuKernelDType, WgpuTensorExecutor,
    dispatch::{
        buffer_size, check_storage_buffer_limit, compiled_pipeline, f32_bytes, read_f32s,
        readback_buffer, tensor_f32_values, u32_count,
    },
    kernel_support::{invalid, number_value, round, shape_error},
};

pub(crate) fn execute_linalg_dispatch(
    executor: &WgpuTensorExecutor,
    cx: &mut sim_kernel::Cx,
    request: &TensorRequest,
    op: crate::WgpuKernelOp,
    dtype: WgpuKernelDType,
) -> std::result::Result<(Tensor, Symbol), TensorExecError> {
    let Some(context) = &executor.context else {
        return Err(invalid("wgpu device context is unavailable"));
    };
    let plan = LinalgPlan::new(request, op)?;
    let left_values = tensor_f32_values(cx, plan.left, dtype)?;
    let right_values = plan
        .right
        .map(|right| tensor_f32_values(cx, right, dtype))
        .transpose()?
        .unwrap_or_else(|| vec![0.0]);
    let len = bounded_element_count(request.output.shape()).map_err(TensorExecError::from)?;
    check_storage_buffer_limit(executor, left_values.len(), "wgpu linalg left input")?;
    check_storage_buffer_limit(executor, right_values.len(), "wgpu linalg right input")?;
    check_storage_buffer_limit(executor, len, "wgpu linalg output")?;
    let output_size = buffer_size(len)?;
    let left = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-linalg-left"),
            contents: &f32_bytes(&left_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let right = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-linalg-right"),
            contents: &f32_bytes(&right_values),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let output = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("sim-compute-wgpu-linalg-output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = readback_buffer(&context.device, output_size, "linalg");
    let params = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim-compute-wgpu-linalg-params"),
            contents: &plan.params_bytes(),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let pipeline = compiled_pipeline(executor, context, op, dtype, request.output.shape().len());
    let layout = pipeline.pipeline.get_bind_group_layout(0);
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim-compute-wgpu-linalg-bind-group"),
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
            label: Some("sim-compute-wgpu-linalg-encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sim-compute-wgpu-linalg-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let (x, y) = plan.workgroups()?;
        pass.dispatch_workgroups(x, y, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_size);
    context.queue.submit([encoder.finish()]);
    let cells = read_f32s(context, &readback, len)?
        .into_iter()
        .map(|value| number_value(cx, request.output.dtype(), round(dtype, value)))
        .collect::<std::result::Result<Arc<[_]>, _>>()?;
    let tensor = sim_lib_numbers_tensor::build_tensor_value(
        cx,
        request.output.shape().to_vec(),
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

struct LinalgPlan<'a> {
    left: &'a Tensor,
    right: Option<&'a Tensor>,
    rows: usize,
    inner: usize,
    cols: usize,
    op_code: u32,
}

impl<'a> LinalgPlan<'a> {
    fn new(
        request: &'a TensorRequest,
        op: crate::WgpuKernelOp,
    ) -> std::result::Result<Self, TensorExecError> {
        match op {
            crate::WgpuKernelOp::Transpose => {
                let [left] = request.inputs.as_ref() else {
                    return Err(invalid("wgpu transpose expects exactly one tensor input"));
                };
                let [rows, cols] = left.shape() else {
                    return Err(invalid("wgpu transpose expects rank-2 input"));
                };
                if request.output.shape() != [*cols, *rows] {
                    return Err(shape_error("wgpu transpose output shape mismatch"));
                }
                validate_u32_dims(&[*rows, *cols], "wgpu transpose")?;
                Ok(Self {
                    left,
                    right: None,
                    rows: *rows,
                    inner: 0,
                    cols: *cols,
                    op_code: 0,
                })
            }
            crate::WgpuKernelOp::Dot => {
                let [left, right] = request.inputs.as_ref() else {
                    return Err(invalid("wgpu dot expects exactly two tensor inputs"));
                };
                if left.shape().len() != 1 || left.shape() != right.shape() {
                    return Err(invalid("wgpu dot expects matching rank-1 inputs"));
                }
                if !request.output.shape().is_empty() {
                    return Err(shape_error("wgpu dot output must be scalar"));
                }
                validate_u32_dims(&[left.shape()[0]], "wgpu dot")?;
                Ok(Self {
                    left,
                    right: Some(right),
                    rows: 1,
                    inner: left.shape()[0],
                    cols: 1,
                    op_code: 1,
                })
            }
            crate::WgpuKernelOp::Matmul => Self::matmul(request),
            _ => Err(invalid("wgpu linalg dispatch received a non-linalg op")),
        }
    }

    fn matmul(request: &'a TensorRequest) -> std::result::Result<Self, TensorExecError> {
        let [left, right] = request.inputs.as_ref() else {
            return Err(invalid("wgpu matmul expects exactly two tensor inputs"));
        };
        let (rows, inner, cols, expected_shape) = match (left.shape(), right.shape()) {
            ([n], [m]) if n == m => (1, *n, 1, Vec::new()),
            ([rows, inner_left], [inner_right, cols]) if inner_left == inner_right => {
                (*rows, *inner_left, *cols, vec![*rows, *cols])
            }
            ([rows, inner_left], [inner_right]) if inner_left == inner_right => {
                (*rows, *inner_left, 1, vec![*rows])
            }
            ([inner_left], [inner_right, cols]) if inner_left == inner_right => {
                (1, *inner_left, *cols, vec![*cols])
            }
            _ => {
                return Err(invalid(
                    "wgpu matmul supports rank-1 and rank-2 tensors with matching inner dimensions",
                ));
            }
        };
        if request.output.shape() != expected_shape {
            return Err(shape_error("wgpu matmul output shape mismatch"));
        }
        validate_u32_dims(&[rows, inner, cols], "wgpu matmul")?;
        Ok(Self {
            left,
            right: Some(right),
            rows,
            inner,
            cols,
            op_code: 2,
        })
    }

    fn params_bytes(&self) -> [u8; 16] {
        let mut bytes = [0_u8; 16];
        let rows = u32::try_from(self.rows).expect("linalg plan rows validated as u32");
        let inner = u32::try_from(self.inner).expect("linalg plan inner validated as u32");
        let cols = u32::try_from(self.cols).expect("linalg plan cols validated as u32");
        bytes[..4].copy_from_slice(&rows.to_ne_bytes());
        bytes[4..8].copy_from_slice(&inner.to_ne_bytes());
        bytes[8..12].copy_from_slice(&cols.to_ne_bytes());
        bytes[12..].copy_from_slice(&self.op_code.to_ne_bytes());
        bytes
    }

    fn workgroups(&self) -> std::result::Result<(u32, u32), TensorExecError> {
        if self.op_code == 1 {
            Ok((1, 1))
        } else {
            Ok((
                u32_count(
                    self.cols.div_ceil(16).max(1),
                    "wgpu linalg column workgroups",
                )?,
                u32_count(self.rows.div_ceil(16).max(1), "wgpu linalg row workgroups")?,
            ))
        }
    }
}

fn validate_u32_dims(dims: &[usize], label: &str) -> std::result::Result<(), TensorExecError> {
    for dim in dims {
        u32_count(*dim, label)?;
    }
    Ok(())
}
