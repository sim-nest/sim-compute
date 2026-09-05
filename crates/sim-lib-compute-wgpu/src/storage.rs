//! Resident tensor storage produced by portable wgpu kernels.

use std::{
    any::Any,
    sync::{Arc, OnceLock},
};

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};
use sim_lib_numbers_tensor::{TensorLocation, TensorStorage};

use crate::{
    WgpuArenaAllocation, WgpuMaterializationCache, WgpuPhysicalCounters, WgpuResidentSegment,
    dispatch::{read_f32s, readback_buffer},
    site::WgpuExecutionContext,
};

/// Resident storage that records a validated wgpu pipeline result.
pub struct WgpuResidentStorage {
    site: Symbol,
    allocation: WgpuArenaAllocation,
    pipeline: Symbol,
    segments: Arc<[WgpuResidentSegment]>,
    dtype: Symbol,
    len: usize,
    buffer: Arc<wgpu::Buffer>,
    context: WgpuExecutionContext,
    counters: WgpuPhysicalCounters,
    cache: WgpuMaterializationCache,
    materialized: OnceLock<Result<Arc<dyn TensorStorage>>>,
}

pub(crate) struct WgpuResidentStorageDescriptor {
    pub(crate) site: Symbol,
    pub(crate) allocation: WgpuArenaAllocation,
    pub(crate) pipeline: Symbol,
    pub(crate) segments: Vec<WgpuResidentSegment>,
    pub(crate) dtype: Symbol,
    pub(crate) len: usize,
    pub(crate) buffer: Arc<wgpu::Buffer>,
    pub(crate) context: WgpuExecutionContext,
    pub(crate) counters: WgpuPhysicalCounters,
}

impl WgpuResidentStorage {
    /// Builds resident storage around a real device buffer.
    pub(crate) fn new(descriptor: WgpuResidentStorageDescriptor) -> Self {
        Self {
            site: descriptor.site,
            allocation: descriptor.allocation,
            pipeline: descriptor.pipeline,
            segments: descriptor.segments.into(),
            dtype: descriptor.dtype,
            len: descriptor.len,
            buffer: descriptor.buffer,
            context: descriptor.context,
            counters: descriptor.counters,
            cache: WgpuMaterializationCache::default(),
            materialized: OnceLock::new(),
        }
    }

    /// Returns the resident allocation.
    pub fn allocation(&self) -> &WgpuArenaAllocation {
        &self.allocation
    }

    /// Returns the validated pipeline symbol.
    pub fn pipeline(&self) -> &Symbol {
        &self.pipeline
    }

    /// Returns the segmented resident layout.
    pub fn segments(&self) -> &[WgpuResidentSegment] {
        &self.segments
    }

    /// Returns the bindable resident device buffer.
    pub(crate) fn buffer(&self) -> Arc<wgpu::Buffer> {
        self.buffer.clone()
    }

    /// Returns the retained device context that owns the buffer.
    pub(crate) fn context(&self) -> &WgpuExecutionContext {
        &self.context
    }
}

impl TensorStorage for WgpuResidentStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.len
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Resident {
            site: self.site.clone(),
            allocation: self.allocation.id_symbol(),
        }
    }

    fn cell(&self, index: usize) -> Result<Value> {
        self.materialize()?.cell(index)
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        self.materialized
            .get_or_init(|| {
                self.counters.record_full_readback();
                let size = self.allocation.bytes.max(4);
                let readback = readback_buffer(&self.context.device, size, "materialize");
                let mut encoder =
                    self.context
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("sim-compute-wgpu-materialize-encoder"),
                        });
                encoder.copy_buffer_to_buffer(&self.buffer, 0, &readback, 0, size);
                self.context.queue.submit([encoder.finish()]);
                self.counters.record_submit(&self.segments);
                let values = self
                    .cache
                    .get_or_try_init(|| {
                        let values = read_f32s(&self.context, &readback, self.len)
                            .map_err(|err| err.to_string())?;
                        Ok(values
                            .iter()
                            .flat_map(|value| value.to_ne_bytes())
                            .collect::<Vec<_>>()
                            .into())
                    })
                    .map_err(Error::Eval)?;
                let cells = values
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .take(self.len)
                    .map(|bytes| {
                        let value = f32::from_ne_bytes(*bytes);
                        DefaultFactory.number_literal(self.dtype.clone(), value.to_string())
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Arc::new(BoxedWgpuStorage::new(
                    self.dtype.clone(),
                    cells.into(),
                )))
            })
            .clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct BoxedWgpuStorage {
    dtype: Symbol,
    cells: Arc<[Value]>,
}

impl BoxedWgpuStorage {
    fn new(dtype: Symbol, cells: Arc<[Value]>) -> Self {
        Self { dtype, cells }
    }
}

impl TensorStorage for BoxedWgpuStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Host
    }

    fn cell(&self, index: usize) -> Result<Value> {
        self.cells
            .get(index)
            .cloned()
            .ok_or_else(|| Error::Eval("wgpu tensor cell index was out of bounds".to_owned()))
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        Ok(Arc::new(Self {
            dtype: self.dtype.clone(),
            cells: self.cells.clone(),
        }))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl WgpuArenaAllocation {
    fn id_symbol(&self) -> Symbol {
        Symbol::qualified("compute.alloc.wgpu", format!("{:?}", self.id))
    }
}

impl sim_kernel::Object for WgpuArenaAllocation {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok(format!("#<wgpu-allocation {:?}>", self.id))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for WgpuArenaAllocation {
    fn class(&self, _cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("compute", "WgpuAllocation"),
        )
    }
}
