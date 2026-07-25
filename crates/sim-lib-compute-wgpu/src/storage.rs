//! Resident tensor storage produced by portable wgpu kernels.

use std::{
    any::Any,
    sync::{Arc, OnceLock},
};

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};
use sim_lib_numbers_tensor::{Tensor, TensorLocation, TensorStorage};

use crate::{WgpuArenaAllocation, WgpuMaterializationCache, WgpuResidentSegment};

/// Resident storage that records a validated wgpu pipeline result.
pub struct WgpuResidentStorage {
    site: Symbol,
    allocation: WgpuArenaAllocation,
    pipeline: Symbol,
    segments: Arc<[WgpuResidentSegment]>,
    shape: Arc<[usize]>,
    dtype: Symbol,
    cells: Arc<[Value]>,
    cache: WgpuMaterializationCache,
    materialized: OnceLock<Result<Arc<dyn TensorStorage>>>,
}

impl WgpuResidentStorage {
    /// Builds resident storage around host-equivalent cells.
    pub fn new(
        site: Symbol,
        allocation: WgpuArenaAllocation,
        pipeline: Symbol,
        segments: Vec<WgpuResidentSegment>,
        shape: Vec<usize>,
        dtype: Symbol,
        cells: Arc<[Value]>,
    ) -> Self {
        Self {
            site,
            allocation,
            pipeline,
            segments: segments.into(),
            shape: shape.into(),
            dtype,
            cells,
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

    /// Rebuilds a canonical tensor without counting a user materialization.
    pub fn resident_tensor(&self) -> Option<Tensor> {
        Tensor::from_storage(
            self.shape.to_vec(),
            self.dtype.clone(),
            Arc::new(BoxedWgpuStorage::new(
                self.dtype.clone(),
                self.cells.clone(),
            )),
        )
        .ok()
    }
}

impl TensorStorage for WgpuResidentStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.cells.len()
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
                self.cache
                    .get_or_try_init(|| Ok(Arc::<[u8]>::from([])))
                    .map_err(Error::Eval)?;
                Ok(Arc::new(BoxedWgpuStorage::new(
                    self.dtype.clone(),
                    self.cells.clone(),
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
