//! Resident tensor storage produced by CUDA matmul.

use std::{
    any::Any,
    sync::{Arc, OnceLock},
};

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};
use sim_lib_numbers_tensor::{Tensor, TensorLocation, TensorStorage};

use crate::runtime::CudaDeviceBuffer;

/// Opaque CUDA allocation evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaAllocation {
    /// Monotonic allocation id within the executor.
    pub id: usize,
    /// Logical resident byte count.
    pub bytes: u64,
    /// CUDA operation that produced the allocation.
    pub operation: Symbol,
}

impl CudaAllocation {
    fn id_symbol(&self) -> Symbol {
        Symbol::qualified("compute.alloc.cuda", self.id.to_string())
    }
}

/// Resident storage for a CUDA matmul result.
pub struct CudaResidentStorage {
    site: Symbol,
    allocation: CudaAllocation,
    shape: Arc<[usize]>,
    dtype: Symbol,
    backing: CudaBacking,
    materialized: OnceLock<Result<Arc<dyn TensorStorage>>>,
}

enum CudaBacking {
    Host(Arc<[Value]>),
    Device(Arc<CudaDeviceBuffer>),
}

impl CudaResidentStorage {
    /// Builds resident CUDA storage around checked host-equivalent cells.
    pub fn new(
        site: Symbol,
        allocation: CudaAllocation,
        shape: Vec<usize>,
        dtype: Symbol,
        cells: Arc<[Value]>,
    ) -> Self {
        Self {
            site,
            allocation,
            shape: shape.into(),
            dtype,
            backing: CudaBacking::Host(cells),
            materialized: OnceLock::new(),
        }
    }

    pub(crate) fn from_device(
        site: Symbol,
        allocation: CudaAllocation,
        shape: Vec<usize>,
        dtype: Symbol,
        buffer: Arc<CudaDeviceBuffer>,
    ) -> Self {
        Self {
            site,
            allocation,
            shape: shape.into(),
            dtype,
            backing: CudaBacking::Device(buffer),
            materialized: OnceLock::new(),
        }
    }

    /// Returns the resident allocation evidence.
    pub fn allocation(&self) -> &CudaAllocation {
        &self.allocation
    }

    /// Rebuilds a canonical tensor without counting a user readback.
    pub fn resident_tensor(&self) -> Option<Tensor> {
        let storage = self.materialize().ok()?;
        Tensor::from_storage(self.shape.to_vec(), self.dtype.clone(), storage).ok()
    }

    pub(crate) fn device_buffer(&self) -> Option<&Arc<CudaDeviceBuffer>> {
        match &self.backing {
            CudaBacking::Host(_) => None,
            CudaBacking::Device(buffer) => Some(buffer),
        }
    }

    fn materialized_storage(&self) -> Result<Arc<dyn TensorStorage>> {
        let cells = match &self.backing {
            CudaBacking::Host(cells) => Arc::clone(cells),
            CudaBacking::Device(buffer) => buffer
                .read()
                .map_err(|error| Error::Eval(error.to_string()))?
                .into_iter()
                .map(|value| DefaultFactory.number_literal(self.dtype.clone(), value.to_string()))
                .collect::<Result<Vec<_>>>()?
                .into(),
        };
        Ok(Arc::new(BoxedCudaStorage::new(self.dtype.clone(), cells)))
    }
}

impl TensorStorage for CudaResidentStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        match &self.backing {
            CudaBacking::Host(cells) => cells.len(),
            CudaBacking::Device(buffer) => buffer.len(),
        }
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
            .get_or_init(|| self.materialized_storage())
            .clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct BoxedCudaStorage {
    dtype: Symbol,
    cells: Arc<[Value]>,
}

impl BoxedCudaStorage {
    fn new(dtype: Symbol, cells: Arc<[Value]>) -> Self {
        Self { dtype, cells }
    }
}

impl TensorStorage for BoxedCudaStorage {
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
            .ok_or_else(|| Error::Eval("cuda tensor cell index was out of bounds".to_owned()))
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

impl sim_kernel::Object for CudaAllocation {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok(format!("#<cuda-allocation {} {}b>", self.id, self.bytes))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for CudaAllocation {
    fn class(&self, _cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("compute", "CudaAllocation"),
        )
    }
}
