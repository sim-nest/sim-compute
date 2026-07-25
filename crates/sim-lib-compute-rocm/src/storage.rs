//! Resident tensor storage produced by ROCm matmul.

use std::{
    any::Any,
    sync::{Arc, OnceLock},
};

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};
use sim_lib_numbers_tensor::{Tensor, TensorLocation, TensorStorage};

/// Opaque ROCm allocation evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RocmAllocation {
    /// Monotonic allocation id within the executor.
    pub id: usize,
    /// Logical resident byte count.
    pub bytes: u64,
    /// ROCm operation that produced the allocation.
    pub operation: Symbol,
}

impl RocmAllocation {
    fn id_symbol(&self) -> Symbol {
        Symbol::qualified("compute.alloc.rocm", self.id.to_string())
    }
}

/// Resident storage for a ROCm matmul result.
pub struct RocmResidentStorage {
    site: Symbol,
    allocation: RocmAllocation,
    shape: Arc<[usize]>,
    dtype: Symbol,
    cells: Arc<[Value]>,
    materialized: OnceLock<Result<Arc<dyn TensorStorage>>>,
}

impl RocmResidentStorage {
    /// Builds resident ROCm storage around checked host-equivalent cells.
    pub fn new(
        site: Symbol,
        allocation: RocmAllocation,
        shape: Vec<usize>,
        dtype: Symbol,
        cells: Arc<[Value]>,
    ) -> Self {
        Self {
            site,
            allocation,
            shape: shape.into(),
            dtype,
            cells,
            materialized: OnceLock::new(),
        }
    }

    /// Returns the resident allocation evidence.
    pub fn allocation(&self) -> &RocmAllocation {
        &self.allocation
    }

    /// Rebuilds a canonical tensor without counting a user readback.
    pub fn resident_tensor(&self) -> Option<Tensor> {
        Tensor::from_storage(
            self.shape.to_vec(),
            self.dtype.clone(),
            Arc::new(BoxedRocmStorage::new(
                self.dtype.clone(),
                self.cells.clone(),
            )),
        )
        .ok()
    }
}

impl TensorStorage for RocmResidentStorage {
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
                Ok(Arc::new(BoxedRocmStorage::new(
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

struct BoxedRocmStorage {
    dtype: Symbol,
    cells: Arc<[Value]>,
}

impl BoxedRocmStorage {
    fn new(dtype: Symbol, cells: Arc<[Value]>) -> Self {
        Self { dtype, cells }
    }
}

impl TensorStorage for BoxedRocmStorage {
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
            .ok_or_else(|| Error::Eval("rocm tensor cell index was out of bounds".to_owned()))
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

impl sim_kernel::Object for RocmAllocation {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok(format!("#<rocm-allocation {} {}b>", self.id, self.bytes))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for RocmAllocation {
    fn class(&self, _cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("compute", "RocmAllocation"),
        )
    }
}
