//! Resident tensor storage for the modeled compute provider.

use std::any::Any;
use std::sync::{Arc, OnceLock};

use sim_kernel::{DefaultFactory, Error, Factory, Result, Symbol, Value};
use sim_lib_numbers_tensor::{Tensor, TensorLocation, TensorStorage};

use crate::model::{ModeledComputeFault, ModeledTensorExecutor};

/// Opaque resident allocation handle owned by the modeled site.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentHandle {
    symbol: Symbol,
}

impl ResidentHandle {
    pub(crate) fn new(id: usize) -> Self {
        Self {
            symbol: Symbol::qualified("compute.alloc", id.to_string()),
        }
    }

    /// Returns the opaque allocation symbol.
    pub fn symbol(&self) -> &Symbol {
        &self.symbol
    }
}

/// Resident storage that models readback into host tensor cells.
pub struct ModeledResidentStorage {
    site: Symbol,
    allocation: ResidentHandle,
    shape: Arc<[usize]>,
    dtype: Symbol,
    cells: Arc<[Value]>,
    executor: ModeledTensorExecutor,
    fault: Option<ModeledComputeFault>,
    materialized: OnceLock<Result<Arc<dyn TensorStorage>>>,
}

impl ModeledResidentStorage {
    pub(crate) fn new(
        site: Symbol,
        allocation: ResidentHandle,
        shape: Vec<usize>,
        dtype: Symbol,
        cells: Arc<[Value]>,
        executor: ModeledTensorExecutor,
        fault: Option<ModeledComputeFault>,
    ) -> Self {
        Self {
            site,
            allocation,
            shape: shape.into(),
            dtype,
            cells,
            executor,
            fault,
            materialized: OnceLock::new(),
        }
    }

    /// Returns the resident allocation handle.
    pub fn allocation(&self) -> &ResidentHandle {
        &self.allocation
    }

    /// Rebuilds a canonical host tensor without counting a user readback.
    pub fn resident_tensor(&self) -> Option<Tensor> {
        Tensor::from_storage(
            self.shape.to_vec(),
            self.dtype.clone(),
            Arc::new(BoxedTensorStorageForResident::new(
                self.dtype.clone(),
                self.cells.clone(),
            )),
        )
        .ok()
    }
}

impl TensorStorage for ModeledResidentStorage {
    fn dtype(&self) -> &Symbol {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn location(&self) -> TensorLocation {
        TensorLocation::Resident {
            site: self.site.clone(),
            allocation: self.allocation.symbol().clone(),
        }
    }

    fn cell(&self, index: usize) -> Result<Value> {
        let storage = self.materialize()?;
        storage.cell(index)
    }

    fn materialize(&self) -> Result<Arc<dyn TensorStorage>> {
        self.materialized
            .get_or_init(|| {
                self.executor.increment_readbacks();
                if self.fault == Some(ModeledComputeFault::ReadbackFailure) {
                    return Err(Error::Eval("modeled compute readback failed".to_owned()));
                }
                Ok(Arc::new(BoxedTensorStorageForResident::new(
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

struct BoxedTensorStorageForResident {
    dtype: Symbol,
    cells: Arc<[Value]>,
}

impl BoxedTensorStorageForResident {
    fn new(dtype: Symbol, cells: Arc<[Value]>) -> Self {
        Self { dtype, cells }
    }
}

impl TensorStorage for BoxedTensorStorageForResident {
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
            .ok_or_else(|| Error::Eval("tensor cell index was out of bounds".to_owned()))
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

impl sim_kernel::Object for ResidentHandle {
    fn display(&self, _cx: &mut sim_kernel::Cx) -> Result<String> {
        Ok(format!("#<compute-resident {}>", self.symbol))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl sim_kernel::ObjectCompat for ResidentHandle {
    fn class(&self, _cx: &mut sim_kernel::Cx) -> Result<sim_kernel::ClassRef> {
        DefaultFactory.class_stub(
            sim_kernel::CORE_FUNCTION_CLASS_ID,
            Symbol::qualified("compute", "ResidentHandle"),
        )
    }

    fn as_table(&self, cx: &mut sim_kernel::Cx) -> Result<Value> {
        cx.factory().table(vec![(
            Symbol::new("allocation"),
            cx.factory().symbol(self.symbol.clone())?,
        )])
    }
}
