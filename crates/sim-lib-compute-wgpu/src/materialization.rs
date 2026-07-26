//! Stable host materialization cache for resident wgpu tensors.

use std::sync::{Arc, OnceLock};

/// Caches exactly one synchronized host materialization result.
#[derive(Debug, Default)]
pub struct WgpuMaterializationCache {
    cells: OnceLock<Result<Arc<[u8]>, String>>,
}

impl WgpuMaterializationCache {
    /// Returns the cached materialization or records the first result.
    pub fn get_or_try_init(
        &self,
        materialize: impl FnOnce() -> Result<Arc<[u8]>, String>,
    ) -> Result<Arc<[u8]>, String> {
        self.cells.get_or_init(materialize).clone()
    }
}
