//! Bounded validated pipeline cache for portable wgpu tensor kernels.

use std::collections::VecDeque;

use sim_kernel::Symbol;

use crate::{WgpuAdapterProbe, kernels::PORTABLE_ELEMENTWISE_WGSL};

/// Portable operation implemented by a wgpu tensor kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgpuKernelOp {
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
    /// Element-wise square root.
    Sqrt,
    /// Element-wise exponential.
    Exp,
    /// Element-wise sine.
    Sin,
    /// Element-wise cosine.
    Cos,
}

impl WgpuKernelOp {
    /// Returns true when the kernel consumes two input tensors.
    pub fn is_binary(self) -> bool {
        matches!(self, Self::Add | Self::Sub | Self::Mul | Self::Div)
    }

    fn symbol_name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Sin => "sin",
            Self::Cos => "cos",
        }
    }
}

/// Dtype strategy selected for a portable kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WgpuKernelDType {
    /// Native f32 arithmetic.
    F32,
    /// Native f16 arithmetic, only when the adapter granted shader f16.
    F16Native,
    /// bf16/unsupported half inputs widened to f32 arithmetic.
    Bf16WidenedToF32,
}

impl WgpuKernelDType {
    fn symbol_name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16Native => "f16",
            Self::Bf16WidenedToF32 => "f32-widened-half",
        }
    }
}

/// Cache key for a validated portable pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuPipelineKey {
    /// Adapter ordinal from discovery evidence.
    pub adapter_ordinal: usize,
    /// Portable kernel operation.
    pub op: WgpuKernelOp,
    /// Kernel dtype strategy.
    pub dtype: WgpuKernelDType,
    /// Output rank.
    pub rank: usize,
}

/// Validated pipeline evidence retained in the cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuPipelineRecord {
    /// Cache key.
    pub key: WgpuPipelineKey,
    /// Stable symbol for this validated pipeline.
    pub symbol: Symbol,
    /// WGSL source byte length.
    pub wgsl_bytes: usize,
}

/// Snapshot of cache pressure and reuse.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WgpuPipelineCacheSnapshot {
    /// Cached pipeline count.
    pub entries: usize,
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Number of oldest-entry evictions.
    pub evictions: usize,
}

/// Small FIFO cache for validated pipelines.
#[derive(Clone, Debug)]
pub struct WgpuPipelineCache {
    capacity: usize,
    entries: VecDeque<WgpuPipelineRecord>,
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl WgpuPipelineCache {
    /// Builds a cache with a bounded entry count.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Returns an existing pipeline or validates and inserts one.
    pub fn get_or_insert(
        &mut self,
        probe: &WgpuAdapterProbe,
        op: WgpuKernelOp,
        dtype: WgpuKernelDType,
        rank: usize,
    ) -> WgpuPipelineRecord {
        let key = WgpuPipelineKey {
            adapter_ordinal: probe.adapter.ordinal,
            op,
            dtype,
            rank,
        };
        if let Some(record) = self.entries.iter().find(|record| record.key == key) {
            self.hits += 1;
            return record.clone();
        }
        self.misses += 1;
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
            self.evictions += 1;
        }
        let record = WgpuPipelineRecord {
            symbol: Symbol::qualified(
                "compute.pipeline.wgpu",
                format!(
                    "{}/{}/{}/rank-{}",
                    probe.adapter.ordinal,
                    op.symbol_name(),
                    dtype.symbol_name(),
                    rank
                ),
            ),
            key,
            wgsl_bytes: PORTABLE_ELEMENTWISE_WGSL.len(),
        };
        self.entries.push_back(record.clone());
        record
    }

    /// Returns cache pressure and reuse counters.
    pub fn snapshot(&self) -> WgpuPipelineCacheSnapshot {
        WgpuPipelineCacheSnapshot {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }
}

impl Default for WgpuPipelineCache {
    fn default() -> Self {
        Self::new(16)
    }
}
