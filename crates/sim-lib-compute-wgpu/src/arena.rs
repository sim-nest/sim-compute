//! Bounded resident allocation arena for wgpu tensor planning.

use std::collections::VecDeque;

/// Opaque resident allocation id inside a planned wgpu arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WgpuAllocationId(u64);

/// Allocation record returned by the arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuArenaAllocation {
    /// Allocation id.
    pub id: WgpuAllocationId,
    /// Allocation size in bytes.
    pub bytes: u64,
}

#[derive(Clone, Debug)]
struct LiveAllocation {
    allocation: WgpuArenaAllocation,
    active: bool,
}

/// Snapshot of resident arena pressure.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WgpuArenaSnapshot {
    /// Active resident allocation count.
    pub live_allocations: usize,
    /// Active resident bytes.
    pub resident_bytes: u64,
    /// Evictions performed to stay inside the bound.
    pub evictions: usize,
}

/// Bounded resident arena with oldest-first eviction.
#[derive(Clone, Debug)]
pub struct WgpuResidentArena {
    max_bytes: u64,
    next_id: u64,
    resident_bytes: u64,
    evictions: usize,
    allocations: VecDeque<LiveAllocation>,
}

impl WgpuResidentArena {
    /// Creates an arena with a byte ceiling.
    pub fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            next_id: 0,
            resident_bytes: 0,
            evictions: 0,
            allocations: VecDeque::new(),
        }
    }

    /// Allocates resident bytes, evicting older allocations when necessary.
    pub fn allocate(&mut self, bytes: u64) -> Result<WgpuArenaAllocation, String> {
        if bytes > self.max_bytes {
            return Err("wgpu resident allocation exceeds arena".to_owned());
        }
        while self.resident_bytes.saturating_add(bytes) > self.max_bytes {
            let Some(mut allocation) = self.allocations.pop_front() else {
                break;
            };
            if allocation.active {
                allocation.active = false;
                self.resident_bytes = self
                    .resident_bytes
                    .saturating_sub(allocation.allocation.bytes);
                self.evictions += 1;
            }
            self.allocations.push_back(allocation);
        }
        self.next_id += 1;
        let allocation = WgpuArenaAllocation {
            id: WgpuAllocationId(self.next_id),
            bytes,
        };
        self.resident_bytes += bytes;
        self.allocations.push_back(LiveAllocation {
            allocation: allocation.clone(),
            active: true,
        });
        Ok(allocation)
    }

    /// Returns whether an allocation is still resident.
    pub fn contains(&self, id: WgpuAllocationId) -> bool {
        self.allocations
            .iter()
            .any(|allocation| allocation.active && allocation.allocation.id == id)
    }

    /// Returns current arena pressure.
    pub fn snapshot(&self) -> WgpuArenaSnapshot {
        WgpuArenaSnapshot {
            live_allocations: self
                .allocations
                .iter()
                .filter(|allocation| allocation.active)
                .count(),
            resident_bytes: self.resident_bytes,
            evictions: self.evictions,
        }
    }
}
