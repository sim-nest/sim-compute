//! Bounded submission queue planning for wgpu tensor work.

/// Snapshot of queued wgpu tensor submissions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WgpuQueueSnapshot {
    /// Queued node count.
    pub nodes: usize,
    /// Queued byte count.
    pub bytes: u64,
}

/// Queue bounds for resident tensor submissions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuQueueLimits {
    /// Maximum queued nodes.
    pub max_nodes: usize,
    /// Maximum queued bytes.
    pub max_bytes: u64,
    /// Maximum accepted deadline tick.
    pub deadline_tick: u64,
}

/// Deterministic bounded queue used before command submission.
#[derive(Clone, Debug)]
pub struct WgpuSubmissionQueue {
    limits: WgpuQueueLimits,
    nodes: usize,
    bytes: u64,
}

impl WgpuSubmissionQueue {
    /// Builds an empty bounded queue.
    pub fn new(limits: WgpuQueueLimits) -> Self {
        Self {
            limits,
            nodes: 0,
            bytes: 0,
        }
    }

    /// Accepts one submission when node, byte, and deadline bounds hold.
    pub fn push(&mut self, bytes: u64, deadline_tick: u64) -> Result<(), String> {
        if deadline_tick > self.limits.deadline_tick {
            return Err("wgpu submission deadline expired".to_owned());
        }
        if self.nodes >= self.limits.max_nodes {
            return Err("wgpu submission queue node limit reached".to_owned());
        }
        if self.bytes.saturating_add(bytes) > self.limits.max_bytes {
            return Err("wgpu submission queue byte limit reached".to_owned());
        }
        self.nodes += 1;
        self.bytes += bytes;
        Ok(())
    }

    /// Clears synchronized submissions.
    pub fn flush(&mut self) -> WgpuQueueSnapshot {
        let snapshot = self.snapshot();
        self.nodes = 0;
        self.bytes = 0;
        snapshot
    }

    /// Returns current queued pressure.
    pub fn snapshot(&self) -> WgpuQueueSnapshot {
        WgpuQueueSnapshot {
            nodes: self.nodes,
            bytes: self.bytes,
        }
    }
}
