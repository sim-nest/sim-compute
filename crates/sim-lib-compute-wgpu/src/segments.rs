//! Segment planning for resident wgpu tensor buffers.

/// One contiguous resident buffer segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuResidentSegment {
    /// Segment ordinal.
    pub index: usize,
    /// Byte offset from the start of the logical tensor.
    pub offset: u64,
    /// Segment length in bytes.
    pub bytes: u64,
}

/// Checked segment plan for a tensor payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuSegmentPlan {
    /// Planned segments.
    pub segments: Vec<WgpuResidentSegment>,
}

impl WgpuSegmentPlan {
    /// Splits a payload by the stricter of the tile size and binding boundary.
    pub fn new(total_bytes: u64, tile_bytes: u64, binding_bytes: u64) -> Self {
        let boundary = tile_bytes.min(binding_bytes).max(1);
        let mut segments = Vec::new();
        let mut offset = 0;
        while offset < total_bytes {
            let bytes = (total_bytes - offset).min(boundary);
            segments.push(WgpuResidentSegment {
                index: segments.len(),
                offset,
                bytes,
            });
            offset += bytes;
        }
        Self { segments }
    }

    /// Returns the number of resident bytes in the plan.
    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|segment| segment.bytes).sum()
    }
}
