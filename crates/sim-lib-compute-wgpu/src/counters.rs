//! Physical queue-derived evidence for resident wgpu submissions.

use std::sync::{Arc, Mutex};

use crate::WgpuResidentSegment;

/// Physical buffer range touched by a resident wgpu submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferRange {
    /// Byte offset inside the logical resident tensor buffer.
    pub offset: u64,
    /// Number of bytes touched.
    pub bytes: u64,
}

impl From<&WgpuResidentSegment> for BufferRange {
    fn from(segment: &WgpuResidentSegment) -> Self {
        Self {
            offset: segment.offset,
            bytes: segment.bytes,
        }
    }
}

/// Queue-derived evidence for physical wgpu tensor work.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PhysicalSubmissionEvidence {
    /// Queue submissions issued by the executor.
    pub submissions: u64,
    /// Host bytes uploaded through queue writes.
    pub uploaded_bytes: u64,
    /// Full tensor readbacks requested by terminal materialization.
    pub full_readbacks: u64,
    /// Scalar synchronizations performed for scalar reductions.
    pub scalar_syncs: u64,
    /// Resident ranges touched by submitted command buffers.
    pub segments_touched: Vec<BufferRange>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct WgpuPhysicalCounters {
    inner: Arc<Mutex<PhysicalSubmissionEvidence>>,
}

impl WgpuPhysicalCounters {
    pub(crate) fn snapshot(&self) -> PhysicalSubmissionEvidence {
        self.inner
            .lock()
            .expect("wgpu physical counters poisoned")
            .clone()
    }

    pub(crate) fn record_upload(&self, bytes: u64) {
        let mut evidence = self.inner.lock().expect("wgpu physical counters poisoned");
        evidence.uploaded_bytes = evidence.uploaded_bytes.saturating_add(bytes);
    }

    pub(crate) fn record_submit(&self, segments: &[WgpuResidentSegment]) {
        let mut evidence = self.inner.lock().expect("wgpu physical counters poisoned");
        evidence.submissions = evidence.submissions.saturating_add(1);
        evidence
            .segments_touched
            .extend(segments.iter().map(BufferRange::from));
    }

    pub(crate) fn record_full_readback(&self) {
        let mut evidence = self.inner.lock().expect("wgpu physical counters poisoned");
        evidence.full_readbacks = evidence.full_readbacks.saturating_add(1);
    }

    pub(crate) fn record_scalar_sync(&self) {
        let mut evidence = self.inner.lock().expect("wgpu physical counters poisoned");
        evidence.scalar_syncs = evidence.scalar_syncs.saturating_add(1);
    }
}
