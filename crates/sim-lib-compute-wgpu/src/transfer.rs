//! Transfer planning for segmented wgpu tensor payloads.

use crate::WgpuSegmentPlan;

/// One host-to-device or device-to-host copy span.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuTransferSpan {
    /// Byte offset from the logical tensor start.
    pub offset: u64,
    /// Copy length in bytes.
    pub bytes: u64,
}

/// Planned transfer spans for a segment plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuTransferPlan {
    /// Copy spans.
    pub spans: Vec<WgpuTransferSpan>,
}

impl WgpuTransferPlan {
    /// Builds a transfer plan aligned to resident segments.
    pub fn from_segments(plan: &WgpuSegmentPlan) -> Self {
        Self {
            spans: plan
                .segments
                .iter()
                .map(|segment| WgpuTransferSpan {
                    offset: segment.offset,
                    bytes: segment.bytes,
                })
                .collect(),
        }
    }
}
