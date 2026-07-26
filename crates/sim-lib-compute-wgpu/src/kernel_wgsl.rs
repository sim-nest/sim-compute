//! WGSL source fixtures for portable wgpu pipelines.

/// WGSL source used by real pointwise wgpu dispatches.
pub const POINTWISE_DISPATCH_WGSL: &str = r#"
struct KernelParams { op: u32, len: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: KernelParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) { return; }
    let x = left[i];
    let y = right[i];
    if (params.op == 0u) { out[i] = x + y; }
    else if (params.op == 1u) { out[i] = x - y; }
    else if (params.op == 2u) { out[i] = x * y; }
    else if (params.op == 3u) { out[i] = x / y; }
    else if (params.op == 4u) { out[i] = -x; }
    else if (params.op == 5u) { out[i] = sqrt(x); }
    else if (params.op == 6u) { out[i] = exp(x); }
    else if (params.op == 7u) { out[i] = log(x); }
    else if (params.op == 8u) { out[i] = sin(x); }
    else { out[i] = cos(x); }
}
"#;

/// Portable WGSL source used by validated element-wise pipelines.
pub const PORTABLE_ELEMENTWISE_WGSL: &str = r#"
struct KernelParams { op: u32, len: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: KernelParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i >= params.len) { return; }
    let x = left[i];
    let y = right[i];
    if (params.op == 0u) { out[i] = x + y; }
    else if (params.op == 1u) { out[i] = x - y; }
    else if (params.op == 2u) { out[i] = x * y; }
    else if (params.op == 3u) { out[i] = x / y; }
    else if (params.op == 4u) { out[i] = sqrt(x); }
    else if (params.op == 5u) { out[i] = exp(x); }
    else if (params.op == 6u) { out[i] = sin(x); }
    else { out[i] = cos(x); }
}
"#;

/// Portable WGSL source used by fixed-tree reduction pipelines.
pub const PORTABLE_REDUCTION_WGSL: &str = r#"
struct KernelParams { op: u32, len: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<uniform> params: KernelParams;
var<workgroup> lane: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let base = wid.x * 512u + lid.x;
    var value = select(0.0, 3.4028234663852886e38, params.op == 1u);
    value = select(value, -3.4028234663852886e38, params.op == 2u);
    for (var pass = 0u; pass < 2u; pass = pass + 1u) {
        let index = base + pass * 256u;
        if (index < params.len) {
            let x = input[index];
            if (params.op == 1u) { value = min(value, x); }
            else if (params.op == 2u) { value = max(value, x); }
            else if (params.op == 3u) { value = value + x * x; }
            else { value = value + x; }
        }
    }
    lane[lid.x] = value;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) {
            let other = lane[lid.x + stride];
            if (params.op == 1u) { lane[lid.x] = min(lane[lid.x], other); }
            else if (params.op == 2u) { lane[lid.x] = max(lane[lid.x], other); }
            else { lane[lid.x] = lane[lid.x] + other; }
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { partials[wid.x] = lane[0]; }
}
"#;

/// Portable WGSL source used by transpose, dot, and tiled matmul pipelines.
pub const PORTABLE_LINALG_WGSL: &str = r#"
struct KernelParams { rows: u32, inner: u32, cols: u32, op: u32 }
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: KernelParams;
var<workgroup> tile_left: array<f32, 256>;
var<workgroup> tile_right: array<f32, 256>;
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    if (params.op == 0u) {
        if (gid.x < params.cols && gid.y < params.rows) {
            out[gid.x * params.rows + gid.y] = left[gid.y * params.cols + gid.x];
        }
        return;
    }
    if (params.op == 1u) {
        let i = lid.y * 16u + lid.x;
        var acc = 0.0;
        var k = i;
        loop {
            if (k >= params.inner) { break; }
            acc = acc + left[k] * right[k];
            k = k + 256u;
        }
        tile_left[i] = acc;
        workgroupBarrier();
        var stride = 128u;
        loop {
            if (stride == 0u) { break; }
            if (i < stride) { tile_left[i] = tile_left[i] + tile_left[i + stride]; }
            workgroupBarrier();
            stride = stride / 2u;
        }
        if (i == 0u) { out[0] = tile_left[0]; }
        return;
    }
    if (gid.x >= params.cols || gid.y >= params.rows) { return; }
    var acc = 0.0;
    var base = 0u;
    loop {
        if (base >= params.inner) { break; }
        let local_index = lid.y * 16u + lid.x;
        let left_k = base + lid.x;
        let right_k = base + lid.y;
        tile_left[local_index] = select(0.0, left[gid.y * params.inner + left_k], left_k < params.inner);
        tile_right[local_index] = select(0.0, right[right_k * params.cols + gid.x], right_k < params.inner);
        workgroupBarrier();
        for (var k = 0u; k < 16u; k = k + 1u) {
            acc = acc + tile_left[lid.y * 16u + k] * tile_right[k * 16u + lid.x];
        }
        workgroupBarrier();
        base = base + 16u;
    }
    out[gid.y * params.cols + gid.x] = acc;
}
"#;
