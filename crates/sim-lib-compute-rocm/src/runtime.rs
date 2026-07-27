//! Narrow runtime-loaded HIP/rocBLAS execution boundary.

use std::{ffi::c_void, sync::Arc};

use libloading::Library;

use crate::{RocmLibrarySet, RocmLoadError};

const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const ROCBLAS_OPERATION_NONE: i32 = 111;

type HipMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type HipFree = unsafe extern "C" fn(*mut c_void) -> i32;
type HipMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
type HipDeviceSynchronize = unsafe extern "C" fn() -> i32;
type RocblasCreate = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type RocblasDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type RocblasSgemm = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    i32,
    *const f32,
    *const f32,
    i32,
    *const f32,
    i32,
    *const f32,
    *mut f32,
    i32,
) -> i32;

pub(crate) struct RocmDeviceBuffer {
    runtime: Arc<RocmLibrarySet>,
    address: usize,
    len: usize,
}

// SAFETY: HIP device allocations are opaque integer handles. HIP and rocBLAS
// calls are thread-safe, and the allocation is freed once after its last Arc.
unsafe impl Send for RocmDeviceBuffer {}
// SAFETY: See `Send`; device addresses are never dereferenced by host code.
unsafe impl Sync for RocmDeviceBuffer {}

impl RocmDeviceBuffer {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn runtime(&self) -> &Arc<RocmLibrarySet> {
        &self.runtime
    }

    fn pointer(&self) -> *mut c_void {
        self.address as *mut c_void
    }

    pub(crate) fn read(&self) -> Result<Vec<f32>, RocmLoadError> {
        self.runtime.download(self)
    }
}

impl Drop for RocmDeviceBuffer {
    fn drop(&mut self) {
        let _ = self.runtime.free(self.pointer());
    }
}

impl RocmLibrarySet {
    pub(crate) fn upload(
        self: &Arc<Self>,
        values: &[f32],
    ) -> Result<Arc<RocmDeviceBuffer>, RocmLoadError> {
        let bytes = byte_count(values.len())?;
        let (hip, _) = self.execution_handles();
        let malloc = symbol::<HipMalloc>(hip, b"hipMalloc\0")?;
        let copy = symbol::<HipMemcpy>(hip, b"hipMemcpy\0")?;
        let mut pointer = std::ptr::null_mut();
        // SAFETY: Function pointers use the documented HIP runtime ABI.
        check_hip(unsafe { malloc(&mut pointer, bytes) }, "hipMalloc")?;
        // SAFETY: HIP owns `pointer` for `bytes`, and the host slice is valid
        // for the same byte count.
        let status = unsafe {
            copy(
                pointer,
                values.as_ptr().cast::<c_void>(),
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if let Err(error) = check_hip(status, "hipMemcpy host-to-device") {
            let _ = self.free(pointer);
            return Err(error);
        }
        Ok(Arc::new(RocmDeviceBuffer {
            runtime: Arc::clone(self),
            address: pointer as usize,
            len: values.len(),
        }))
    }

    pub(crate) fn matmul(
        self: &Arc<Self>,
        left: &RocmDeviceBuffer,
        right: &RocmDeviceBuffer,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Arc<RocmDeviceBuffer>, RocmLoadError> {
        validate_matrix_lengths(left.len(), right.len(), rows, inner, cols)?;
        if !Arc::ptr_eq(self, left.runtime()) || !Arc::ptr_eq(self, right.runtime()) {
            return Err(error("ROCm inputs belong to another runtime"));
        }
        let output_len = rows
            .checked_mul(cols)
            .ok_or_else(|| error("ROCm output length overflowed"))?;
        let output = self.upload(&vec![0.0_f32; output_len])?;
        let (_, rocblas) = self.execution_handles();
        let create = symbol::<RocblasCreate>(rocblas, b"rocblas_create_handle\0")?;
        let destroy = symbol::<RocblasDestroy>(rocblas, b"rocblas_destroy_handle\0")?;
        let sgemm = symbol::<RocblasSgemm>(rocblas, b"rocblas_sgemm\0")?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: Function pointers use the documented rocBLAS ABI.
        check_rocblas(unsafe { create(&mut handle) }, "rocblas_create_handle")?;
        let dimensions = matrix_dimensions(rows, inner, cols)?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        // Row-major C=A*B is column-major C^T=B^T*A^T.
        // SAFETY: Device buffers cover the validated dimensions.
        let status = unsafe {
            sgemm(
                handle,
                ROCBLAS_OPERATION_NONE,
                ROCBLAS_OPERATION_NONE,
                dimensions.cols,
                dimensions.rows,
                dimensions.inner,
                &alpha,
                right.pointer().cast::<f32>(),
                dimensions.cols,
                left.pointer().cast::<f32>(),
                dimensions.inner,
                &beta,
                output.pointer().cast::<f32>(),
                dimensions.cols,
            )
        };
        let gemm_result = check_rocblas(status, "rocblas_sgemm");
        // SAFETY: `handle` came from the successful create call.
        let destroy_result = check_rocblas(unsafe { destroy(handle) }, "rocblas_destroy_handle");
        gemm_result?;
        destroy_result?;
        Ok(output)
    }

    /// Runs one row-major dense f32 matrix multiplication through rocBLAS and
    /// returns a synchronized host result.
    pub fn matmul_f32(
        self: &Arc<Self>,
        left: &[f32],
        right: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, RocmLoadError> {
        let left = self.upload(left)?;
        let right = self.upload(right)?;
        self.matmul(&left, &right, rows, inner, cols)?.read()
    }

    fn download(&self, buffer: &RocmDeviceBuffer) -> Result<Vec<f32>, RocmLoadError> {
        let bytes = byte_count(buffer.len())?;
        let (hip, _) = self.execution_handles();
        let copy = symbol::<HipMemcpy>(hip, b"hipMemcpy\0")?;
        let synchronize = symbol::<HipDeviceSynchronize>(hip, b"hipDeviceSynchronize\0")?;
        let mut values = vec![0.0_f32; buffer.len()];
        // SAFETY: Host and device allocations both cover `bytes`.
        check_hip(
            unsafe {
                copy(
                    values.as_mut_ptr().cast::<c_void>(),
                    buffer.pointer(),
                    bytes,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy device-to-host",
        )?;
        // SAFETY: The function has no arguments and uses the documented ABI.
        check_hip(unsafe { synchronize() }, "hipDeviceSynchronize")?;
        Ok(values)
    }

    fn free(&self, pointer: *mut c_void) -> Result<(), RocmLoadError> {
        if pointer.is_null() {
            return Ok(());
        }
        let (hip, _) = self.execution_handles();
        let free = symbol::<HipFree>(hip, b"hipFree\0")?;
        // SAFETY: The pointer came from `hipMalloc` and is freed once.
        check_hip(unsafe { free(pointer) }, "hipFree")
    }
}

struct MatrixDimensions {
    rows: i32,
    inner: i32,
    cols: i32,
}

fn matrix_dimensions(
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<MatrixDimensions, RocmLoadError> {
    Ok(MatrixDimensions {
        rows: i32::try_from(rows).map_err(|_| error("ROCm row count exceeds i32"))?,
        inner: i32::try_from(inner).map_err(|_| error("ROCm inner count exceeds i32"))?,
        cols: i32::try_from(cols).map_err(|_| error("ROCm column count exceeds i32"))?,
    })
}

fn validate_matrix_lengths(
    left: usize,
    right: usize,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), RocmLoadError> {
    if rows.checked_mul(inner) != Some(left) || inner.checked_mul(cols) != Some(right) {
        return Err(error("ROCm matmul shape does not match input lengths"));
    }
    Ok(())
}

fn byte_count(len: usize) -> Result<usize, RocmLoadError> {
    len.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| error("ROCm byte count overflowed"))
}

fn symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
) -> Result<libloading::Symbol<'library, T>, RocmLoadError> {
    // SAFETY: Callers supply the official HIP/rocBLAS signature for each name.
    unsafe { library.get(name) }.map_err(|load| error(load.to_string()))
}

fn check_hip(status: i32, operation: &str) -> Result<(), RocmLoadError> {
    (status == 0)
        .then_some(())
        .ok_or_else(|| error(format!("{operation} failed with HIP status {status}")))
}

fn check_rocblas(status: i32, operation: &str) -> Result<(), RocmLoadError> {
    (status == 0)
        .then_some(())
        .ok_or_else(|| error(format!("{operation} failed with rocBLAS status {status}")))
}

fn error(message: impl Into<String>) -> RocmLoadError {
    RocmLoadError {
        message: message.into(),
    }
}
