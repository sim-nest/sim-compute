//! Narrow runtime-loaded CUDA execution boundary.

use std::{ffi::c_void, sync::Arc};

use libloading::Library;

use crate::{CudaLibrarySet, CudaLoadError};

const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUBLAS_OP_N: i32 = 0;

type CudaMalloc = unsafe extern "C" fn(*mut *mut c_void, usize) -> i32;
type CudaFree = unsafe extern "C" fn(*mut c_void) -> i32;
type CudaMemcpy = unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32) -> i32;
type CudaDeviceSynchronize = unsafe extern "C" fn() -> i32;
type CublasCreate = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type CublasDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type CublasSgemm = unsafe extern "C" fn(
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

pub(crate) struct CudaDeviceBuffer {
    runtime: Arc<CudaLibrarySet>,
    address: usize,
    len: usize,
}

// SAFETY: CUDA device allocations are opaque integer handles. CUDA runtime and
// cuBLAS calls are thread-safe, and the allocation is freed exactly once by
// `Drop` after the last Arc owner is released.
unsafe impl Send for CudaDeviceBuffer {}
// SAFETY: See the `Send` justification; no host dereference of the device
// address is ever performed.
unsafe impl Sync for CudaDeviceBuffer {}

impl CudaDeviceBuffer {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn runtime(&self) -> &Arc<CudaLibrarySet> {
        &self.runtime
    }

    fn pointer(&self) -> *mut c_void {
        self.address as *mut c_void
    }

    pub(crate) fn read(&self) -> Result<Vec<f32>, CudaLoadError> {
        self.runtime.download(self)
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        let _ = self.runtime.free(self.pointer());
    }
}

impl CudaLibrarySet {
    pub(crate) fn upload(
        self: &Arc<Self>,
        values: &[f32],
    ) -> Result<Arc<CudaDeviceBuffer>, CudaLoadError> {
        let bytes = byte_count(values.len())?;
        let (runtime, _) = self.execution_handles();
        let malloc = symbol::<CudaMalloc>(runtime, b"cudaMalloc\0")?;
        let copy = symbol::<CudaMemcpy>(runtime, b"cudaMemcpy\0")?;
        let mut pointer = std::ptr::null_mut();
        // SAFETY: Function pointers use the documented CUDA runtime ABI.
        // `pointer` is an initialized out parameter and `values` covers `bytes`.
        let status = unsafe { malloc(&mut pointer, bytes) };
        check_cuda(status, "cudaMalloc")?;
        // SAFETY: CUDA owns `pointer` for `bytes`, and the host slice is valid
        // for the same byte count.
        let status = unsafe {
            copy(
                pointer,
                values.as_ptr().cast::<c_void>(),
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if let Err(error) = check_cuda(status, "cudaMemcpy host-to-device") {
            let _ = self.free(pointer);
            return Err(error);
        }
        Ok(Arc::new(CudaDeviceBuffer {
            runtime: Arc::clone(self),
            address: pointer as usize,
            len: values.len(),
        }))
    }

    pub(crate) fn matmul(
        self: &Arc<Self>,
        left: &CudaDeviceBuffer,
        right: &CudaDeviceBuffer,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Arc<CudaDeviceBuffer>, CudaLoadError> {
        validate_matrix_lengths(left.len(), right.len(), rows, inner, cols)?;
        if !Arc::ptr_eq(self, left.runtime()) || !Arc::ptr_eq(self, right.runtime()) {
            return Err(error("CUDA inputs belong to another runtime"));
        }
        let output_len = rows
            .checked_mul(cols)
            .ok_or_else(|| error("CUDA output length overflowed"))?;
        let zeros = vec![0.0_f32; output_len];
        let output = self.upload(&zeros)?;
        let (_, cublas) = self.execution_handles();
        let create = symbol::<CublasCreate>(cublas, b"cublasCreate_v2\0")?;
        let destroy = symbol::<CublasDestroy>(cublas, b"cublasDestroy_v2\0")?;
        let sgemm = symbol::<CublasSgemm>(cublas, b"cublasSgemm_v2\0")?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: Function pointers use the documented cuBLAS ABI and the
        // handle out parameter is initialized.
        check_cublas(unsafe { create(&mut handle) }, "cublasCreate_v2")?;
        let dimensions = matrix_dimensions(rows, inner, cols)?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        // Row-major C=A*B is column-major C^T=B^T*A^T.
        // SAFETY: All device buffers cover the validated dimensions; cuBLAS
        // receives the documented column-major SGEMM arguments.
        let status = unsafe {
            sgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
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
        let gemm_result = check_cublas(status, "cublasSgemm_v2");
        // SAFETY: `handle` came from the successful create call.
        let destroy_result = check_cublas(unsafe { destroy(handle) }, "cublasDestroy_v2");
        gemm_result?;
        destroy_result?;
        Ok(output)
    }

    /// Runs one row-major dense f32 matrix multiplication through cuBLAS and
    /// returns a synchronized host result.
    pub fn matmul_f32(
        self: &Arc<Self>,
        left: &[f32],
        right: &[f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Result<Vec<f32>, CudaLoadError> {
        let left = self.upload(left)?;
        let right = self.upload(right)?;
        self.matmul(&left, &right, rows, inner, cols)?.read()
    }

    fn download(&self, buffer: &CudaDeviceBuffer) -> Result<Vec<f32>, CudaLoadError> {
        let bytes = byte_count(buffer.len())?;
        let (runtime, _) = self.execution_handles();
        let copy = symbol::<CudaMemcpy>(runtime, b"cudaMemcpy\0")?;
        let synchronize = symbol::<CudaDeviceSynchronize>(runtime, b"cudaDeviceSynchronize\0")?;
        let mut values = vec![0.0_f32; buffer.len()];
        // SAFETY: The output vector covers `bytes`, and the CUDA allocation was
        // created for the same logical f32 length.
        check_cuda(
            unsafe {
                copy(
                    values.as_mut_ptr().cast::<c_void>(),
                    buffer.pointer(),
                    bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy device-to-host",
        )?;
        // SAFETY: The function has no arguments and uses the documented ABI.
        check_cuda(unsafe { synchronize() }, "cudaDeviceSynchronize")?;
        Ok(values)
    }

    fn free(&self, pointer: *mut c_void) -> Result<(), CudaLoadError> {
        if pointer.is_null() {
            return Ok(());
        }
        let (runtime, _) = self.execution_handles();
        let free = symbol::<CudaFree>(runtime, b"cudaFree\0")?;
        // SAFETY: The pointer was returned by `cudaMalloc` and is freed once.
        check_cuda(unsafe { free(pointer) }, "cudaFree")
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
) -> Result<MatrixDimensions, CudaLoadError> {
    Ok(MatrixDimensions {
        rows: i32::try_from(rows).map_err(|_| error("CUDA row count exceeds i32"))?,
        inner: i32::try_from(inner).map_err(|_| error("CUDA inner count exceeds i32"))?,
        cols: i32::try_from(cols).map_err(|_| error("CUDA column count exceeds i32"))?,
    })
}

fn validate_matrix_lengths(
    left: usize,
    right: usize,
    rows: usize,
    inner: usize,
    cols: usize,
) -> Result<(), CudaLoadError> {
    if rows.checked_mul(inner) != Some(left) || inner.checked_mul(cols) != Some(right) {
        return Err(error("CUDA matmul shape does not match input lengths"));
    }
    Ok(())
}

fn byte_count(len: usize) -> Result<usize, CudaLoadError> {
    len.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| error("CUDA byte count overflowed"))
}

fn symbol<'library, T>(
    library: &'library Library,
    name: &[u8],
) -> Result<libloading::Symbol<'library, T>, CudaLoadError> {
    // SAFETY: Every caller supplies the official function signature for the
    // named CUDA/cuBLAS symbol. Library handles outlive returned symbols.
    unsafe { library.get(name) }.map_err(|load| error(load.to_string()))
}

fn check_cuda(status: i32, operation: &str) -> Result<(), CudaLoadError> {
    (status == 0)
        .then_some(())
        .ok_or_else(|| error(format!("{operation} failed with CUDA status {status}")))
}

fn check_cublas(status: i32, operation: &str) -> Result<(), CudaLoadError> {
    (status == 0)
        .then_some(())
        .ok_or_else(|| error(format!("{operation} failed with cuBLAS status {status}")))
}

fn error(message: impl Into<String>) -> CudaLoadError {
    CudaLoadError {
        message: message.into(),
    }
}
