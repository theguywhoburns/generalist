//! Single swap point for the test-suite backend.
//!
//! Every `#[cfg(test)]` module uses [`TestBackend`] + [`test_device`].
//! To run the whole suite on another backend, change the three marked lines
//! below (e.g. `NdArray` -> `Cuda`, `NdArrayDevice` -> `CudaDevice`).

use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};

/// Backend for the entire test suite. CHANGE THIS to swap backends.
/// GPU runs via `Autodiff<Cuda>` are already kernel-fused (the `Cuda` alias
/// wraps `Fusion` while the `fusion` feature is on).
pub(crate) type TestBackend = Autodiff<NdArray>;
/// Device matching [`TestBackend`]. CHANGE THIS to swap backends.
pub(crate) type TestDevice = NdArrayDevice;

/// Device for the entire test suite. CHANGE THIS to swap backends.
pub(crate) fn test_device() -> TestDevice {
    TestDevice::default()
}

/// GPU presence check, independent of the suite backend above.
/// Note: `burn::backend::Cuda` is already `Fusion`-wrapped while the
/// `fusion` feature is on, so this exercises the fused path.
#[cfg(feature = "cuda")]
#[test]
fn cuda_smoke() {
    use burn::backend::{Cuda, cuda::CudaDevice};
    use burn::tensor::Tensor;

    let device = CudaDevice::default();
    let x = Tensor::<Cuda, 2>::ones([4, 4], &device);
    let vals = x
        .clone()
        .matmul(x)
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec();
    assert!(vals.iter().all(|v| (*v - 4.0).abs() < 1e-4));
}
