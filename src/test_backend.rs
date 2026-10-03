//! Single swap point for the test-suite backend.
//!
//! Every `#[cfg(test)]` module uses [`TestBackend`] + [`test_device`]. The
//! suite runs on the CPU when the `ndarray` feature is on and falls back to
//! CUDA when it is off, so `cargo test --features cuda` runs the whole suite
//! on the GPU without editing this file.

/// The suite runs on the CPU by default and falls back to the GPU when the
/// `ndarray` feature is off.
///
/// The fallback exists because naming `NdArray` unconditionally made a
/// cuda-only build fail to compile -- and cuda-only is the feature set the GPU
/// sweeps actually use, so "run the suite on the GPU" was documented here but
/// not actually buildable. With the fallback it is, at the cost of needing a
/// working GPU.
#[cfg(feature = "ndarray")]
mod selected {
    use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};

    pub(crate) type TestBackend = Autodiff<NdArray>;
    pub(crate) type TestDevice = NdArrayDevice;
}

#[cfg(all(feature = "cuda", not(feature = "ndarray")))]
mod selected {
    use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};

    pub(crate) type TestBackend = Autodiff<Cuda>;
    pub(crate) type TestDevice = CudaDevice;
}

#[cfg(not(any(feature = "ndarray", feature = "cuda")))]
compile_error!("the test suite needs a backend: enable `ndarray` (CPU, the default) or `cuda`");

pub(crate) use selected::{TestBackend, TestDevice};

/// Device for the entire test suite.
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
