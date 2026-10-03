//! Manifest-driven training: `cargo run --example train -- configs/<run>.json [gpu]`
//! Single manifest; for chained stages see the `chain` example.
//! The manifest path is required: silently defaulting could launch a long
//! run on the wrong config.

use generalist::{harness::load_run, train::run_stage};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .expect("usage: cargo run --example train -- configs/<run>.json [gpu]");
    let gpu = args.iter().any(|a| a == "gpu");
    let run =
        load_run(std::path::Path::new(path)).unwrap_or_else(|e| panic!("load manifest:\n{e}"));
    if gpu {
        #[cfg(feature = "cuda")]
        {
            use burn::backend::{Autodiff, Cuda, cuda::CudaDevice};
            println!("backend: CUDA (fused)");
            run_stage::<Autodiff<Cuda>>(&run, &CudaDevice::default(), None, &[]);
        }
        #[cfg(not(feature = "cuda"))]
        {
            panic!("gpu requested but the `cuda` feature is off");
        }
    } else {
        // Feature-gated: naming `burn::backend::NdArray` unconditionally broke
        // the cuda-only build, which is the feature set the GPU runs use.
        #[cfg(feature = "ndarray")]
        {
            println!("backend: NdArray (CPU)");
            run_stage::<burn::backend::Autodiff<burn::backend::NdArray>>(
                &run,
                &Default::default(),
                None,
                &[],
            );
        }
        #[cfg(not(feature = "ndarray"))]
        panic!("cpu requested but the `ndarray` feature is off");
    }
}
