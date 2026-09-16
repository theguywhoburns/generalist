//! Manifest-driven training: `cargo run --example train -- configs/<run>.json [gpu]`
//! Single manifest; for chained stages see the `curriculum` example.
//! The manifest path is required: silently defaulting could launch a long
//! run on the wrong config.

use generalist::{harness::RunConfig, train::run_stage};

fn main() {
    generalist::fail_fast::install();
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).expect(
        "usage: cargo run --example train -- configs/<run>.json [gpu]",
    );
    let gpu = args.iter().any(|a| a == "gpu");
    let run = RunConfig::load_json(std::path::Path::new(path)).expect("load manifest");
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
        println!("backend: NdArray (CPU)");
        run_stage::<burn::backend::Autodiff<burn::backend::NdArray>>(&run, &Default::default(), None, &[]);
    }
}
