//! CPU micro-profile (NdArray — GPU untouched). Times full-size forward
//! (fixed loops 1/4/8, ACT, converge) and one backward, then prints a static
//! FLOP/memory model. Run with `cargo run --example profile`.

use std::time::Instant;

use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Int, Tensor},
};
use generalist::model::{LoopedConfig, LoopedTransformer, StopMode};

type B = Autodiff<NdArray>;

fn ms_of(f: impl FnOnce()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let device = Default::default();
    let config = LoopedConfig::base_1m();
    println!("params: {}", config.param_count());

    // Static model per token per loop: attn ~8*d^2 (QKVO+scores+mix at T=64
    // adds ~4*T*d for scores), mlp ~6*d*h. d=256, h=768, T=64:
    // attn ~= 8*65536 + 4*64*256 = 589_824 FLOP, mlp ~= 6*256*768 = 1_179_648.
    let d = 256.0f64;
    let h = 768.0f64;
    let t = 64.0f64;
    let attn = 8.0 * d * d + 4.0 * t * d;
    let mlp = 6.0 * d * h;
    println!("static FLOP/token/loop: attn {attn:.0}, mlp {mlp:.0}, total {:.0}", attn + mlp);
    println!("static FLOP/token @8 loops: {:.2} MFLOP", 8.0 * (attn + mlp) / 1e6);

    let model = LoopedTransformer::<B>::new(&config, &device);
    let tokens: Tensor<B, 2, Int> = Tensor::zeros([2, 64], &device);

    // Warmup (autotune/fusion caches).
    let lens = vec![64usize, 64];
    let _ = model.forward(tokens.clone(), &config, StopMode::Fixed { loops: 1 }, &lens);

    for loops in [1, 4, 8] {
        let ms = ms_of(|| {
            let _ = model.forward(tokens.clone(), &config, StopMode::Fixed { loops }, &lens);
        });
        println!("fixed loops={loops}: {ms:.1} ms (B=2,T=64)");
    }
    let ms = ms_of(|| {
        let _ = model.forward(tokens.clone(), &config, StopMode::Act, &lens);
    });
    println!("act (max 8): {ms:.1} ms");
    let ms = ms_of(|| {
        let _ = model.forward(tokens.clone(), &config, StopMode::Converge, &lens);
    });
    println!("converge (max 8): {ms:.1} ms");

    let out = model.forward(tokens.clone(), &config, StopMode::Act, &lens);
    println!("act steps_used={} mean_halt={:.2}", out.steps_used, out.mean_halt);
    let ms = ms_of(|| {
        let _ = model
            .forward(tokens.clone(), &config, StopMode::Fixed { loops: 4 }, &lens)
            .logits
            .sum()
            .backward();
    });
    println!("backward (4 loops): {ms:.1} ms");
}
