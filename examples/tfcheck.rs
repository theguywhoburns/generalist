//! Teacher-forced check: `cargo run --example tfcheck -- <manifest> <ckpt> [n]`
//! Forwards eval instances WITH targets (training-style, padded batch) and
//! reports argmax accuracy at scored positions. Distinguishes "weights fit
//! but free-running diverges" from "weights never fit these instances".
//! CPU only (loads any checkpoint, runs on NdArray).

use burn::backend::{Autodiff, NdArray, ndarray::NdArrayDevice};
use burn::module::AutodiffModule;
use generalist::{
    harness::{RunConfig, collate, generate},
    tasks::TaskRegistry,
    train::{Trainer, eval_split},
};

type B = Autodiff<NdArray>;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let manifest = args.get(1).map(|s| s.as_str()).unwrap_or("configs/stage0-smoke.json");
    let ckpt = args.get(2).map(|s| s.as_str()).unwrap_or("checkpoints/step001000.mpk");
    let n: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(12);
    let skip: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);

    let device = NdArrayDevice::default();
    let run = RunConfig::load_json(std::path::Path::new(manifest)).expect("manifest");
    let registry = TaskRegistry::builtin();
    let pool = generate(&run.experiment, &registry);
    // Beyond-eval-split pool instances: same generation, never evaluated.
    // Pass skip>=16 to leave the eval split behind.
    let eval_like: Vec<_> = pool
        .iter()
        .filter(|i| i.info.task == "parity")
        .skip(skip)
        .take(n)
        .cloned()
        .collect();
    let eval_set = if skip == 0 {
        eval_split(&pool, 16)
            .into_iter()
            .take(n)
            .collect::<Vec<_>>()
    } else {
        eval_like
    };
    let model = Trainer::<B>::load_checkpoint(&run.model, std::path::Path::new(ckpt), &device);

    let valid = model.valid();
    let mut correct = 0usize;
    let mut total = 0usize;
    for inst in eval_set.iter() {
        let col = collate(std::slice::from_ref(inst), &device);
        let out = valid.forward_fixed(col.tokens, &col.lengths, run.model.max_loops);
        // Argmax at scored positions only.
        let [_, t, v] = out.logits.dims();
        let flat = out.logits.reshape([t, v]).argmax(1);
        let preds: Vec<i64> = flat.into_data().as_slice::<i64>().unwrap().to_vec();
        let tgts: Vec<i64> = col.targets.into_data().as_slice::<i64>().unwrap().to_vec();
        let mask: Vec<f32> = col.loss_mask.into_data().as_slice::<f32>().unwrap().to_vec();
        let exp = String::from_utf8_lossy(&inst.target);
        let mut got = String::new();
        for i in 0..t {
            if mask[i] > 0.5 {
                total += 1;
                let b = preds[i];
                if b == tgts[i] {
                    correct += 1;
                }
                if (32..127).contains(&b) {
                    got.push(b as u8 as char);
                } else {
                    got.push_str(&format!("<{b}>"));
                }
            }
        }
        println!("task={} track={:?} expected={exp:?} greedy_tf={got:?}", inst.info.task, inst.info.track);
    }
    println!("teacher-forced scored accuracy: {correct}/{total} = {:.3}", correct as f64 / total.max(1) as f64);
}
