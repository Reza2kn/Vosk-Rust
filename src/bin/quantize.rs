//! int4-quantize a Vosk AM: `quantize <model_dir>` writes <model_dir>/am/final.int4
//! (weight matrices → int4 + per-group scales; biases/batchnorm/tid2pdf kept). Ship the .int4
//! instead of am/final.mdl. Recognizer::load auto-detects it.
use vosk_rust::nnet3::{Nnet3, GROUP};

fn main() {
    let dir = std::env::args().nth(1).expect("usage: quantize <model_dir> [group]");
    let group = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(GROUP);
    let mdl = format!("{dir}/am/final.mdl");
    let out = format!("{dir}/am/final.int4");
    Nnet3::quantize_model(&mdl, &out, group);
    println!("group={group}");
    let a = std::fs::metadata(&mdl).map(|m| m.len()).unwrap_or(0);
    let b = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!("{mdl} {:.1} MB  ->  {out} {:.1} MB  ({:.1}x)",
             a as f64 / 1e6, b as f64 / 1e6, a as f64 / b as f64);
}
