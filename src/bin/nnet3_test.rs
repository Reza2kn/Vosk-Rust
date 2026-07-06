//! Verify the Rust nnet3 forward against the numpy oracle: read /tmp/vosk_feats.bin, run forward,
//! compare loglikes to /tmp/vosk_loglikes.bin (max abs diff must be ~0).
use vosk_rust::nnet3::{Mat, Nnet3};
use std::io::Read;
use std::time::Instant;

fn load_bin(path: &str) -> Mat {
    let mut f = std::fs::File::open(path).unwrap();
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr).unwrap();
    let r = i32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let c = i32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let mut raw = vec![0u8; r * c * 4];
    f.read_exact(&mut raw).unwrap();
    let d = raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Mat { r, c, d }
}

fn main() {
    let model = "/Users/Ajab/AI/w2v-bert-2.0/vosk-model-fa-0.42";
    let feats = load_bin("/tmp/vosk_feats.bin");
    let refll = load_bin("/tmp/vosk_loglikes.bin");
    println!("feats {}x{}, ref loglikes {}x{}", feats.r, feats.c, refll.r, refll.c);

    let t0 = Instant::now();
    let net = Nnet3::load(&format!("{model}/am/final.mdl"));
    let ivector = Mat::new(feats.r, 40); // zeros
    let ll = net.forward(feats, ivector);
    println!("Rust forward: {}x{} in {:?}", ll.r, ll.c, t0.elapsed());

    assert_eq!((ll.r, ll.c), (refll.r, refll.c));
    let mut maxabs = 0.0f32;
    let mut sumsq = 0.0f64;
    for (a, b) in ll.d.iter().zip(&refll.d) {
        let e = (a - b).abs();
        maxabs = maxabs.max(e);
        sumsq += (e as f64) * (e as f64);
    }
    let rms = (sumsq / ll.d.len() as f64).sqrt();
    println!("vs numpy oracle: max|Δ| = {maxabs:.3e},  rms = {rms:.3e}");
    println!("{}", if maxabs < 1e-2 { "✅ RUST nnet3 MATCHES numpy oracle" } else { "❌ MISMATCH" });
}
