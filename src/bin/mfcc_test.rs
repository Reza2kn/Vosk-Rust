//! Verify the Rust Kaldi MFCC vs the torchaudio oracle: read /tmp/vosk_wav16k.bin, ×32768,
//! compute MFCC, compare to /tmp/vosk_feats.bin.
use shenava_kaldi::mfcc::Mfcc;
use shenava_kaldi::nnet3::Mat;
use std::io::Read;

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
    let wav = load_bin("/tmp/vosk_wav16k.bin"); // [1, nsamples], normalized [-1,1]
    let samples: Vec<f32> = wav.d.iter().map(|x| x * 32768.0).collect();
    let reff = load_bin("/tmp/vosk_feats.bin");

    let mfcc = Mfcc::vosk(16000.0);
    let feats = mfcc.compute(&samples);
    println!("Rust MFCC {}x{}, ref {}x{}", feats.r, feats.c, reff.r, reff.c);

    let n = feats.r.min(reff.r) * feats.c;
    let mut maxabs = 0.0f32;
    let mut worst = 0usize;
    for i in 0..n {
        let e = (feats.d[i] - reff.d[i]).abs();
        if e > maxabs {
            maxabs = e;
            worst = i;
        }
    }
    println!("max|Δ| = {maxabs:.4e} at frame {} ceps {}", worst / feats.c, worst % feats.c);
    println!("row0 mine[:5]: {:?}", &feats.d[..5]);
    println!("row0 ref [:5]: {:?}", &reff.d[..5]);
    println!("{}", if maxabs < 5e-2 { "✅ MFCC matches oracle" } else { "❌ MFCC mismatch" });
}
