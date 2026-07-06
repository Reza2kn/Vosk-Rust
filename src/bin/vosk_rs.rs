//! Pure-Rust Vosk recognizer CLI. Args: <model_dir> <wav16k.bin>
//! (wav16k.bin = [i32 1][i32 n][f32 n] mono 16 kHz samples, normalized ~[-1,1]).
//! Works for both the big (static HCLG) and small (offline-composed HCLG) models via the same code.
use std::io::Read;
use std::time::Instant;
use vosk_rust::Recognizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = &args[1];
    let wav_path = &args[2];

    let mut f = std::fs::File::open(wav_path).unwrap();
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr).unwrap();
    let n = i32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw).unwrap();
    let samples: Vec<f32> =
        raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();

    let t0 = Instant::now();
    let rec = Recognizer::load(model_dir).unwrap();
    let t1 = Instant::now();
    let text = rec.recognize(&samples);
    println!("model: {model_dir}");
    println!("load {:?}, recognize {:?}", t1 - t0, t1.elapsed());
    println!("TEXT: {text}");
}
