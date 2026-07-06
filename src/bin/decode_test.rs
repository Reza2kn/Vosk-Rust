//! 100% pure-Rust vosk: 16k waveform → Rust MFCC → Rust nnet3 → Rust WFST decode → words,
//! vs the vosk oracle. Reads /tmp/vosk_wav16k.bin (16k mono samples, normalized).
use rustfst::prelude::*;
use shenava_kaldi::mfcc::Mfcc;
use shenava_kaldi::nnet3::{Mat, Nnet3};
use shenava_kaldi::{transition_model::TransitionModel, Decoder};
use std::collections::HashMap;
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

    let tm = TransitionModel::load(&format!("{model}/am/final.mdl")).unwrap();
    let net = Nnet3::load(&format!("{model}/am/final.mdl"));

    // 16k waveform → Kaldi MFCC (int16-scaled), all in Rust
    let wav = load_bin("/tmp/vosk_wav16k.bin");
    let samples: Vec<f32> = wav.d.iter().map(|x| x * 32768.0).collect();
    let tm0 = Instant::now();
    let feats = Mfcc::vosk(16000.0).compute(&samples);
    println!("MFCC {}x{} in {:?}", feats.r, feats.c, tm0.elapsed());

    let t0 = Instant::now();
    let ivector = Mat::new(feats.r, 40);
    let ll = net.forward(feats, ivector);
    println!("nnet3 forward {}x{} in {:?}", ll.r, ll.c, t0.elapsed());
    // to Vec<Vec<f32>> for the decoder
    let loglikes: Vec<Vec<f32>> = (0..ll.r).map(|i| ll.d[i * ll.c..(i + 1) * ll.c].to_vec()).collect();

    let t1 = Instant::now();
    let fst = ConstFst::<TropicalWeight>::read(format!("{model}/graph/HCLG.fst")).unwrap();
    println!("HCLG {} states in {:?}", fst.num_states(), t1.elapsed());

    let t2 = Instant::now();
    let dec = Decoder::new(13.0, 1.0).with_max_active(7000);
    let words = dec.decode(&fst, &tm.tid2pdf, &loglikes);
    println!("decode in {:?}", t2.elapsed());

    let wtxt = std::fs::read_to_string(format!("{model}/graph/words.txt")).unwrap();
    let mut id2w: HashMap<u32, &str> = HashMap::new();
    for line in wtxt.lines() {
        let mut it = line.split_whitespace();
        if let (Some(w), Some(i)) = (it.next(), it.next()) {
            id2w.insert(i.parse().unwrap(), w);
        }
    }
    let out: Vec<&str> = words.iter()
        .map(|&w| *id2w.get(&(w as u32)).unwrap_or(&"?"))
        .filter(|w| *w != "!SIL" && *w != "<eps>")
        .collect();
    println!("\n[100% Rust nnet3 + decoder]");
    println!("DECODED: {}", out.join(" "));
    println!("ORACLE : من عرضه این کار رو ندارم");
}
