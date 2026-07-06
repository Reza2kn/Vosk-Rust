//! Small-model decode: offline-composed static HCLG (const) + small AM loglikes → words.
//! Uses /tmp/small_HCLG.fst (fstcompose of HCLr∘Gr), /tmp/small_loglikes.bin (small nnet3 forward),
//! /tmp/small_words.txt (word table extracted from Gr.fst), tid2pdf from the small final.mdl.
use rustfst::prelude::*;
use vosk_rust::{transition_model::TransitionModel, Decoder};
use std::collections::HashMap;
use std::io::Read;
use std::time::Instant;

fn main() {
    let sm = "/Users/Ajab/conductor/workspaces/iotype-distill-smoke/models/vosk-model-small-fa-0.5";
    let tm = TransitionModel::load(&format!("{sm}/am/final.mdl")).unwrap();
    println!("small tid2pdf: {} tids, {} pdfs", tm.num_tids(), tm.num_pdfs);

    // loglikes
    let mut f = std::fs::File::open("/tmp/small_loglikes.bin").unwrap();
    let mut hdr = [0u8; 8];
    f.read_exact(&mut hdr).unwrap();
    let t = i32::from_le_bytes(hdr[0..4].try_into().unwrap()) as usize;
    let d = i32::from_le_bytes(hdr[4..8].try_into().unwrap()) as usize;
    let mut raw = vec![0u8; t * d * 4];
    f.read_exact(&mut raw).unwrap();
    let loglikes: Vec<Vec<f32>> = (0..t)
        .map(|i| (0..d).map(|j| f32::from_le_bytes(raw[(i * d + j) * 4..(i * d + j) * 4 + 4].try_into().unwrap())).collect())
        .collect();
    println!("loglikes {t}x{d}");

    let t0 = Instant::now();
    let fst = ConstFst::<TropicalWeight>::read("/tmp/small_HCLG_nodis.fst").unwrap();
    println!("HCLG {} states in {:?}", fst.num_states(), t0.elapsed());

    // guard: max ilabel must fit tid2pdf
    let mut maxi = 0u32;
    for s in 0..fst.num_states() {
        if let Ok(trs) = fst.get_trs(s as StateId) {
            for tr in trs.trs() {
                maxi = maxi.max(tr.ilabel);
            }
        }
    }
    println!("HCLG max ilabel {maxi} (tids {})", tm.num_tids());

    let t1 = Instant::now();
    let dec = Decoder::new(10.0, 1.0).with_max_active(3000); // small model.conf
    let words = dec.decode(&fst, &tm.tid2pdf, &loglikes);
    println!("decode in {:?}", t1.elapsed());

    let wtxt = std::fs::read_to_string("/tmp/small_words.txt").unwrap();
    let mut id2w: HashMap<u32, &str> = HashMap::new();
    for line in wtxt.lines() {
        let mut it = line.split_whitespace();
        if let (Some(w), Some(i)) = (it.next(), it.next()) {
            id2w.insert(i.parse().unwrap_or(0), w);
        }
    }
    let out: Vec<&str> = words.iter()
        .map(|&w| *id2w.get(&(w as u32)).unwrap_or(&"?"))
        .filter(|w| *w != "!SIL" && *w != "<eps>")
        .collect();
    println!("\nDECODED: {}", out.join(" "));
    println!("ORACLE : من عرضه این کار را ندارم");
}
