//! Kaldi TransitionModel → `tid2pdf` for a **chain** model.
//!
//! Chain models store `<Tuples>` of `(phone, hmm_state, forward_pdf, self_loop_pdf)`. With the
//! standard chain topology (one emitting state, two transitions), each tuple is one
//! transition-state with exactly 2 transition-ids: `2*i+1` = self-loop (→ self_loop_pdf) and
//! `2*i+2` = forward (→ forward_pdf), numbered from 1. We reach the tuples by seeking the
//! `<Tuples>` token (the topology in between is pure binary with no tokens). The mapping is
//! consistency-checked against the HCLG's max input label.

use crate::kaldi_io::KaldiReader;
use std::io::Result;

pub struct TransitionModel {
    /// tid2pdf[transition_id] = pdf_id; index 0 unused.
    pub tid2pdf: Vec<i32>,
    pub num_pdfs: usize,
    pub num_transition_states: usize,
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

impl TransitionModel {
    pub fn load(path: &str) -> Result<TransitionModel> {
        let data = std::fs::read(path)?;
        let marker = b"<Tuples> ";
        let pos = find(&data, marker)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no <Tuples>"))?
            + marker.len();
        let mut kr = KaldiReader::new(&data[pos..]);
        let n = kr.read_i32()? as usize;
        let mut tid2pdf = vec![-1i32; 2 * n + 1];
        let mut num_pdfs = 0i32;
        for i in 0..n {
            let _phone = kr.read_i32()?;
            let _hmm_state = kr.read_i32()?;
            let fwd = kr.read_i32()?;
            let slf = kr.read_i32()?;
            num_pdfs = num_pdfs.max(fwd + 1).max(slf + 1);
            tid2pdf[2 * i + 1] = slf; // self-loop
            tid2pdf[2 * i + 2] = fwd; // forward
        }
        Ok(TransitionModel { tid2pdf, num_pdfs: num_pdfs as usize, num_transition_states: n })
    }

    /// Max valid transition-id (== number of transition-ids).
    pub fn num_tids(&self) -> usize {
        self.tid2pdf.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustfst::prelude::*;

    const MODEL: &str = "/Users/Ajab/AI/w2v-bert-2.0/vosk-model-fa-0.42/am/final.mdl";
    const HCLG: &str = "/Users/Ajab/AI/w2v-bert-2.0/vosk-model-fa-0.42/graph/HCLG.fst";

    #[test]
    fn tid2pdf_consistent_with_hclg() {
        let tm = match TransitionModel::load(MODEL) {
            Ok(t) => t,
            Err(_) => return, // model absent — skip
        };
        assert_eq!(tm.num_transition_states, 6775);
        assert_eq!(tm.num_tids(), 2 * 6775);
        println!("num_pdfs={} num_tids={}", tm.num_pdfs, tm.num_tids());
        // every tid maps to a valid pdf
        for &p in tm.tid2pdf.iter().skip(1) {
            assert!(p >= 0 && (p as usize) < tm.num_pdfs);
        }
        // the HCLG's input labels are transition-ids: max ilabel must fit our tid range.
        if let Ok(fst) = ConstFst::<TropicalWeight>::read(HCLG) {
            let mut max_ilabel = 0u32;
            for s in 0..fst.num_states() {
                if let Ok(trs) = fst.get_trs(s as StateId) {
                    for tr in trs.trs() {
                        if tr.ilabel > max_ilabel {
                            max_ilabel = tr.ilabel;
                        }
                    }
                }
            }
            println!("HCLG max ilabel = {max_ilabel}, our num_tids = {}", tm.num_tids());
            assert!(max_ilabel as usize <= tm.num_tids(), "HCLG references a tid beyond the model");
        }
    }
}
