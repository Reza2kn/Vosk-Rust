//! Pure-Rust Kaldi/Vosk **best-path** WFST decoder.
//!
//! Token-passing Viterbi over a Kaldi HCLG (loaded via `rustfst`), driven by per-frame acoustic
//! log-likelihoods. Emits the 1-best word sequence — which is all we need to guide the Shenava
//! CTC beam (Vosk's per-utterance words become hotwords). No lattices, no endpointing.
//!
//! HCLG arc `ilabel` = Kaldi transition-id (1-based); `olabel` = word id; `weight` = graph cost
//! (−log). Acoustic frames are indexed by pdf-id via `tid2pdf` (from the TransitionModel).

pub mod kaldi_io;
pub mod mfcc;
pub mod nnet3;
pub mod transition_model;

use rustfst::prelude::*;
use std::collections::HashMap;

/// Read `path`, transparently gunzipping `path.gz` if that exists instead (WFSTs are large).
fn read_maybe_gz(path: &str) -> std::io::Result<Vec<u8>> {
    let gz = format!("{path}.gz");
    if std::path::Path::new(&gz).exists() {
        let f = std::io::BufReader::new(std::fs::File::open(&gz)?);
        let mut dec = flate2::read::GzDecoder::new(f);
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut dec, &mut buf)?;
        Ok(buf)
    } else {
        std::fs::read(path)
    }
}

/// High-level pure-Rust Vosk recognizer: 16 kHz audio → words. Loads a standard Vosk model
/// directory (`am/final.mdl`, `graph/HCLG.fst`, `graph/words.txt`) and runs MFCC → nnet3 chain
/// forward → WFST best-path decode entirely in Rust (no libvosk / Kaldi / numpy).
pub struct Recognizer {
    net: nnet3::Nnet3,
    tm: transition_model::TransitionModel,
    fst: ConstFst<TropicalWeight>,
    words: Vec<String>,
    mfcc: mfcc::Mfcc,
    decoder: Decoder,
}

impl Recognizer {
    /// Load a Vosk model directory (static-HCLG "big" models).
    pub fn load(model_dir: &str) -> std::io::Result<Recognizer> {
        let mdl = format!("{model_dir}/am/final.mdl");
        let int4 = format!("{model_dir}/am/final.int4");
        // prefer the compact int4 AM package (weights + tid2pdf) if present; else raw final.mdl
        let (net, tm) = if std::path::Path::new(&int4).exists() {
            (nnet3::Nnet3::load_int4(&int4), transition_model::TransitionModel::load_int4(&int4)?)
        } else {
            (nnet3::Nnet3::load(&mdl), transition_model::TransitionModel::load(&mdl)?)
        };
        // graph + words load transparently from a .gz if present (the composed HCLG is large)
        let fst_bytes = read_maybe_gz(&format!("{model_dir}/graph/HCLG.fst"))?;
        let fst = ConstFst::<TropicalWeight>::load(&fst_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let wtxt = String::from_utf8(read_maybe_gz(&format!("{model_dir}/graph/words.txt"))?)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let mut words: Vec<String> = Vec::new();
        for line in wtxt.lines() {
            let mut it = line.split_whitespace();
            if let (Some(w), Some(i)) = (it.next(), it.next()) {
                let id: usize = i.parse().unwrap_or(0);
                if id >= words.len() {
                    words.resize(id + 1, String::new());
                }
                words[id] = w.to_string();
            }
        }
        // decode config from conf/model.conf (big: beam 13 / max-active 7000; small: 10 / 3000)
        let conf = std::fs::read_to_string(format!("{model_dir}/conf/model.conf")).unwrap_or_default();
        let getf = |key: &str, def: f32| -> f32 {
            for line in conf.lines() {
                if let Some(v) = line.trim().strip_prefix(&format!("--{key}=")) {
                    return v.trim().parse().unwrap_or(def);
                }
            }
            def
        };
        let beam = getf("beam", 13.0);
        let ascale = getf("acoustic-scale", 1.0);
        let max_active = getf("max-active", 7000.0) as usize;
        Ok(Recognizer {
            net,
            tm,
            fst,
            words,
            mfcc: mfcc::Mfcc::from_conf(model_dir, 16000.0),
            decoder: Decoder::new(beam, ascale).with_max_active(max_active),
        })
    }

    /// Recognize an utterance. `samples` = mono 16 kHz, normalized to roughly [-1, 1].
    pub fn recognize(&self, samples: &[f32]) -> String {
        let scaled: Vec<f32> = samples.iter().map(|x| x * 32768.0).collect();
        let feats = self.mfcc.compute(&scaled);
        let ivector = nnet3::Mat::new(feats.r, self.net.ivector_dim);
        let ll = self.net.forward(feats, ivector);
        let loglikes: Vec<Vec<f32>> =
            (0..ll.r).map(|i| ll.d[i * ll.c..(i + 1) * ll.c].to_vec()).collect();
        let ids = self.decoder.decode(&self.fst, &self.tm.tid2pdf, &loglikes);
        ids.iter()
            .filter_map(|&w| self.words.get(w as usize))
            .map(String::as_str)
            .filter(|w| !w.is_empty() && *w != "!SIL" && *w != "<eps>")
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Clone, Copy)]
struct Tok {
    cost: f32,
    back: i32,
    word: Label,
}

pub struct Decoder {
    beam: f32,
    acoustic_scale: f32,
    max_active: usize,
}

impl Decoder {
    pub fn new(beam: f32, acoustic_scale: f32) -> Self {
        Decoder { beam, acoustic_scale, max_active: 7000 }
    }

    pub fn with_max_active(mut self, max_active: usize) -> Self {
        self.max_active = max_active;
        self
    }

    /// Keep only the `max_active` lowest-cost states (Kaldi's --max-active).
    fn cap_active(&self, arena: &[Tok], active: &mut HashMap<StateId, usize>) {
        if active.len() <= self.max_active {
            return;
        }
        let mut costs: Vec<f32> = active.values().map(|&i| arena[i].cost).collect();
        let k = self.max_active;
        costs.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
        let thresh = costs[k];
        active.retain(|_, &mut i| arena[i].cost < thresh);
    }

    /// Decode `loglikes` (T frames × n_pdf) over `fst`. `tid2pdf[tid]` maps transition-id → pdf-id.
    /// Returns the 1-best word-id sequence (non-zero olabels).
    pub fn decode<F: ExpandedFst<TropicalWeight>>(
        &self,
        fst: &F,
        tid2pdf: &[i32],
        loglikes: &[Vec<f32>],
    ) -> Vec<Label> {
        let start = match fst.start() {
            Some(s) => s,
            None => return vec![],
        };
        let mut arena: Vec<Tok> = vec![Tok { cost: 0.0, back: -1, word: 0 }];
        let mut active: HashMap<StateId, usize> = HashMap::new();
        active.insert(start, 0);
        self.epsilon_closure(fst, &mut arena, &mut active);

        for frame in loglikes {
            let mut next: HashMap<StateId, usize> = HashMap::new();
            let mut best = f32::INFINITY;
            for (&s, &ti) in active.iter() {
                let base = arena[ti].cost;
                if let Ok(trs) = fst.get_trs(s) {
                    for tr in trs.trs() {
                        if tr.ilabel == 0 {
                            continue;
                        }
                        let pdf = tid2pdf[tr.ilabel as usize] as usize;
                        let cost = base + tr.weight.value() + (-frame[pdf] * self.acoustic_scale);
                        if cost < best {
                            best = cost;
                        }
                        relax(&mut arena, &mut next, tr.nextstate, cost, ti as i32, tr.olabel);
                    }
                }
            }
            let cutoff = best + self.beam;
            next.retain(|_, &mut ti| arena[ti].cost <= cutoff);
            active = next;
            self.epsilon_closure(fst, &mut arena, &mut active);
            self.cap_active(&arena, &mut active);
        }

        let mut best_ti: i32 = -1;
        let mut best_cost = f32::INFINITY;
        for (&s, &ti) in active.iter() {
            if let Ok(Some(fw)) = fst.final_weight(s) {
                let c = arena[ti].cost + fw.value();
                if c < best_cost {
                    best_cost = c;
                    best_ti = ti as i32;
                }
            }
        }
        if best_ti < 0 {
            for &ti in active.values() {
                if arena[ti].cost < best_cost {
                    best_cost = arena[ti].cost;
                    best_ti = ti as i32;
                }
            }
        }
        let mut words = Vec::new();
        let mut t = best_ti;
        while t >= 0 {
            let tok = arena[t as usize];
            if tok.word != 0 {
                words.push(tok.word);
            }
            t = tok.back;
        }
        words.reverse();
        words
    }

    /// Expand non-emitting (ilabel==0) arcs to a fixpoint, relaxing costs.
    fn epsilon_closure<F: ExpandedFst<TropicalWeight>>(
        &self,
        fst: &F,
        arena: &mut Vec<Tok>,
        active: &mut HashMap<StateId, usize>,
    ) {
        let mut queue: Vec<StateId> = active.keys().copied().collect();
        while let Some(s) = queue.pop() {
            let ti = match active.get(&s) {
                Some(&i) => i,
                None => continue,
            };
            let base = arena[ti].cost;
            let mut updates: Vec<(StateId, f32, i32, Label)> = Vec::new();
            if let Ok(trs) = fst.get_trs(s) {
                for tr in trs.trs() {
                    if tr.ilabel != 0 {
                        continue;
                    }
                    updates.push((tr.nextstate, base + tr.weight.value(), ti as i32, tr.olabel));
                }
            }
            for (ns, cost, back, word) in updates {
                if relax(arena, active, ns, cost, back, word) {
                    queue.push(ns);
                }
            }
        }
    }
}

fn relax(
    arena: &mut Vec<Tok>,
    map: &mut HashMap<StateId, usize>,
    state: StateId,
    cost: f32,
    back: i32,
    word: Label,
) -> bool {
    match map.get(&state) {
        Some(&idx) if arena[idx].cost <= cost => false,
        _ => {
            let idx = arena.len();
            arena.push(Tok { cost, back, word });
            map.insert(state, idx);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_two_word_path() {
        // 0 -(tid1 : word100)-> 1 -(tid2 : word200)-> 2(final)
        let mut fst = VectorFst::<TropicalWeight>::new();
        let s0 = fst.add_state();
        let s1 = fst.add_state();
        let s2 = fst.add_state();
        fst.set_start(s0).unwrap();
        fst.set_final(s2, TropicalWeight::one()).unwrap();
        fst.add_tr(s0, Tr::new(1, 100, TropicalWeight::one(), s1)).unwrap();
        fst.add_tr(s1, Tr::new(2, 200, TropicalWeight::one(), s2)).unwrap();
        let tid2pdf = vec![-1i32, 0, 1];
        let loglikes = vec![vec![0.0f32, -9.0], vec![-9.0f32, 0.0]];
        let dec = Decoder::new(15.0, 1.0);
        assert_eq!(dec.decode(&fst, &tid2pdf, &loglikes), vec![100, 200]);
    }
}
