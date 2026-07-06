// Probe: can rustfst load the Kaldi HCLG ConstFst?
use rustfst::prelude::*;
fn main() {
    let path = std::env::args().nth(1).expect("fst path");
    match ConstFst::<TropicalWeight>::read(&path) {
        Ok(f) => println!("LOADED ConstFst: {} states, start={:?}", f.num_states(), f.start()),
        Err(e) => println!("ConstFst read err: {e}\n trying VectorFst…"),
    }
}
