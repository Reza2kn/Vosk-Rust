# Vosk-Rust

**A pure-Rust reimplementation of Vosk / Kaldi nnet3 chain ASR decoding.** No `libvosk`, no
Kaldi, no C++, no Python — a standard Vosk model directory goes in, words come out.

```rust
use vosk_rust::Recognizer;

let rec = Recognizer::load("vosk-model-fa-0.42")?;   // standard Vosk model dir
let words = rec.recognize(&samples_16k);              // mono f32, ~[-1, 1]
println!("{words}");                                 // «من عرضه این کار رو ندارم»
```

It reproduces vosk-python **exactly** on the reference clip (`test/test.wav` → identical text).

## Why

Vosk is Kaldi under the hood, which means a C++ dependency (`libvosk`) that is painful to ship
on-device and cross-compile. This crate reimplements the whole inference path in safe-ish Rust so
the acoustic model + decoder can run anywhere Rust runs — as an on-device ASR **guide**, or as a
teacher model outside an app. Built for the [Shenava](https://github.com/Reza2kn/shenava-ctc-beam)
keyword-robust Persian ASR ensemble, but the code is model-agnostic (any Kaldi nnet3 **chain**
TDNN-F model with a static HCLG).

## What's inside (each independently verified)

| stage | module | verification |
|---|---|---|
| Kaldi binary reader (`\0B` format) | `kaldi_io.rs` | parses the real `final.mdl` header |
| TransitionModel → `tid2pdf` | `transition_model.rs` | consistent with HCLG (max ilabel ≤ #tids; #pdfs cross-checks nnet3 output) |
| Kaldi MFCC (40 ceps, povey, preemph, DCT, lifter) | `mfcc.rs` | max\|Δ\|≈1e-3 vs `torchaudio.compliance.kaldi` |
| nnet3 chain forward (TDNN-F, 3-stream, 248 comps) | `nnet3.rs` | max\|Δ\|≈7e-6 vs a numpy reference |
| Token-passing Viterbi WFST decode over HCLG | `lib.rs` | full pipeline == vosk oracle |

The acoustic model is executed straight from `am/final.mdl`: the descriptor graph is walked over
whole-utterance matrices, identity components (dropout/no-op/spec-augment) pass through, and the
xent training branch is skipped. Matmuls use [`matrixmultiply`](https://crates.io/crates/matrixmultiply)
(pure-Rust SIMD SGEMM); the FFT uses [`rustfft`](https://crates.io/crates/rustfft); FSTs load via
[`rustfst`](https://crates.io/crates/rustfst).

## Performance

On the 5.45 s reference clip (Apple M2, release):

```
MFCC            4 ms
nnet3 forward 320 ms      (RTF ≈ 0.06)
WFST decode   356 ms
HCLG load     485 ms      (one-time, 10.7 M states / 698 MB)
```

## Status

- ✅ **Big model** (static `HCLG.fst`) — fully working, verified.
- 🚧 Small models (`HCLr ∘ Gr` lookahead composition) — planned.
- 🚧 int4 weight quantization + GPU (candle/wgpu) acoustic backend — planned.
- ℹ️ i-vectors are fed as zeros; sufficient for clean-audio guide quality.

## Layout

```
src/lib.rs                Recognizer + Viterbi WFST decoder
src/mfcc.rs               Kaldi-compatible MFCC
src/nnet3.rs              nnet3 chain forward
src/transition_model.rs   tid → pdf
src/kaldi_io.rs           Kaldi binary reader
tools/nnet3_ref.py        numpy reference (the layer-wise oracle)
src/bin/*.rs              decode_test / nnet3_test / mfcc_test verifiers
```

## License

Apache-2.0.
