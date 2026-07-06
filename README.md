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

## Accuracy vs libvosk (honest)

Benchmarked against libvosk (vosk-python, i.e. the C++/Kaldi reference) on 6,669 hard
Persian clips (obstructed/DHH conditions), fair-normalized WER (punctuation-, digit-, and
ZWNJ/compound-folded):

| system | WER | CER |
|---|---|---|
| libvosk (Kaldi, real i-vectors) | **19.3%** | 6.25% |
| vosk-rust (this crate) | **22.0%** | 7.97% |

**The +2.7 WER gap is one thing: i-vectors.** libvosk feeds the acoustic model an online
i-vector (speaker/channel adaptation); vosk-rust currently feeds **zeros**. Proof it's the whole
gap: feeding libvosk's *actual* i-vectors into vosk-rust's acoustic model recovers **19.8% WER —
parity**. The acoustic forward, MFCC, `tid2pdf`, and WFST decode are each verified bit-close to
Kaldi (MFCC/gselect/batch-i-vector match). The zero-i-vector default matters most on noisy audio;
on clean speech the two are near-identical.

Online i-vector extraction is implemented and verified against Kaldi's **batch** extractor
(`ivector-extract`, corr 0.999), but Kaldi's **online** variant (`ivector-extract-online2`) has an
extraction-order behavior that even Kaldi's own batch tool doesn't reproduce, so it is not yet
bit-faithful and is left disabled by default. For a keyword/hotword **guide** role (the intended use
here), the zero-i-vector gap is on function words and does not materially change which keywords are
emitted.

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
- ✅ **Small model** (`HCLr ∘ Gr` lookahead graphs) — working. The one-time offline graph prep
  (`scripts/prep_small_graph.sh`, needs `brew install openfst`) composes the lookahead graphs into a
  static `const` HCLG; the runtime then loads it in pure Rust exactly like the big model. Decodes the
  reference clip identically to vosk (small AM = 20-dim MFCC, ivector-30, different topology — all
  handled by the same generic code). On the reference clip: `recognize` 115 ms (6× faster than big).
- ⚠️ **int4 weight quantization — implemented but not recommended.** `bin/quantize <model_dir>` writes
  `am/final.int4` (weight matrices → int4 + per-group scales, tid2pdf embedded) and `Recognizer::load`
  auto-detects it (6.2× smaller AM). It is **bit-identical on easy/clean clips but degrades badly at
  scale** — on 400 hard clips, int4 roughly **doubles WER** (big 11.9→22.4, small also +7.7). These
  Kaldi chain models are weight-precision-sensitive (unlike a FastConformer, which tolerates int4), and
  the AM isn't the footprint bottleneck anyway (the graph dominates). **Ship f32** (leave `final.mdl`;
  don't place `final.int4`). The quantizer is kept only for size-over-accuracy experiments.
- ✅ **Fast matmul** — on macOS the nnet3 matmuls run through **Apple Accelerate (`cblas_sgemm`, the
  AMX coprocessor)** — GPU-class ~1000 GFLOP/s, forward 322 → 103 ms, no GPU dependency; other targets
  use threaded `matrixmultiply` (`MATMUL_NUM_THREADS=4`).
- ❌ **GPU (candle/Metal)** — investigated and **deliberately skipped**: the WFST decode is CPU-only and
  dominates the per-utterance pipeline (Amdahl), a one-shot guide pays Metal warmup every session, and
  Accelerate/AMX already matches realistic GPU latency with zero deps. GPU would only pay off for
  *batch/offline* transcription — not the live guide.
- ℹ️ i-vectors are fed as zeros; sufficient for clean-audio guide quality.

## Footprint (small model, on-device)

| artifact | raw | shipped |
|---|---|---|
| `graph/HCLG.fst` (composed) | 371 MB | **146 MB** (`.gz`, loaded transparently) |
| `graph/words.txt` | 8.6 MB | 2.2 MB (`.gz`) |
| `am/final.int4` (int4 AM) | 19 MB (f32) | **3.8 MB** |

The small model's front end differs (20 mel/ceps, ivector-30, `lda` splice vs `idct`/delta) — all
read from the model's own `conf/` so one `Recognizer::load` handles both. The offline compose grows
the graph on disk (~110 MB lookahead → ~350 MB static `const`); that's the tradeoff for a pure-Rust
runtime with no `libvosk` lookahead machinery.

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
