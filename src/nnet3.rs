//! Pure-Rust forward pass for the Vosk chain TDNN-F nnet3 (fa-0.42).
//!
//! Faithful port of `tools/nnet3_ref.py` (the numpy reference that exact-matches vosk). Parses
//! `am/final.mdl` directly, executes the descriptor graph over whole-utterance [T, dim] matrices,
//! and returns chain-output log-likelihoods (frame-subsampled by 3). Identity components
//! (Dropout/NoOp/SpecAugment) pass through; the xent branch is never evaluated.

use crate::kaldi_io::KaldiReader;
use std::collections::HashMap;

// Apple Accelerate BLAS — the AMX matrix coprocessor gives GPU-class SGEMM (~1000 GFLOP/s) with no
// GPU dependency, warmup, or host↔device copies. Used for the nnet3 matmuls on macOS; other targets
// fall back to the threaded pure-Rust `matrixmultiply`.
#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transa: i32, transb: i32, m: i32, n: i32, k: i32, alpha: f32,
        a: *const f32, lda: i32, b: *const f32, ldb: i32, beta: f32, c: *mut f32, ldc: i32,
    );
}

/// Row-major dense matrix [rows × cols].
#[derive(Clone)]
pub struct Mat {
    pub r: usize,
    pub c: usize,
    pub d: Vec<f32>,
}

impl Mat {
    pub fn new(r: usize, c: usize) -> Self {
        Mat { r, c, d: vec![0.0; r * c] }
    }
    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self.d[i * self.c..(i + 1) * self.c]
    }
    /// y[T, out] = self[T, in] · w[out, in]ᵀ (+ bias), via pure-Rust SGEMM.
    fn affine(&self, w: &Mat, bias: Option<&[f32]>) -> Mat {
        assert_eq!(self.c, w.c, "affine dim mismatch");
        let (t, inn, out) = (self.r, self.c, w.r);
        let mut y = Mat::new(t, out);
        // C[T,out] = A[T,in] · Wᵀ.  A row-major; W row-major [out,in] so op(W)=Wᵀ (TransB).
        #[cfg(target_os = "macos")]
        unsafe {
            cblas_sgemm(
                101, 111, 112, // CblasRowMajor, CblasNoTrans, CblasTrans
                t as i32, out as i32, inn as i32, 1.0,
                self.d.as_ptr(), inn as i32,
                w.d.as_ptr(), inn as i32,
                0.0, y.d.as_mut_ptr(), out as i32,
            );
        }
        #[cfg(not(target_os = "macos"))]
        unsafe {
            matrixmultiply::sgemm(
                t, inn, out, 1.0,
                self.d.as_ptr(), inn as isize, 1,   // A: rs=inn, cs=1
                w.d.as_ptr(), 1, inn as isize,        // B = wᵀ: rs=1, cs=inn
                0.0,
                y.d.as_mut_ptr(), out as isize, 1,    // C: rs=out, cs=1
            );
        }
        if let Some(b) = bias {
            for i in 0..t {
                let yr = &mut y.d[i * out..(i + 1) * out];
                for o in 0..out {
                    yr[o] += b[o];
                }
            }
        }
        y
    }
}

pub enum Comp {
    Affine { w: Mat, b: Vec<f32> },
    Linear { w: Mat },
    Tdnn { offsets: Vec<i32>, w: Mat, b: Vec<f32> },
    BatchNorm { mean: Vec<f32>, var: Vec<f32>, eps: f32, trms: f32 },
    Relu,
    Identity,
}

pub enum Desc {
    Ref(String),
    Offset(Box<Desc>, i32),
    Scale(f32, Box<Desc>),
    Sum(Box<Desc>, Box<Desc>),
    Append(Vec<Desc>),
    ReplaceIndex(Box<Desc>),
}

enum Node {
    Input,
    DimRange { src: String, off: usize, dim: usize },
    Component { comp: String, desc: Desc },
    Output { desc: Desc },
}

pub struct Nnet3 {
    comps: HashMap<String, Comp>,
    nodes: HashMap<String, Node>,
    /// dim of the `ivector` input node (40 big / 30 small); 0 if none.
    pub ivector_dim: usize,
}

// ---- binary component parsing helpers ----
fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    hay[from..].windows(needle.len()).position(|w| w == needle).map(|p| p + from)
}

fn mat_at(buf: &[u8], pos: usize) -> Mat {
    let (r, c, d) = KaldiReader::new(&buf[pos..]).read_float_matrix().unwrap();
    Mat { r, c, d }
}
fn vec_at(buf: &[u8], pos: usize) -> Vec<f32> {
    KaldiReader::new(&buf[pos..]).read_float_vec().unwrap()
}
/// find `tok` inside [s,e); return matrix/vector read right after it.
fn named_mat(buf: &[u8], s: usize, e: usize, tok: &[u8]) -> Option<Mat> {
    find(&buf[s..e], tok, 0).map(|off| mat_at(buf, s + off + tok.len()))
}
fn named_vec(buf: &[u8], s: usize, e: usize, tok: &[u8]) -> Option<Vec<f32>> {
    find(&buf[s..e], tok, 0).map(|off| vec_at(buf, s + off + tok.len()))
}
fn named_i32(buf: &[u8], s: usize, e: usize, tok: &[u8]) -> Option<i32> {
    find(&buf[s..e], tok, 0).map(|off| KaldiReader::new(&buf[s + off + tok.len()..]).read_i32().unwrap())
}
fn named_f32(buf: &[u8], s: usize, e: usize, tok: &[u8]) -> Option<f32> {
    find(&buf[s..e], tok, 0).map(|off| KaldiReader::new(&buf[s + off + tok.len()..]).read_f32().unwrap())
}

impl Nnet3 {
    pub fn load(path: &str) -> Nnet3 {
        let buf = std::fs::read(path).unwrap();
        let ivector_dim = ivector_dim_of(&buf);
        Nnet3 { comps: parse_components(&buf), nodes: parse_graph(&buf), ivector_dim }
    }

    /// feats [T,40], ivector [T,40] → chain loglikes [ceil(T/3), 3736].
    pub fn forward(&self, feats: Mat, ivector: Mat) -> Mat {
        let t = feats.r;
        let mut cache: HashMap<String, Mat> = HashMap::new();
        cache.insert("input".into(), feats);
        cache.insert("ivector".into(), ivector);
        let out = self.eval_node("output", &mut cache, t);
        // subsample by 3
        let cols = out.c;
        let rows = (out.r + 2) / 3;
        let mut y = Mat::new(rows, cols);
        for (j, i) in (0..out.r).step_by(3).enumerate() {
            y.d[j * cols..(j + 1) * cols].copy_from_slice(out.row(i));
        }
        y
    }

    fn eval_node(&self, name: &str, cache: &mut HashMap<String, Mat>, t: usize) -> Mat {
        if let Some(m) = cache.get(name) {
            return m.clone();
        }
        let r = match self.nodes.get(name).unwrap_or_else(|| panic!("no node {name}")) {
            Node::Input => panic!("input {name} not pre-populated"),
            Node::DimRange { src, off, dim } => {
                let s = self.eval_node(src, cache, t);
                let mut m = Mat::new(s.r, *dim);
                for i in 0..s.r {
                    m.d[i * dim..(i + 1) * dim].copy_from_slice(&s.row(i)[*off..off + dim]);
                }
                m
            }
            Node::Component { comp, desc } => {
                let x = self.eval_desc(desc, cache, t);
                self.apply(comp, x)
            }
            Node::Output { desc } => self.eval_desc(desc, cache, t),
        };
        cache.insert(name.into(), r.clone());
        r
    }

    fn eval_desc(&self, d: &Desc, cache: &mut HashMap<String, Mat>, t: usize) -> Mat {
        match d {
            Desc::Ref(n) => self.eval_node(n, cache, t),
            Desc::Offset(a, n) => clamp_offset(&self.eval_desc(a, cache, t), *n),
            Desc::Scale(s, a) => {
                let mut m = self.eval_desc(a, cache, t);
                m.d.iter_mut().for_each(|v| *v *= *s);
                m
            }
            Desc::Sum(a, b) => {
                let mut m = self.eval_desc(a, cache, t);
                let n = self.eval_desc(b, cache, t);
                m.d.iter_mut().zip(&n.d).for_each(|(x, y)| *x += *y);
                m
            }
            Desc::Append(parts) => {
                let mats: Vec<Mat> = parts.iter().map(|p| self.eval_desc(p, cache, t)).collect();
                let rows = mats[0].r;
                let cols: usize = mats.iter().map(|m| m.c).sum();
                let mut out = Mat::new(rows, cols);
                for i in 0..rows {
                    let mut off = 0;
                    for m in &mats {
                        out.d[i * cols + off..i * cols + off + m.c].copy_from_slice(m.row(i));
                        off += m.c;
                    }
                }
                out
            }
            Desc::ReplaceIndex(a) => {
                let m = self.eval_desc(a, cache, t);
                let mut out = Mat::new(t, m.c);
                for i in 0..t {
                    out.d[i * m.c..(i + 1) * m.c].copy_from_slice(m.row(0));
                }
                out
            }
        }
    }

    fn apply(&self, comp: &str, x: Mat) -> Mat {
        match self.comps.get(comp).unwrap_or_else(|| panic!("no comp {comp}")) {
            Comp::Identity => x,
            Comp::Relu => {
                let mut m = x;
                m.d.iter_mut().for_each(|v| *v = v.max(0.0));
                m
            }
            Comp::BatchNorm { mean, var, eps, trms } => {
                let mut m = x;
                let scale: Vec<f32> = var.iter().map(|v| trms / (v + eps).sqrt()).collect();
                for i in 0..m.r {
                    let row = &mut m.d[i * m.c..(i + 1) * m.c];
                    for j in 0..m.c {
                        row[j] = (row[j] - mean[j]) * scale[j];
                    }
                }
                m
            }
            Comp::Affine { w, b } => x.affine(w, Some(b)),
            Comp::Linear { w } => x.affine(w, None),
            Comp::Tdnn { offsets, w, b } => {
                // splice: concat x shifted by each offset → [T, in*len]
                let inn = x.c;
                let sp = offsets.len();
                let mut spliced = Mat::new(x.r, inn * sp);
                for (k, &o) in offsets.iter().enumerate() {
                    let shifted = clamp_offset(&x, o);
                    for i in 0..x.r {
                        spliced.d[i * inn * sp + k * inn..i * inn * sp + k * inn + inn]
                            .copy_from_slice(shifted.row(i));
                    }
                }
                spliced.affine(w, if b.is_empty() { None } else { Some(b) })
            }
        }
    }
}

fn clamp_offset(m: &Mat, n: i32) -> Mat {
    let mut out = Mat::new(m.r, m.c);
    for i in 0..m.r {
        let src = (i as i32 + n).clamp(0, m.r as i32 - 1) as usize;
        out.d[i * m.c..(i + 1) * m.c].copy_from_slice(m.row(src));
    }
    out
}

fn parse_components(buf: &[u8]) -> HashMap<String, Comp> {
    let tag = b"<ComponentName> ";
    let mut starts = vec![];
    let mut p = 0;
    while let Some(i) = find(buf, tag, p) {
        starts.push(i);
        p = i + tag.len();
    }
    starts.push(buf.len());
    let mut comps = HashMap::new();
    for k in 0..starts.len() - 1 {
        let (s, e) = (starts[k], starts[k + 1]);
        let mut r = KaldiReader::new(&buf[s + tag.len()..]);
        let name = r.read_token().unwrap();
        let ctype = r.read_token().unwrap();
        let comp = match ctype.as_str() {
            "<NaturalGradientAffineComponent>" | "<FixedAffineComponent>" => Comp::Affine {
                w: named_mat(buf, s, e, b"<LinearParams> ").unwrap(),
                b: named_vec(buf, s, e, b"<BiasParams> ").unwrap_or_default(),
            },
            "<LinearComponent>" => Comp::Linear { w: named_mat(buf, s, e, b"<Params> ").unwrap() },
            "<TdnnComponent>" => {
                let off = find(&buf[s..e], b"<TimeOffsets> ", 0).unwrap() + s + b"<TimeOffsets> ".len();
                let mut rr = KaldiReader::new(&buf[off..]);
                let offsets = rr.read_i32_vec().unwrap();
                Comp::Tdnn {
                    offsets,
                    w: named_mat(buf, s, e, b"<LinearParams> ").unwrap(),
                    b: named_vec(buf, s, e, b"<BiasParams> ").unwrap_or_default(),
                }
            }
            "<BatchNormComponent>" => Comp::BatchNorm {
                mean: named_vec(buf, s, e, b"<StatsMean> ").unwrap(),
                var: named_vec(buf, s, e, b"<StatsVar> ").unwrap(),
                eps: named_f32(buf, s, e, b"<Epsilon> ").unwrap_or(1e-3),
                trms: named_f32(buf, s, e, b"<TargetRms> ").unwrap_or(1.0),
            },
            "<RectifiedLinearComponent>" => Comp::Relu,
            _ => Comp::Identity, // NoOp / Dropout / SpecAugment / LogSoftmax(unused)
        };
        // silence the unused i32 helper warning while keeping it available
        let _ = named_i32;
        comps.insert(name, comp);
    }
    comps
}

fn kv<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!(" {key}=");
    let l = format!(" {line}");
    let i = l.find(&pat)? + pat.len();
    let rest = &line[i - 1..];
    Some(rest.split_whitespace().next().unwrap())
}

fn ivector_dim_of(buf: &[u8]) -> usize {
    let a = find(buf, b"<Nnet3>", 0).unwrap();
    let b = find(buf, b"<NumComponents>", 0).unwrap();
    for line in String::from_utf8_lossy(&buf[a..b]).lines() {
        let line = line.trim();
        if line.starts_with("input-node") && line.contains("name=ivector") {
            if let Some(d) = kv(line.strip_prefix("input-node").unwrap(), "dim") {
                return d.parse().unwrap_or(0);
            }
        }
    }
    0
}

fn parse_graph(buf: &[u8]) -> HashMap<String, Node> {
    let a = find(buf, b"<Nnet3>", 0).unwrap();
    let b = find(buf, b"<NumComponents>", 0).unwrap();
    parse_graph_text(&String::from_utf8_lossy(&buf[a..b]))
}

fn parse_graph_text(txt: &str) -> HashMap<String, Node> {
    let mut nodes = HashMap::new();
    for line in txt.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("input-node") {
            let name = kv(rest, "name").unwrap().to_string();
            nodes.insert(name, Node::Input);
        } else if let Some(rest) = line.strip_prefix("dim-range-node") {
            let name = kv(rest, "name").unwrap().to_string();
            nodes.insert(name, Node::DimRange {
                src: kv(rest, "input-node").unwrap().to_string(),
                off: kv(rest, "dim-offset").unwrap().parse().unwrap(),
                dim: kv(rest, "dim").unwrap().parse().unwrap(),
            });
        } else if let Some(rest) = line.strip_prefix("component-node") {
            let name = kv(rest, "name").unwrap().to_string();
            let comp = kv(rest, "component").unwrap().to_string();
            let desc = &line[line.find("input=").unwrap() + 6..];
            nodes.insert(name, Node::Component { comp, desc: parse_desc(desc.trim()) });
        } else if let Some(rest) = line.strip_prefix("output-node") {
            let name = kv(rest, "name").unwrap().to_string();
            let di = line.find("input=").unwrap() + 6;
            let desc = &line[di..line[di..].find(" objective=").map(|x| x + di).unwrap_or(line.len())];
            nodes.insert(name, Node::Output { desc: parse_desc(desc.trim()) });
        }
    }
    nodes
}

// ---- descriptor expression parser ----
fn parse_desc(s: &str) -> Desc {
    let toks = tokenize(s);
    let mut pos = 0;
    parse_expr(&toks, &mut pos)
}
fn tokenize(s: &str) -> Vec<String> {
    let mut out = vec![];
    let mut cur = String::new();
    for ch in s.chars() {
        if ch == '(' || ch == ')' || ch == ',' {
            if !cur.trim().is_empty() {
                out.push(cur.trim().to_string());
            }
            cur.clear();
            out.push(ch.to_string());
        } else {
            cur.push(ch);
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}
fn parse_expr(toks: &[String], pos: &mut usize) -> Desc {
    let head = toks[*pos].clone();
    *pos += 1;
    let is_op = matches!(head.as_str(),
        "Offset" | "Scale" | "Sum" | "Append" | "ReplaceIndex" | "Round" | "IfDefined");
    if is_op && *pos < toks.len() && toks[*pos] == "(" {
        *pos += 1; // (
        let mut args = vec![parse_expr(toks, pos)];
        while toks[*pos] == "," {
            *pos += 1;
            args.push(parse_expr(toks, pos));
        }
        assert_eq!(toks[*pos], ")");
        *pos += 1;
        return build_op(&head, args, toks, pos);
    }
    Desc::Ref(head)
}
fn build_op(op: &str, mut args: Vec<Desc>, _t: &[String], _p: &mut usize) -> Desc {
    match op {
        "Offset" => {
            // args: [desc, Ref(n)]
            let n = leaf_int(&args[1]);
            Desc::Offset(Box::new(args.remove(0)), n)
        }
        "Scale" => {
            let s = leaf_f32(&args[0]);
            Desc::Scale(s, Box::new(args.remove(1)))
        }
        "Sum" => {
            let b = args.remove(1);
            let a = args.remove(0);
            Desc::Sum(Box::new(a), Box::new(b))
        }
        "Append" => Desc::Append(args),
        "ReplaceIndex" => Desc::ReplaceIndex(Box::new(args.remove(0))), // (x, t, 0) → index 0
        "IfDefined" => args.remove(0),
        _ => panic!("unhandled desc op {op}"),
    }
}
fn leaf_int(d: &Desc) -> i32 {
    if let Desc::Ref(s) = d {
        s.parse().unwrap()
    } else {
        panic!("expected int leaf")
    }
}
fn leaf_f32(d: &Desc) -> f32 {
    if let Desc::Ref(s) = d {
        s.parse().unwrap()
    } else {
        panic!("expected f32 leaf")
    }
}

// ================= int4 weight quantization =================
// Per-row, group-wise symmetric int4. Each group of `GROUP` weights in a row shares one f32 scale
// (absmax/7); weights quantize to signed 4-bit [-7,7] packed 2/byte. ~6.4× smaller than f32.
// The `model.int4` file stores the graph text + every component (weight matrices quantized; biases,
// batchnorm stats, offsets kept f32). `load_int4` dequantizes back to f32 — the forward is unchanged.

pub const GROUP: usize = 32;

fn quantize_mat(m: &Mat, group: usize) -> (Vec<f32>, Vec<u8>) {
    let (r, c) = (m.r, m.c);
    let gpr = (c + group - 1) / group;
    let mut scales = vec![0f32; r * gpr];
    let mut packed = vec![0u8; (r * c + 1) / 2];
    for i in 0..r {
        for g in 0..gpr {
            let (s, e) = (g * group, ((g + 1) * group).min(c));
            let amax = (s..e).map(|k| m.d[i * c + k].abs()).fold(0f32, f32::max);
            let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            scales[i * gpr + g] = scale;
            for k in s..e {
                let q = ((m.d[i * c + k] / scale).round().clamp(-7.0, 7.0)) as i8;
                let idx = i * c + k;
                let nib = (q as u8) & 0x0F;
                if idx % 2 == 0 { packed[idx / 2] |= nib } else { packed[idx / 2] |= nib << 4 }
            }
        }
    }
    (scales, packed)
}

fn dequantize_mat(r: usize, c: usize, group: usize, scales: &[f32], packed: &[u8]) -> Mat {
    let gpr = (c + group - 1) / group;
    let mut m = Mat::new(r, c);
    for idx in 0..r * c {
        let byte = packed[idx / 2];
        let nib = if idx % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let q = ((nib as i8) << 4) >> 4; // sign-extend 4-bit
        let (i, k) = (idx / c, idx % c);
        m.d[idx] = q as f32 * scales[i * gpr + k / group];
    }
    m
}

fn wu32(o: &mut Vec<u8>, v: u32) { o.extend_from_slice(&v.to_le_bytes()) }
fn wf32(o: &mut Vec<u8>, v: f32) { o.extend_from_slice(&v.to_le_bytes()) }
fn wvec(o: &mut Vec<u8>, v: &[f32]) { wu32(o, v.len() as u32); for &x in v { wf32(o, x) } }
fn wmat(o: &mut Vec<u8>, m: &Mat, group: usize) {
    wu32(o, m.r as u32);
    wu32(o, m.c as u32);
    let (scales, packed) = quantize_mat(m, group);
    for &s in &scales { wf32(o, s) }
    o.extend_from_slice(&packed);
}

struct Cur<'a> { b: &'a [u8], p: usize }
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] { let s = &self.b[self.p..self.p + n]; self.p += n; s }
    fn u32(&mut self) -> u32 { u32::from_le_bytes(self.take(4).try_into().unwrap()) }
    fn u16(&mut self) -> u16 { u16::from_le_bytes(self.take(2).try_into().unwrap()) }
    fn f32(&mut self) -> f32 { f32::from_le_bytes(self.take(4).try_into().unwrap()) }
    fn vec(&mut self) -> Vec<f32> { let n = self.u32() as usize; (0..n).map(|_| self.f32()).collect() }
    fn mat(&mut self, group: usize) -> Mat {
        let (r, c) = (self.u32() as usize, self.u32() as usize);
        let gpr = (c + group - 1) / group;
        let scales: Vec<f32> = (0..r * gpr).map(|_| self.f32()).collect();
        let packed = self.take((r * c + 1) / 2);
        dequantize_mat(r, c, group, &scales, packed)
    }
}

impl Nnet3 {
    /// Quantize `final.mdl` → a compact `model.int4` (weight matrices to int4, rest f32).
    pub fn quantize_model(mdl: &str, out: &str, group: usize) {
        let buf = std::fs::read(mdl).unwrap();
        let comps = parse_components(&buf);
        let a = find(&buf, b"<Nnet3>", 0).unwrap();
        let b = find(&buf, b"<NumComponents>", 0).unwrap();
        let graph = &buf[a..b];
        let ivd = ivector_dim_of(&buf);
        let tm = crate::transition_model::TransitionModel::load(mdl).unwrap();
        let mut o = Vec::new();
        o.extend_from_slice(b"VRQ4");
        wu32(&mut o, group as u32);
        wu32(&mut o, ivd as u32);
        wu32(&mut o, tm.num_pdfs as u32);
        wu32(&mut o, tm.tid2pdf.len() as u32);
        for &t in &tm.tid2pdf { o.extend_from_slice(&t.to_le_bytes()) }
        wu32(&mut o, graph.len() as u32);
        o.extend_from_slice(graph);
        wu32(&mut o, comps.len() as u32);
        for (name, comp) in &comps {
            o.extend_from_slice(&(name.len() as u16).to_le_bytes());
            o.extend_from_slice(name.as_bytes());
            match comp {
                Comp::Affine { w, b } => { o.push(0); wmat(&mut o, w, group); wvec(&mut o, b) }
                Comp::Linear { w } => { o.push(1); wmat(&mut o, w, group) }
                Comp::Tdnn { offsets, w, b } => {
                    o.push(2);
                    o.extend_from_slice(&(offsets.len() as u16).to_le_bytes());
                    for &of in offsets { o.extend_from_slice(&of.to_le_bytes()) }
                    wmat(&mut o, w, group);
                    wvec(&mut o, b);
                }
                Comp::BatchNorm { mean, var, eps, trms } => {
                    o.push(3);
                    wvec(&mut o, mean);
                    wvec(&mut o, var);
                    wf32(&mut o, *eps);
                    wf32(&mut o, *trms);
                }
                Comp::Relu => o.push(4),
                Comp::Identity => o.push(5),
            }
        }
        std::fs::write(out, o).unwrap();
    }

    /// Load a `model.int4` produced by `quantize_model` (dequantizes weights to f32).
    pub fn load_int4(path: &str) -> Nnet3 {
        let buf = std::fs::read(path).unwrap();
        let mut c = Cur { b: &buf, p: 0 };
        assert_eq!(c.take(4), b"VRQ4", "bad int4 magic");
        let group = c.u32() as usize;
        let ivector_dim = c.u32() as usize;
        let _num_pdfs = c.u32();
        let n_tid = c.u32() as usize;
        c.take(n_tid * 4); // tid2pdf — read by TransitionModel::load_int4, skipped here
        let glen = c.u32() as usize;
        let graph = String::from_utf8_lossy(c.take(glen)).to_string();
        let nodes = parse_graph_text(&graph);
        let ncomp = c.u32() as usize;
        let mut comps = HashMap::new();
        for _ in 0..ncomp {
            let nl = c.u16() as usize;
            let name = String::from_utf8_lossy(c.take(nl)).to_string();
            let tag = c.take(1)[0];
            let comp = match tag {
                0 => Comp::Affine { w: c.mat(group), b: c.vec() },
                1 => Comp::Linear { w: c.mat(group) },
                2 => {
                    let no = c.u16() as usize;
                    let offsets = (0..no).map(|_| i32::from_le_bytes(c.take(4).try_into().unwrap())).collect();
                    Comp::Tdnn { offsets, w: c.mat(group), b: c.vec() }
                }
                3 => Comp::BatchNorm { mean: c.vec(), var: c.vec(), eps: c.f32(), trms: c.f32() },
                4 => Comp::Relu,
                _ => Comp::Identity,
            };
            comps.insert(name, comp);
        }
        Nnet3 { comps, nodes, ivector_dim }
    }
}
