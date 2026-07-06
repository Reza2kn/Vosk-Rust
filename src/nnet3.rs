//! Pure-Rust forward pass for the Vosk chain TDNN-F nnet3 (fa-0.42).
//!
//! Faithful port of `tools/nnet3_ref.py` (the numpy reference that exact-matches vosk). Parses
//! `am/final.mdl` directly, executes the descriptor graph over whole-utterance [T, dim] matrices,
//! and returns chain-output log-likelihoods (frame-subsampled by 3). Identity components
//! (Dropout/NoOp/SpecAugment) pass through; the xent branch is never evaluated.

use crate::kaldi_io::KaldiReader;
use std::collections::HashMap;

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
        // C[t,o] = A[t,k] · B[k,o] with A=self (row-major), B = wᵀ (so B[k,o] = w.d[o*inn+k]).
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
        Nnet3 { comps: parse_components(&buf), nodes: parse_graph(&buf) }
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

fn parse_graph(buf: &[u8]) -> HashMap<String, Node> {
    let a = find(buf, b"<Nnet3>", 0).unwrap();
    let b = find(buf, b"<NumComponents>", 0).unwrap();
    let txt = String::from_utf8_lossy(&buf[a..b]);
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
