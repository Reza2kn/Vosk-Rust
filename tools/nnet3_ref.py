#!/usr/bin/env python3
"""Numpy reference forward for the Vosk chain TDNN-F nnet3 (fa-0.42).

Prototype/oracle for the pure-Rust port. Parses am/final.mdl directly (no Kaldi), runs the
acoustic forward, and (later) decodes over HCLG. We debug numerics HERE (fast), then port the
verified forward to Rust layer-by-layer.

Reference target (vosk-python on test/test.wav): «من عرضه این کار رو ندارم»
"""
import sys, struct, re
import numpy as np

MODEL = "/Users/Ajab/AI/w2v-bert-2.0/vosk-model-fa-0.42"
MDL = MODEL + "/am/final.mdl"


# ---------- Kaldi binary reader over an in-memory buffer ----------
class R:
    def __init__(self, buf, pos=0):
        self.b = buf
        self.p = pos

    def i32(self):
        assert self.b[self.p] == 4, f"i32 size byte {self.b[self.p]} @ {self.p}"
        v = struct.unpack_from("<i", self.b, self.p + 1)[0]
        self.p += 5
        return v

    def f32(self):
        assert self.b[self.p] == 4
        v = struct.unpack_from("<f", self.b, self.p + 1)[0]
        self.p += 5
        return v

    def token(self):
        while self.b[self.p] in (0x20, 0x0A, 0x09):
            self.p += 1
        s = self.p
        while self.b[self.p] not in (0x20, 0x0A, 0x09):
            self.p += 1
        t = self.b[s:self.p].decode("latin1")
        self.p += 1  # consume the single delimiter (Kaldi tokens are space-terminated)
        return t

    def matrix(self):
        t = self.token()
        assert t == "FM", f"expected FM got {t}"
        rows, cols = self.i32(), self.i32()
        a = np.frombuffer(self.b, "<f4", rows * cols, self.p).reshape(rows, cols).copy()
        self.p += 4 * rows * cols
        return a

    def vector(self):
        t = self.token()
        assert t == "FV", f"expected FV got {t}"
        dim = self.i32()
        a = np.frombuffer(self.b, "<f4", dim, self.p).copy()
        self.p += 4 * dim
        return a


def find(buf, needle, start=0):
    i = buf.find(needle, start)
    return i


# ---------- parse all components into dicts ----------
def parse_components(buf):
    """Return {name: {type, ...params}} by locating each component's byte span and pulling
    named sub-tokens (robust to field ordering / unknown training-state fields)."""
    starts = [m.start() for m in re.finditer(rb"<ComponentName> ", buf)]
    starts.append(len(buf))
    comps = {}
    order = []
    for k in range(len(starts) - 1):
        s, e = starts[k], starts[k + 1]
        seg = buf[s:e]
        r = R(buf, s + len(b"<ComponentName> "))
        name = r.token()
        ctype = r.token()  # e.g. <TdnnComponent>
        c = {"type": ctype}

        def read_named(tok, kind):
            i = seg.find(tok)
            if i < 0:
                return None
            rr = R(buf, s + i + len(tok))
            # skip spaces then read FM/FV/ivec
            if kind == "M":
                return rr.matrix()
            if kind == "V":
                return rr.vector()
            if kind == "ivec":
                return rr.i32(), rr  # count handled by caller

        if ctype in ("<NaturalGradientAffineComponent>", "<FixedAffineComponent>"):
            c["W"] = read_named(b"<LinearParams> ", "M")
            c["b"] = read_named(b"<BiasParams> ", "V")
        elif ctype == "<LinearComponent>":
            c["W"] = read_named(b"<Params> ", "M")  # LinearComponent uses <Params>, not <LinearParams>
        elif ctype == "<TdnnComponent>":
            i = seg.find(b"<TimeOffsets> ")
            rr = R(buf, s + i + len(b"<TimeOffsets> "))
            n = rr.i32()
            c["offsets"] = [struct.unpack_from("<i", buf, rr.p + 4 * j)[0] for j in range(n)]
            c["W"] = read_named(b"<LinearParams> ", "M")
            c["b"] = read_named(b"<BiasParams> ", "V")
        elif ctype == "<BatchNormComponent>":
            for tok in (b"<Dim> ", b"<BlockDim> "):
                i = seg.find(tok)
                if i >= 0:
                    c[tok.strip().decode()[1:-1].lower()] = R(buf, s + i + len(tok)).i32()
            for tok in (b"<Epsilon> ", b"<TargetRms> "):
                i = seg.find(tok)
                if i >= 0:
                    c[tok.strip().decode()[1:-1].lower()] = R(buf, s + i + len(tok)).f32()
            c["mean"] = read_named(b"<StatsMean> ", "V")
            c["var"] = read_named(b"<StatsVar> ", "V")
        elif ctype in ("<RectifiedLinearComponent>", "<NoOpComponent>",
                       "<GeneralDropoutComponent>", "<SpecAugmentTimeMaskComponent>",
                       "<LogSoftmaxComponent>"):
            pass  # forward handled structurally
        else:
            print("  ?? unhandled type", ctype, name, file=sys.stderr)
        comps[name] = c
        order.append(name)
    return comps, order


# ---------- parse the nnet3 config graph (text lines before <NumComponents>) ----------
def parse_graph(buf):
    txt = buf[buf.find(b"<Nnet3>"):buf.find(b"<NumComponents>")].decode("latin1")
    nodes = {}
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("input-node"):
            m = dict(re.findall(r"(\w[\w-]*)=(\S+)", line))
            nodes[m["name"]] = {"kind": "input", "dim": int(m["dim"])}
        elif line.startswith("dim-range-node"):
            m = dict(re.findall(r"(\w[\w-]*)=(\S+)", line))
            nodes[m["name"]] = {"kind": "dimrange", "src": m["input-node"],
                                "off": int(m["dim-offset"]), "dim": int(m["dim"])}
        elif line.startswith("component-node"):
            name = re.search(r"name=(\S+)", line).group(1)
            comp = re.search(r"component=(\S+)", line).group(1)
            desc = line[line.index("input=") + 6:].strip()
            nodes[name] = {"kind": "component", "component": comp, "desc": desc}
        elif line.startswith("output-node"):
            name = re.search(r"name=(\S+)", line).group(1)
            desc = re.search(r"input=(.+?)\s+objective=", line).group(1)
            nodes[name] = {"kind": "output", "desc": desc}
    return nodes


# ---------- descriptor expression parser → nested tuples ----------
def parse_desc(s):
    s = s.strip()
    toks = re.findall(r"[A-Za-z_][\w.\-]*|-?\d+\.?\d*|[(),]", s)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def nxt():
        t = toks[pos[0]]; pos[0] += 1; return t

    OPS = {"Offset", "Scale", "Sum", "Append", "ReplaceIndex", "Round", "IfDefined", "Const"}

    def parse():
        t = nxt()
        if t in OPS and peek() == "(":
            nxt()  # (
            args = [parse()]
            while peek() == ",":
                nxt(); args.append(parse())
            assert nxt() == ")"
            return (t, *args)
        return t  # a node name or a literal number/index-var

    return parse()


def clamp_offset(arr, n):
    T = arr.shape[0]
    idx = np.clip(np.arange(T) + n, 0, T - 1)
    return arr[idx]


IDENTITY = {"<NoOpComponent>", "<GeneralDropoutComponent>",
            "<SpecAugmentTimeMaskComponent>", "<DropoutComponent>"}


def apply_component(cname, x, comps):
    c = comps[cname]
    t = c["type"]
    if t in IDENTITY:
        return x
    if t == "<RectifiedLinearComponent>":
        return np.maximum(x, 0.0)
    if t == "<BatchNormComponent>":
        eps = c.get("epsilon", 1e-3)
        trms = c.get("targetrms", 1.0)
        scale = trms / np.sqrt(c["var"] + eps)
        return (x - c["mean"]) * scale
    if t == "<TdnnComponent>":
        spliced = np.concatenate([clamp_offset(x, o) for o in c["offsets"]], axis=1)
        y = spliced @ c["W"].T
        b = c.get("b")
        if b is not None and b.size > 0:  # .linear Tdnns carry an empty bias
            y = y + b
        return y
    if t in ("<NaturalGradientAffineComponent>", "<FixedAffineComponent>"):
        y = x @ c["W"].T
        b = c.get("b")
        return y + b if (b is not None and b.size > 0) else y
    if t == "<LinearComponent>":
        return x @ c["W"].T
    if t == "<LogSoftmaxComponent>":
        m = x.max(1, keepdims=True)
        return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))
    raise ValueError("unhandled component type " + t)


def eval_desc(d, nodes, comps, cache, T):
    if isinstance(d, str):
        if re.fullmatch(r"-?\d+\.?\d*", d) or d == "t":
            return d  # literal / index var (handled by caller)
        return eval_node(d, nodes, comps, cache, T)
    op = d[0]
    if op == "Offset":
        return clamp_offset(eval_desc(d[1], nodes, comps, cache, T), int(d[2]))
    if op == "Scale":
        return float(d[1]) * eval_desc(d[2], nodes, comps, cache, T)
    if op == "Sum":
        return eval_desc(d[1], nodes, comps, cache, T) + eval_desc(d[2], nodes, comps, cache, T)
    if op == "Append":
        return np.concatenate([eval_desc(a, nodes, comps, cache, T) for a in d[1:]], axis=1)
    if op == "ReplaceIndex":  # ReplaceIndex(x, t, 0) -> x evaluated at time index 0, broadcast
        arr = eval_desc(d[1], nodes, comps, cache, T)
        return np.tile(arr[0:1], (T, 1))
    if op == "IfDefined":
        return eval_desc(d[1], nodes, comps, cache, T)
    raise ValueError("unhandled descriptor op " + op)


def eval_node(name, nodes, comps, cache, T):
    if name in cache:
        return cache[name]
    node = nodes[name]
    k = node["kind"]
    if k == "input":
        raise ValueError("input node not pre-populated: " + name)
    if k == "dimrange":
        src = eval_node(node["src"], nodes, comps, cache, T)
        r = src[:, node["off"]:node["off"] + node["dim"]]
    elif k == "component":
        x = eval_desc(parse_desc(node["desc"]), nodes, comps, cache, T)
        r = apply_component(node["component"], x, comps)
    elif k == "output":
        r = eval_desc(parse_desc(node["desc"]), nodes, comps, cache, T)
    else:
        raise ValueError("bad node kind " + k)
    cache[name] = r
    return r


def nnet3_forward(feats, ivector, comps, nodes):
    """feats [T,40], ivector [T,40] -> chain-output loglikes [ceil(T/3), 3736]."""
    T = feats.shape[0]
    cache = {"input": feats, "ivector": ivector}
    out = eval_node("output", nodes, comps, cache, T)  # [T, 3736]
    return out[::3]  # frame-subsampling-factor 3 (chain output)


if __name__ == "__main__":
    sys.setrecursionlimit(100000)
    buf = open(MDL, "rb").read()
    comps, order = parse_components(buf)
    nodes = parse_graph(buf)
    print(f"parsed {len(comps)} components, {len(nodes)} graph nodes")
    # dims sanity — the chain output must be num_pdfs = 3736
    oa = comps["output.affine"]
    print("output.affine W:", oa["W"].shape, "b:", oa["b"].shape)
    assert oa["W"].shape[0] == 3736, "chain output dim must equal num_pdfs=3736"

    # ---- MFCC (Kaldi-compatible, torchaudio) on test.wav ----
    import torch, torchaudio
    import torchaudio.compliance.kaldi as tk
    wav, sr = torchaudio.load(MODEL + "/test/test.wav")
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    with open("/tmp/vosk_wav16k.bin", "wb") as wf:  # post-resample [-1,1] 16k samples (Rust MFCC input)
        w = wav[0].numpy().astype("<f4")
        wf.write(struct.pack("<ii", 1, w.shape[0]))
        wf.write(w.tobytes())
    wav = wav * 32768.0  # Kaldi computes features on int16-range samples, not normalized [-1,1]
    feats = tk.mfcc(wav, num_mel_bins=40, num_ceps=40, use_energy=False,
                    low_freq=20.0, high_freq=7600.0, sample_frequency=16000.0).numpy()
    print("MFCC feats:", feats.shape)
    ivector = np.zeros((feats.shape[0], 40), np.float32)  # v1: zeros
    loglikes = nnet3_forward(feats.astype(np.float32), ivector, comps, nodes)
    print("loglikes:", loglikes.shape, "min/max/mean=%.2f/%.2f/%.2f" %
          (loglikes.min(), loglikes.max(), loglikes.mean()),
          "finite:", np.isfinite(loglikes).all())
    # ---- layer-by-layer sanity: mean/std + how much each dim varies over time ----
    cache = {"input": feats.astype(np.float32), "ivector": ivector}
    eval_node("output", nodes, comps, cache, feats.shape[0])
    print("\n%-26s %-14s %8s %8s %8s" % ("node", "shape", "mean", "std", "t-var"))
    for n in ["input", "idct", "batchnorm0", "spec-augment.time-mask",
              "spec-augment.time-mask_2", "delta", "input2",
              "tdnnf1.affine", "tdnnf1.batchnorm", "tdnnf2.noop", "tdnnf5.noop",
              "tdnn17.batchnorm", "prefinal-chain.batchnorm2", "output.affine"]:
        if n in cache:
            a = cache[n]
            tvar = a.std(0).mean()  # avg over-time std per dim (0 => constant over time = dead)
            print("%-26s %-14s %8.3f %8.3f %8.3f" % (n, str(a.shape), a.mean(), a.std(), tvar))

    def dump(path, a):
        a = np.ascontiguousarray(a, "<f4")
        with open(path, "wb") as f:
            f.write(struct.pack("<ii", a.shape[0], a.shape[1]))
            f.write(a.tobytes())

    ll = loglikes.astype("<f4")
    dump("/tmp/vosk_loglikes.bin", ll)
    dump("/tmp/vosk_feats.bin", feats)  # MFCC input to nnet3 (Rust-port verification)
    print("saved /tmp/vosk_loglikes.bin [%d x %d] + /tmp/vosk_feats.bin [%d x %d]" %
          (ll.shape + feats.shape))
    sys.exit(0)
    # front-end + first TDNN-F block dims
    for n in ["idct", "batchnorm0", "tdnnf1.affine", "tdnnf2.linear", "tdnnf2.affine",
              "tdnn17.affine", "prefinal-chain.affine", "prefinal-chain.linear"]:
        c = comps[n]
        if "W" in c and c["W"] is not None:
            extra = f" offsets={c.get('offsets')}" if "offsets" in c else ""
            print(f"  {n:22s} {c['type']:32s} W={c['W'].shape}{extra}")
        else:
            print(f"  {n:22s} {c['type']:32s} mean={None if c.get('mean') is None else c['mean'].shape}")
    print("PARSE OK — output dim 3736 confirmed")
