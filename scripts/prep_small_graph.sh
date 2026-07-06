#!/usr/bin/env bash
# One-time offline model-prep for Vosk SMALL models (lookahead graphs → a static const HCLG that
# Vosk-Rust can load). The small model ships graph/HCLr.fst (olabel_lookahead) + graph/Gr.fst
# (ngram) which rustfst can't read; Vosk composes them at runtime with libvosk's OpenFst lookahead
# machinery. Here we do that composition ONCE, offline, with OpenFst — the runtime stays pure Rust.
#
# Requires OpenFst (with its ngram + lookahead plugins):  brew install openfst
# Usage:  scripts/prep_small_graph.sh <vosk-small-model-dir>
# Writes: <dir>/graph/HCLG.fst  (const/standard) and <dir>/graph/words.txt
set -euo pipefail

MODEL="${1:?usage: prep_small_graph.sh <model-dir>}"
G="$MODEL/graph"
PREFIX="$(brew --prefix openfst 2>/dev/null || echo /opt/homebrew/opt/openfst)"
export PATH="$PREFIX/bin:$PATH"
export DYLD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/fst:${DYLD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/fst:${LD_LIBRARY_PATH:-}"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
cd "$TMP"

echo "1/4  recover words.txt from Gr.fst's embedded symbol table + strip symtables"
fstsymbols --save_osymbols=words.txt "$G/Gr.fst" /dev/null
fstsymbols --clear_isymbols --clear_osymbols "$G/Gr.fst" Gr_nosym.fst

echo "2/4  lookahead-compose HCLr (olabel_lookahead) ∘ Gr  → HCLG"
fstcompose "$G/HCLr.fst" Gr_nosym.fst HCLG_la.fst
fstconnect HCLG_la.fst HCLG_conn.fst

echo "3/4  relabel disambiguation transition-ids (#0…#N) to epsilon on the INPUT side"
# these graph-only symbols become bogus emitting tids otherwise → decoder panics.
awk '{print $1" 0"}' "$G/disambig_tid.int" > tid_relabel.txt
fstrelabel --relabel_ipairs=tid_relabel.txt HCLG_conn.fst HCLG_rmdis.fst

echo "4/4  convert to const/standard (the graph rustfst loads)"
fstconvert --fst_type=const HCLG_rmdis.fst "$G/HCLG.fst"
cp words.txt "$G/words.txt"

echo "done → $G/HCLG.fst  +  $G/words.txt"
fstinfo "$G/HCLG.fst" | grep -iE "fst type|arc type|# of states|# of arcs" || true
