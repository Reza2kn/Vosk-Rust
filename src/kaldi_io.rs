//! Minimal reader for Kaldi's binary serialization (the `\0B` format).
//!
//! Tokens (`<Foo>`) are space-terminated ASCII even in binary mode. Basic types are written as
//! `[size-byte][little-endian bytes]` (int32/float → size-byte 4). Integer vectors are
//! `[i32 count][raw i32 × count]`. Float matrices are token `FM` then `[i32 rows][i32 cols]` then
//! `rows*cols` raw f32; float vectors are token `FV` then `[i32 dim]` then `dim` raw f32.

use std::io::{Error, ErrorKind, Read, Result};

pub struct KaldiReader<R: Read> {
    r: R,
}

fn err(msg: &str) -> Error {
    Error::new(ErrorKind::InvalidData, msg.to_string())
}

impl<R: Read> KaldiReader<R> {
    pub fn new(r: R) -> Self {
        KaldiReader { r }
    }

    #[inline]
    fn byte(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.r.read_exact(&mut b)?;
        Ok(b[0])
    }

    /// Consume the `\0B` binary marker.
    pub fn expect_binary(&mut self) -> Result<()> {
        let a = self.byte()?;
        let b = self.byte()?;
        if a != 0 || b != b'B' {
            return Err(err("not a binary Kaldi stream (missing \\0B)"));
        }
        Ok(())
    }

    /// Read one space-terminated token (skips leading spaces). e.g. `<Topology>`, `FM`, `[`.
    pub fn read_token(&mut self) -> Result<String> {
        let mut s = String::new();
        loop {
            let c = self.byte()?;
            if c == b' ' || c == b'\n' || c == b'\t' {
                if s.is_empty() {
                    continue;
                }
                break;
            }
            s.push(c as char);
        }
        Ok(s)
    }

    /// Read a size-prefixed i32 (`[4][LE i32]`).
    pub fn read_i32(&mut self) -> Result<i32> {
        let sz = self.byte()?;
        if sz != 4 {
            return Err(err("expected i32 size byte = 4"));
        }
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(i32::from_le_bytes(b))
    }

    /// Read a size-prefixed f32.
    pub fn read_f32(&mut self) -> Result<f32> {
        let sz = self.byte()?;
        if sz != 4 {
            return Err(err("expected f32 size byte = 4"));
        }
        let mut b = [0u8; 4];
        self.r.read_exact(&mut b)?;
        Ok(f32::from_le_bytes(b))
    }

    /// Read an integer vector: `[i32 count][raw i32 × count]`.
    pub fn read_i32_vec(&mut self) -> Result<Vec<i32>> {
        let n = self.read_i32()?;
        if n < 0 {
            return Err(err("negative vector length"));
        }
        let n = n as usize;
        let mut buf = vec![0u8; n * 4];
        self.r.read_exact(&mut buf)?;
        Ok(buf.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Read raw f32 data (`count` little-endian f32, no size bytes).
    fn read_f32_raw(&mut self, count: usize) -> Result<Vec<f32>> {
        let mut buf = vec![0u8; count * 4];
        self.r.read_exact(&mut buf)?;
        Ok(buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Read a float vector: token `FV` then `[i32 dim]` then `dim` raw f32.
    pub fn read_float_vec(&mut self) -> Result<Vec<f32>> {
        let tok = self.read_token()?;
        if tok != "FV" {
            return Err(err(&format!("expected FV, got {tok}")));
        }
        let dim = self.read_i32()? as usize;
        self.read_f32_raw(dim)
    }

    /// Read a float matrix: token `FM` then `[i32 rows][i32 cols]` then `rows*cols` raw f32.
    /// Returns (rows, cols, row-major data).
    pub fn read_float_matrix(&mut self) -> Result<(usize, usize, Vec<f32>)> {
        let tok = self.read_token()?;
        if tok != "FM" {
            return Err(err(&format!("expected FM, got {tok}")));
        }
        let rows = self.read_i32()? as usize;
        let cols = self.read_i32()? as usize;
        let data = self.read_f32_raw(rows * cols)?;
        Ok((rows, cols, data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::BufReader;

    const MODEL: &str = "/Users/Ajab/AI/w2v-bert-2.0/vosk-model-fa-0.42/am/final.mdl";

    #[test]
    fn reads_final_mdl_header() {
        let f = match std::fs::File::open(MODEL) {
            Ok(f) => f,
            Err(_) => return, // model not present on this machine — skip
        };
        let mut kr = KaldiReader::new(BufReader::new(f));
        kr.expect_binary().unwrap();
        assert_eq!(kr.read_token().unwrap(), "<TransitionModel>");
        assert_eq!(kr.read_token().unwrap(), "<Topology>");
        let phones = kr.read_i32_vec().unwrap();
        // topology lists the phone ids 1..=122
        assert_eq!(phones.len(), 122);
        assert_eq!(phones[0], 1);
        assert_eq!(phones[121], 122);
    }
}
