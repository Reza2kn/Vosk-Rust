//! Kaldi-compatible MFCC front end (matches torchaudio.compliance.kaldi / the vosk conf).
//!
//! 40 mel bins, 40 cepstra, no energy, low 20 / high 7600 Hz, povey window, preemph 0.97,
//! cepstral lifter 22, snip-edges, remove-DC, no dither, FFT rounded to a power of two.
//! Input samples must be in Kaldi's int16 range (multiply normalized [-1,1] audio by 32768).

use crate::nnet3::Mat;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f32::consts::PI;

fn mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

pub struct Mfcc {
    frame_len: usize,
    frame_shift: usize,
    fft_size: usize,
    num_ceps: usize,
    win: Vec<f32>,
    filt: Vec<Vec<f32>>, // [num_mel][nbins]
    dct: Vec<Vec<f32>>,  // [num_ceps][num_mel]
    lift: Vec<f32>,
    preemph: f32,
}

impl Mfcc {
    /// Vosk fa-0.42 config.
    pub fn vosk(sr: f32) -> Mfcc {
        let (num_mel, num_ceps) = (40usize, 40usize);
        let (low, high, preemph, lifter) = (20.0f32, 7600.0f32, 0.97f32, 22.0f32);
        let frame_len = (0.025 * sr).round() as usize; // 400
        let frame_shift = (0.010 * sr).round() as usize; // 160
        let mut fft_size = 1;
        while fft_size < frame_len {
            fft_size <<= 1;
        }
        let nbins = fft_size / 2 + 1;
        let a = 2.0 * PI / (frame_len as f32 - 1.0);
        let win = (0..frame_len).map(|i| (0.5 - 0.5 * (a * i as f32).cos()).powf(0.85)).collect();

        let (mel_low, mel_high) = (mel(low), mel(high));
        let delta = (mel_high - mel_low) / (num_mel as f32 + 1.0);
        let mut filt = vec![vec![0.0f32; nbins]; num_mel];
        for (m, f) in filt.iter_mut().enumerate() {
            let (l, c, r) = (mel_low + m as f32 * delta,
                             mel_low + (m + 1) as f32 * delta,
                             mel_low + (m + 2) as f32 * delta);
            for (i, wgt) in f.iter_mut().enumerate() {
                let ml = mel(sr / fft_size as f32 * i as f32);
                if ml > l && ml < r {
                    *wgt = if ml <= c { (ml - l) / (c - l) } else { (r - ml) / (r - c) };
                }
            }
        }
        let mut dct = vec![vec![0.0f32; num_mel]; num_ceps];
        for (k, row) in dct.iter_mut().enumerate() {
            for (n, v) in row.iter_mut().enumerate() {
                *v = if k == 0 {
                    (1.0 / num_mel as f32).sqrt()
                } else {
                    (2.0 / num_mel as f32).sqrt() * (PI / num_mel as f32 * k as f32 * (n as f32 + 0.5)).cos()
                };
            }
        }
        let lift = (0..num_ceps).map(|k| 1.0 + 0.5 * lifter * (PI * k as f32 / lifter).sin()).collect();
        Mfcc { frame_len, frame_shift, fft_size, num_ceps, win, filt, dct, lift, preemph }
    }

    /// samples: int16-range mono audio. Returns [num_frames × num_ceps].
    pub fn compute(&self, samples: &[f32]) -> Mat {
        let num_mel = self.filt.len();
        let nbins = self.fft_size / 2 + 1;
        if samples.len() < self.frame_len {
            return Mat::new(0, self.num_ceps);
        }
        let num_frames = (samples.len() - self.frame_len) / self.frame_shift + 1;
        let mut out = Mat::new(num_frames, self.num_ceps);
        let fft = FftPlanner::<f32>::new().plan_fft_forward(self.fft_size);
        let mut buf = vec![Complex::new(0.0f32, 0.0); self.fft_size];

        for t in 0..num_frames {
            let start = t * self.frame_shift;
            let frame = &samples[start..start + self.frame_len];
            let mean: f32 = frame.iter().sum::<f32>() / self.frame_len as f32;
            let mut w: Vec<f32> = frame.iter().map(|x| x - mean).collect();
            for i in (1..self.frame_len).rev() {
                w[i] -= self.preemph * w[i - 1];
            }
            w[0] -= self.preemph * w[0];
            for i in 0..self.frame_len {
                w[i] *= self.win[i];
            }
            for (i, b) in buf.iter_mut().enumerate() {
                *b = Complex::new(if i < self.frame_len { w[i] } else { 0.0 }, 0.0);
            }
            fft.process(&mut buf);

            let mut logmel = vec![0.0f32; num_mel];
            let power: Vec<f32> = buf[..nbins].iter().map(|c| c.re * c.re + c.im * c.im).collect();
            for (m, f) in self.filt.iter().enumerate() {
                let e: f32 = f.iter().zip(&power).map(|(a, b)| a * b).sum();
                logmel[m] = e.max(f32::EPSILON).ln();
            }
            let orow = &mut out.d[t * self.num_ceps..(t + 1) * self.num_ceps];
            for k in 0..self.num_ceps {
                let c: f32 = self.dct[k].iter().zip(&logmel).map(|(a, b)| a * b).sum();
                orow[k] = c * self.lift[k];
            }
        }
        out
    }
}
