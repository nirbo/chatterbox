//! CPU preprocessing for Chatterbox T3 fine-tuning.
//!
//! Reads a JSONL manifest, loads+resamples audio to 16kHz, tokenizes text with
//! GPT-2 BPE, and writes compact binary files. These are then fed to the Python
//! GPU stage for S3 tokenization and speaker embedding extraction.
//!
//! Binary format per sample (.bin):
//!   magic:      b"CPR1" (4 bytes)
//!   id_len:     u32 LE
//!   id:         [u8; id_len]
//!   text_len:   u32 LE
//!   text:       [u8; text_len]
//!   n_tokens:   u32 LE
//!   tokens:     [u32 LE; n_tokens]
//!   n_audio:    u32 LE
//!   audio:      [f32 LE; n_audio]     -- target audio at 16kHz
//!   n_ref:      u32 LE
//!   ref_audio:  [f32 LE; n_ref]       -- ref audio at 16kHz (first 15s)

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Deserialize;

const TARGET_SR: u32 = 16_000;
const ENC_COND_LEN: usize = 15 * TARGET_SR as usize; // 15 seconds

#[derive(Parser)]
#[command(about = "Fast CPU preprocessing for Chatterbox T3 (Rust)")]
struct Args {
    /// Path to JSONL manifest (from make_pause_data.py)
    #[arg(long)]
    manifest: String,

    /// Root directory containing wav files
    #[arg(long)]
    data_root: String,

    /// Output directory for .bin files
    #[arg(long)]
    output_dir: String,

    /// Path to tokenizer.json (GPT-2 BPE)
    #[arg(long)]
    tokenizer_json: String,
}

#[derive(Deserialize)]
struct Sample {
    id: String,
    wav: String,
    text: String,
    ref_wav: String,
}

// ── Audio ──────────────────────────────────────────────────────────

fn load_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels as usize;

    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()?
                .into_iter()
                .map(|s| s as f32 * scale)
                .collect()
        }
    };

    // Downmix to mono
    let mono = if ch > 1 {
        raw.chunks(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        raw
    };

    Ok((mono, sr))
}

fn resample(audio: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr || audio.is_empty() {
        return audio.to_vec();
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
        WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_sr as f64 / from_sr as f64;
    let chunk_size = 1024;
    let mut resampler =
        SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1).expect("resampler init");

    let mut output = Vec::with_capacity((audio.len() as f64 * ratio) as usize + chunk_size);

    for chunk in audio.chunks(chunk_size) {
        let mut buf = chunk.to_vec();
        if buf.len() < chunk_size {
            buf.resize(chunk_size, 0.0);
        }
        let input = vec![buf];
        if let Ok(result) = resampler.process(&input, None) {
            output.extend_from_slice(&result[0]);
        }
    }

    output
}

// ── Binary format ──────────────────────────────────────────────────

fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32_slice(w: &mut impl Write, data: &[f32]) -> std::io::Result<()> {
    // Safety: f32 is Pod, reinterpreting as bytes is fine on LE platforms
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    w.write_all(bytes)
}

fn save_bin(
    path: &Path,
    id: &str,
    text: &str,
    tokens: &[u32],
    audio: &[f32],
    ref_audio: &[f32],
) -> Result<()> {
    let mut w = BufWriter::new(fs::File::create(path)?);

    w.write_all(b"CPR1")?;

    let id_b = id.as_bytes();
    write_u32(&mut w, id_b.len() as u32)?;
    w.write_all(id_b)?;

    let text_b = text.as_bytes();
    write_u32(&mut w, text_b.len() as u32)?;
    w.write_all(text_b)?;

    write_u32(&mut w, tokens.len() as u32)?;
    for &t in tokens {
        write_u32(&mut w, t)?;
    }

    write_u32(&mut w, audio.len() as u32)?;
    write_f32_slice(&mut w, audio)?;

    write_u32(&mut w, ref_audio.len() as u32)?;
    write_f32_slice(&mut w, ref_audio)?;

    Ok(())
}

// ── Main ───────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("Loading tokenizer from {}...", args.tokenizer_json);
    let tokenizer = tokenizers::Tokenizer::from_file(&args.tokenizer_json)
        .map_err(|e| anyhow::anyhow!("tokenizer: {}", e))?;
    let tokenizer = Arc::new(tokenizer);

    eprintln!("Reading manifest...");
    let manifest = fs::read_to_string(&args.manifest)?;
    let samples: Vec<Sample> = manifest
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).expect("bad manifest JSON"))
        .collect();
    let total = samples.len();
    eprintln!("Loaded {} samples", total);

    fs::create_dir_all(&args.output_dir)?;

    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                 {pos}/{len} ({per_sec}, ETA {eta})",
            )
            .unwrap()
            .progress_chars("=>-"),
    );

    let skipped = Arc::new(AtomicUsize::new(0));

    samples.par_iter().for_each(|sample| {
        let out = PathBuf::from(&args.output_dir).join(format!("{}.bin", sample.id));
        if out.exists() {
            pb.inc(1);
            return;
        }

        let root = Path::new(&args.data_root);

        // Load + resample target audio
        let audio_16k = match load_wav(&root.join(&sample.wav)) {
            Ok((a, sr)) => resample(&a, sr, TARGET_SR),
            Err(e) => {
                eprintln!("SKIP {}: {}", sample.id, e);
                skipped.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
                return;
            }
        };

        // Load + resample ref audio, truncate to 15s for conditioning
        let ref_16k = match load_wav(&root.join(&sample.ref_wav)) {
            Ok((a, sr)) => {
                let r = resample(&a, sr, TARGET_SR);
                if r.len() > ENC_COND_LEN {
                    r[..ENC_COND_LEN].to_vec()
                } else {
                    r
                }
            }
            Err(e) => {
                eprintln!("SKIP {} (ref): {}", sample.id, e);
                skipped.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
                return;
            }
        };

        // BPE tokenize
        let tokens: Vec<u32> = match tokenizer.encode(sample.text.as_str(), false) {
            Ok(enc) => enc.get_ids().to_vec(),
            Err(e) => {
                eprintln!("SKIP {} (tok): {}", sample.id, e);
                skipped.fetch_add(1, Ordering::Relaxed);
                pb.inc(1);
                return;
            }
        };

        if let Err(e) = save_bin(&out, &sample.id, &sample.text, &tokens, &audio_16k, &ref_16k) {
            eprintln!("SKIP {} (save): {}", sample.id, e);
            skipped.fetch_add(1, Ordering::Relaxed);
        }

        pb.inc(1);
    });

    pb.finish();
    let sk = skipped.load(Ordering::Relaxed);
    eprintln!("Done! Processed: {}, Skipped: {}", total - sk, sk);
    Ok(())
}
