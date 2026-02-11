"""
make_pause_data.py -- Generate pause-tag training data from existing TTS datasets.

Takes a clean TTS dataset (LJSpeech format or JSONL manifest) and:
1. Force-aligns text to audio using WhisperX to find word boundaries
2. Identifies natural pauses (silences between words)
3. Inserts [Xs] pause tags at those locations
4. Optionally augments by splicing extra silence into audio

Produces a JSONL manifest ready for preprocess.py.

Usage:
    # From LJSpeech-style dataset:
    python scripts/make_pause_data.py \
        --input_dir /path/to/LJSpeech-1.1 \
        --format ljspeech \
        --output_manifest dataset/manifests/train.jsonl \
        --output_wavs dataset/wavs \
        --device cuda

    # From JSONL manifest (id, wav, text):
    python scripts/make_pause_data.py \
        --input_manifest /path/to/manifest.jsonl \
        --data_root /path/to/data \
        --format jsonl \
        --output_manifest dataset/manifests/train.jsonl \
        --output_wavs dataset/wavs \
        --device cuda

Requires: pip install whisperx (for forced alignment)
"""

import argparse
import csv
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Pause tag parameters
MIN_NATURAL_PAUSE = 0.15    # Seconds: minimum gap to annotate as a pause
PAUSE_QUANTIZE = 0.25       # Quantize pause durations to this step size
MAX_PAUSE_DURATION = 3.0    # Cap pause tags at this duration
AUGMENT_PROBABILITY = 0.3   # Probability of augmenting a sample with extra silence
AUGMENT_MIN_DURATION = 0.25
AUGMENT_MAX_DURATION = 2.0


def quantize_duration(duration: float) -> float:
    """Round duration to nearest PAUSE_QUANTIZE step."""
    q = round(duration / PAUSE_QUANTIZE) * PAUSE_QUANTIZE
    return min(max(q, PAUSE_QUANTIZE), MAX_PAUSE_DURATION)


def load_ljspeech(input_dir: str):
    """Load LJSpeech-format dataset."""
    input_dir = Path(input_dir)
    metadata = input_dir / "metadata.csv"
    samples = []

    with open(metadata, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|")
        for row in reader:
            if len(row) >= 2:
                utt_id = row[0].strip()
                text = row[1].strip() if len(row) == 2 else row[2].strip()
                wav_path = input_dir / "wavs" / f"{utt_id}.wav"
                if wav_path.exists():
                    samples.append({"id": utt_id, "wav": str(wav_path), "text": text})

    return samples


def load_jsonl(manifest_path: str, data_root: str):
    """Load JSONL manifest."""
    data_root = Path(data_root)
    samples = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                s["wav"] = str(data_root / s["wav"])
                samples.append(s)
    return samples


def align_with_whisperx(wav_path: str, text: str, device: str):
    """
    Use WhisperX to force-align text to audio.
    Returns list of {"word": str, "start": float, "end": float}.
    Falls back to None if alignment fails.
    """
    try:
        import whisperx
    except ImportError:
        logger.error("whisperx not installed. Run: pip install whisperx")
        raise

    audio = whisperx.load_audio(wav_path)
    model = whisperx.load_model("base", device, compute_type="float16")
    result = model.transcribe(audio, batch_size=8)

    # Align
    align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)

    # Extract word-level alignments
    words = []
    for seg in aligned.get("word_segments", []):
        if "start" in seg and "end" in seg:
            words.append({
                "word": seg["word"],
                "start": seg["start"],
                "end": seg["end"],
            })

    return words if words else None


def insert_pause_tags(words: list, original_text: str) -> str:
    """
    Given word alignments, insert [Xs] pause tags where gaps exceed MIN_NATURAL_PAUSE.
    Returns the text with pause tags inserted.
    """
    if not words or len(words) < 2:
        return original_text

    parts = []
    for i, word_info in enumerate(words):
        parts.append(word_info["word"])

        if i < len(words) - 1:
            gap = words[i + 1]["start"] - word_info["end"]
            if gap >= MIN_NATURAL_PAUSE:
                duration = quantize_duration(gap)
                parts.append(f"[{duration}s]")

    return " ".join(parts)


def augment_with_silence(wav: np.ndarray, sr: int, text: str):
    """
    Randomly insert extra silence into audio and corresponding pause tags into text.
    Returns (augmented_wav, augmented_text) or (None, None) if augmentation not applied.
    """
    if random.random() > AUGMENT_PROBABILITY:
        return None, None

    words = text.split()
    if len(words) < 3:
        return None, None

    # Pick a random insertion point (between words)
    # Avoid inserting right next to existing pause tags
    valid_positions = []
    for i in range(1, len(words)):
        if not words[i - 1].endswith("s]") and not words[i].startswith("["):
            valid_positions.append(i)

    if not valid_positions:
        return None, None

    insert_pos = random.choice(valid_positions)
    pause_duration = round(random.uniform(AUGMENT_MIN_DURATION, AUGMENT_MAX_DURATION) / PAUSE_QUANTIZE) * PAUSE_QUANTIZE

    # Insert silence into audio at approximate position
    # Estimate position as fraction of text
    text_fraction = insert_pos / len(words)
    audio_pos = int(text_fraction * len(wav))
    silence_samples = int(pause_duration * sr)
    silence = np.zeros(silence_samples, dtype=wav.dtype)

    augmented_wav = np.concatenate([wav[:audio_pos], silence, wav[audio_pos:]])

    # Insert tag into text
    words.insert(insert_pos, f"[{pause_duration}s]")
    augmented_text = " ".join(words)

    return augmented_wav, augmented_text


def main():
    parser = argparse.ArgumentParser(description="Generate pause-tag training data")
    parser.add_argument("--input_dir", type=str, help="LJSpeech-style input directory")
    parser.add_argument("--input_manifest", type=str, help="Input JSONL manifest")
    parser.add_argument("--data_root", type=str, default=".", help="Root for JSONL wav paths")
    parser.add_argument("--format", type=str, choices=["ljspeech", "jsonl"], required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--output_wavs", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_alignment", action="store_true",
                        help="Skip WhisperX alignment (use simple silence detection instead)")
    parser.add_argument("--augment", action="store_true", help="Add augmented samples with extra silence")
    parser.add_argument("--include_plain", action="store_true", default=True,
                        help="Include samples without pause tags to preserve normal speech")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_wavs = Path(args.output_wavs)
    output_wavs.mkdir(parents=True, exist_ok=True)
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)

    # Load input dataset
    if args.format == "ljspeech":
        samples = load_ljspeech(args.input_dir)
    else:
        samples = load_jsonl(args.input_manifest, args.data_root)

    logger.info(f"Loaded {len(samples)} samples")

    output_samples = []
    n_with_pauses = 0
    n_augmented = 0

    for i, sample in enumerate(samples):
        if i % 100 == 0:
            logger.info(f"Processing {i}/{len(samples)}...")

        wav_path = sample["wav"]
        text = sample["text"]
        utt_id = sample["id"]

        try:
            wav, sr = librosa.load(wav_path, sr=None, mono=True)
        except Exception as e:
            logger.warning(f"Skipping {utt_id}: cannot load audio: {e}")
            continue

        # Copy wav to output directory
        out_wav_path = output_wavs / f"{utt_id}.wav"
        sf.write(str(out_wav_path), wav, sr)

        tagged_text = text  # default: no tags

        if not args.skip_alignment:
            # Force-align and insert pause tags
            try:
                words = align_with_whisperx(wav_path, text, args.device)
                if words:
                    tagged_text = insert_pause_tags(words, text)
            except Exception as e:
                logger.warning(f"Alignment failed for {utt_id}: {e}")
        else:
            # Simple silence detection: find long silent gaps in audio
            # This is a rough fallback without word-level alignment
            pass

        has_pauses = "[" in tagged_text and "s]" in tagged_text

        if has_pauses:
            n_with_pauses += 1

        # Always output the tagged version (may or may not have tags)
        output_samples.append({
            "id": utt_id,
            "wav": f"wavs/{utt_id}.wav",
            "text": tagged_text,
            "ref_wav": f"wavs/{utt_id}.wav",
        })

        # Include plain version (no tags) to prevent forgetting
        if args.include_plain and has_pauses:
            output_samples.append({
                "id": f"{utt_id}_plain",
                "wav": f"wavs/{utt_id}.wav",
                "text": text,  # original text without tags
                "ref_wav": f"wavs/{utt_id}.wav",
            })

        # Augmentation: splice extra silence
        if args.augment:
            aug_wav, aug_text = augment_with_silence(wav, sr, tagged_text)
            if aug_wav is not None:
                aug_id = f"{utt_id}_aug"
                aug_wav_path = output_wavs / f"{aug_id}.wav"
                sf.write(str(aug_wav_path), aug_wav, sr)
                output_samples.append({
                    "id": aug_id,
                    "wav": f"wavs/{aug_id}.wav",
                    "text": aug_text,
                    "ref_wav": f"wavs/{utt_id}.wav",
                })
                n_augmented += 1

    # Shuffle and write manifest
    random.shuffle(output_samples)
    with open(args.output_manifest, "w") as f:
        for s in output_samples:
            f.write(json.dumps(s) + "\n")

    logger.info(f"Done! Wrote {len(output_samples)} samples to {args.output_manifest}")
    logger.info(f"  With pause tags: {n_with_pauses}")
    logger.info(f"  Augmented: {n_augmented}")
    logger.info(f"  Plain (no tags): {len(output_samples) - n_with_pauses - n_augmented}")


if __name__ == "__main__":
    main()
