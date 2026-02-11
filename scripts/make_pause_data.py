"""
make_pause_data.py -- Generate pause-tag training data from existing TTS datasets.

Takes a clean TTS dataset (LJSpeech format or JSONL manifest) and:
1. Force-aligns text to audio using WhisperX alignment (wav2vec2) to find word boundaries
2. Identifies natural pauses (silences between words)
3. Inserts [Xs] pause tags at those locations
4. Optionally augments by splicing extra silence into audio

Key optimizations:
- Skips Whisper ASR entirely (we already have transcripts), only uses wav2vec2 alignment
- Runs N parallel worker processes, each with its own alignment model on GPU
- With 32GB VRAM, can run 8+ workers for near-linear speedup

Produces a JSONL manifest ready for preprocess.py.

Usage:
    python scripts/make_pause_data.py \
        --input_manifest dataset/manifests/train_raw.jsonl \
        --data_root dataset/ \
        --format jsonl \
        --output_manifest dataset/manifests/train.jsonl \
        --output_wavs dataset/wavs \
        --augment \
        --workers 8 \
        --device cuda

Requires: pip install whisperx (for forced alignment)
"""

import argparse
import csv
import json
import logging
import random
import time
from multiprocessing import Process, Queue, current_process
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


def _patch_torch_load():
    """Monkeypatch torch.load to default weights_only=False for pyannote/whisperx compat."""
    import functools
    import torch
    _original = torch.load

    @functools.wraps(_original)
    def _patched(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _original(*args, **kwargs)

    torch.load = _patched


def insert_pause_tags(words: list, original_text: str) -> str:
    """Given word alignments, insert [Xs] pause tags where gaps exceed MIN_NATURAL_PAUSE."""
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


def augment_with_silence(wav: np.ndarray, sr: int, text: str, rng: random.Random):
    """Randomly insert extra silence into audio and corresponding pause tags into text."""
    if rng.random() > AUGMENT_PROBABILITY:
        return None, None

    words = text.split()
    if len(words) < 3:
        return None, None

    valid_positions = []
    for i in range(1, len(words)):
        if not words[i - 1].endswith("s]") and not words[i].startswith("["):
            valid_positions.append(i)

    if not valid_positions:
        return None, None

    insert_pos = rng.choice(valid_positions)
    pause_duration = round(rng.uniform(AUGMENT_MIN_DURATION, AUGMENT_MAX_DURATION) / PAUSE_QUANTIZE) * PAUSE_QUANTIZE

    text_fraction = insert_pos / len(words)
    audio_pos = int(text_fraction * len(wav))
    silence_samples = int(pause_duration * sr)
    silence = np.zeros(silence_samples, dtype=wav.dtype)

    augmented_wav = np.concatenate([wav[:audio_pos], silence, wav[audio_pos:]])

    words.insert(insert_pos, f"[{pause_duration}s]")
    augmented_text = " ".join(words)

    return augmented_wav, augmented_text


def worker_fn(worker_id: int, task_queue: Queue, result_queue: Queue,
              output_wavs_str: str, device: str, do_augment: bool,
              include_plain: bool, seed: int):
    """Worker process: loads its own alignment model, processes samples from queue."""
    import whisperx

    _patch_torch_load()

    # Each worker gets its own RNG for reproducible augmentation
    rng = random.Random(seed + worker_id)

    # Load alignment model in this process
    align_model, align_metadata = whisperx.load_align_model(language_code="en", device=device)
    output_wavs = Path(output_wavs_str)

    logger.info(f"Worker {worker_id}: alignment model loaded, ready")

    processed = 0
    while True:
        item = task_queue.get()
        if item is None:  # poison pill
            break

        sample = item
        wav_path = sample["wav"]
        text = sample["text"]
        utt_id = sample["id"]

        results = []

        try:
            wav, sr = librosa.load(wav_path, sr=None, mono=True)
        except Exception as e:
            result_queue.put(results)
            continue

        # Write wav to output
        out_wav_path = output_wavs / f"{utt_id}.wav"
        sf.write(str(out_wav_path), wav, sr)

        # Align using known transcript (no ASR)
        tagged_text = text
        try:
            audio = whisperx.load_audio(wav_path)
            duration = len(audio) / 16000
            segments = [{"start": 0.0, "end": duration, "text": text}]
            aligned = whisperx.align(segments, align_model, align_metadata, audio, device)

            words = []
            for seg in aligned.get("word_segments", []):
                if "start" in seg and "end" in seg:
                    words.append({"word": seg["word"], "start": seg["start"], "end": seg["end"]})

            if words:
                tagged_text = insert_pause_tags(words, text)
        except Exception as e:
            pass  # keep original text

        has_pauses = "[" in tagged_text and "s]" in tagged_text

        results.append({
            "id": utt_id,
            "wav": f"wavs/{utt_id}.wav",
            "text": tagged_text,
            "ref_wav": f"wavs/{utt_id}.wav",
        })

        if include_plain and has_pauses:
            results.append({
                "id": f"{utt_id}_plain",
                "wav": f"wavs/{utt_id}.wav",
                "text": text,
                "ref_wav": f"wavs/{utt_id}.wav",
            })

        if do_augment:
            aug_wav, aug_text = augment_with_silence(wav, sr, tagged_text, rng)
            if aug_wav is not None:
                aug_id = f"{utt_id}_aug"
                aug_wav_path = output_wavs / f"{aug_id}.wav"
                sf.write(str(aug_wav_path), aug_wav, sr)
                results.append({
                    "id": aug_id,
                    "wav": f"wavs/{aug_id}.wav",
                    "text": aug_text,
                    "ref_wav": f"wavs/{utt_id}.wav",
                })

        result_queue.put(results)
        processed += 1

    logger.info(f"Worker {worker_id}: done ({processed} samples)")


def main():
    parser = argparse.ArgumentParser(description="Generate pause-tag training data")
    parser.add_argument("--input_dir", type=str, help="LJSpeech-style input directory")
    parser.add_argument("--input_manifest", type=str, help="Input JSONL manifest")
    parser.add_argument("--data_root", type=str, default=".", help="Root for JSONL wav paths")
    parser.add_argument("--format", type=str, choices=["ljspeech", "jsonl"], required=True)
    parser.add_argument("--output_manifest", type=str, required=True)
    parser.add_argument("--output_wavs", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (each loads its own model)")
    parser.add_argument("--skip_alignment", action="store_true",
                        help="Skip WhisperX alignment")
    parser.add_argument("--augment", action="store_true",
                        help="Add augmented samples with extra silence")
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

    if args.skip_alignment:
        # Simple pass-through without alignment
        logger.info("Skipping alignment, writing samples directly...")
        output_samples = []
        for sample in samples:
            output_samples.append({
                "id": sample["id"],
                "wav": f"wavs/{sample['id']}.wav",
                "text": sample["text"],
                "ref_wav": f"wavs/{sample['id']}.wav",
            })
        random.shuffle(output_samples)
        with open(args.output_manifest, "w") as f:
            for s in output_samples:
                f.write(json.dumps(s) + "\n")
        logger.info(f"Done! Wrote {len(output_samples)} samples")
        return

    n_workers = min(args.workers, len(samples))
    logger.info(f"Starting {n_workers} worker processes...")

    task_queue = Queue(maxsize=n_workers * 4)
    result_queue = Queue()

    # Start workers
    workers = []
    for i in range(n_workers):
        p = Process(
            target=worker_fn,
            args=(i, task_queue, result_queue, str(output_wavs),
                  args.device, args.augment, args.include_plain, args.seed),
            daemon=True,
        )
        p.start()
        workers.append(p)

    # Feed samples to workers
    t0 = time.time()
    n_sent = 0
    n_received = 0
    output_samples = []
    n_with_pauses = 0
    n_augmented = 0

    # Feed all samples, collecting results as they come
    for sample in samples:
        task_queue.put(sample)
        n_sent += 1

        # Drain results non-blocking to prevent queue from getting too large
        while not result_queue.empty():
            results = result_queue.get_nowait()
            n_received += 1
            for r in results:
                if "[" in r["text"] and "s]" in r["text"]:
                    if r["id"].endswith("_aug"):
                        n_augmented += 1
                    elif not r["id"].endswith("_plain"):
                        n_with_pauses += 1
                output_samples.append(r)

            if n_received % 500 == 0 and n_received > 0:
                elapsed = time.time() - t0
                rate = n_received / elapsed
                eta = (len(samples) - n_received) / rate if rate > 0 else 0
                logger.info(f"Progress: {n_received}/{len(samples)} "
                            f"({rate:.1f} samples/s, ETA {eta/60:.0f}min)")

    # Send poison pills to stop workers
    for _ in range(n_workers):
        task_queue.put(None)

    # Collect remaining results
    while n_received < len(samples):
        results = result_queue.get(timeout=120)
        n_received += 1
        for r in results:
            if "[" in r["text"] and "s]" in r["text"]:
                if r["id"].endswith("_aug"):
                    n_augmented += 1
                elif not r["id"].endswith("_plain"):
                    n_with_pauses += 1
            output_samples.append(r)

        if n_received % 500 == 0:
            elapsed = time.time() - t0
            rate = n_received / elapsed
            eta = (len(samples) - n_received) / rate if rate > 0 else 0
            logger.info(f"Progress: {n_received}/{len(samples)} "
                        f"({rate:.1f} samples/s, ETA {eta/60:.0f}min)")

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=30)

    # Shuffle and write manifest
    random.shuffle(output_samples)
    with open(args.output_manifest, "w") as f:
        for s in output_samples:
            f.write(json.dumps(s) + "\n")

    elapsed = time.time() - t0
    logger.info(f"Done! Wrote {len(output_samples)} samples to {args.output_manifest}")
    logger.info(f"  With pause tags: {n_with_pauses}")
    logger.info(f"  Augmented: {n_augmented}")
    logger.info(f"  Plain (no tags): {len(output_samples) - n_with_pauses - n_augmented}")
    logger.info(f"  Total time: {elapsed/60:.1f} minutes ({len(samples)/elapsed:.1f} samples/s)")


if __name__ == "__main__":
    main()
