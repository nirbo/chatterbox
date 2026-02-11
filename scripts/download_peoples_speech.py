"""
download_peoples_speech.py -- Stream a subset of People's Speech from HuggingFace.

Downloads N samples and saves as WAV + JSONL manifest compatible with make_pause_data.py.

Usage:
    python scripts/download_peoples_speech.py \
        --n_samples 30000 \
        --output_dir dataset \
        --min_duration 2.0 \
        --max_duration 20.0
"""

import argparse
import io
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def decode_audio(audio_dict):
    """Decode audio from HF datasets raw format (bytes) using soundfile."""
    if "array" in audio_dict and audio_dict["array"] is not None:
        return np.array(audio_dict["array"], dtype=np.float32), audio_dict["sampling_rate"]

    if "bytes" in audio_dict and audio_dict["bytes"] is not None:
        audio_bytes = audio_dict["bytes"]
        wav_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return wav_array, sr

    if "path" in audio_dict and audio_dict["path"] is not None:
        wav_array, sr = sf.read(audio_dict["path"], dtype="float32")
        return wav_array, sr

    raise ValueError(f"Cannot decode audio from keys: {list(audio_dict.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Download People's Speech subset")
    parser.add_argument("--n_samples", type=int, default=30000)
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument("--min_duration", type=float, default=2.0,
                        help="Minimum utterance duration in seconds")
    parser.add_argument("--max_duration", type=float, default=20.0,
                        help="Maximum utterance duration in seconds")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wavs"
    manifest_dir = output_dir / "manifests"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "train_raw.jsonl"

    logger.info(f"Streaming People's Speech (target: {args.n_samples} samples)...")

    # Disable automatic audio decoding — we handle it ourselves via soundfile
    ds = load_dataset(
        "MLCommons/peoples_speech",
        "clean",
        split=args.split,
        streaming=True,
    )
    ds = ds.cast_column("audio", datasets_audio_feature_raw())

    count = 0
    skipped = 0

    with open(manifest_path, "w") as manifest_file:
        for sample in ds:
            if count >= args.n_samples:
                break

            try:
                wav_array, sr = decode_audio(sample["audio"])
            except Exception as e:
                skipped += 1
                continue

            text = sample.get("text", sample.get("sentence", "")).strip()

            # Filter by duration
            duration = len(wav_array) / sr
            if duration < args.min_duration or duration > args.max_duration:
                skipped += 1
                continue

            # Filter: must have text
            if not text or len(text) < 5:
                skipped += 1
                continue

            # Convert stereo to mono if needed
            if wav_array.ndim > 1:
                wav_array = wav_array.mean(axis=1)

            utt_id = f"ps_{count:06d}"
            wav_path = wav_dir / f"{utt_id}.wav"

            sf.write(str(wav_path), wav_array, sr)

            manifest_file.write(json.dumps({
                "id": utt_id,
                "wav": f"wavs/{utt_id}.wav",
                "text": text,
            }) + "\n")

            count += 1

            if count % 1000 == 0:
                logger.info(f"Downloaded {count}/{args.n_samples} (skipped {skipped})")

    logger.info(f"Done. Saved {count} samples to {output_dir}")
    logger.info(f"Skipped {skipped} samples (duration/text filters)")
    logger.info(f"Manifest: {manifest_path}")

    total_size = sum(f.stat().st_size for f in wav_dir.glob("*.wav"))
    logger.info(f"Total WAV size: {total_size / 1e9:.1f} GB")


def datasets_audio_feature_raw():
    """Return an Audio feature that doesn't auto-decode (returns raw bytes)."""
    from datasets.features import Audio
    return Audio(decode=False)


if __name__ == "__main__":
    main()
