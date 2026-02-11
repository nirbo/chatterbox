"""
preprocess_gpu.py -- Batched GPU inference for Chatterbox preprocessing.

Reads .bin files produced by the Rust CPU stage (preprocess_rs), runs S3 tokenizer
and VoiceEncoder on GPU in batches, and saves final .pt files for training.

Usage:
    python scripts/preprocess_gpu.py \
        --bin_dir dataset/preprocessed/cpu \
        --output_dir dataset/preprocessed/train \
        --ckpt_dir ./ckpts/turbo \
        --batch_size 32 \
        --device cuda
"""

import argparse
import logging
import struct
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPEECH_COND_PROMPT_LEN = 375
S3_SR = 16_000
STOP_SPEECH_TOKEN = 6562


def read_bin(path: Path) -> dict:
    """Read a .bin file produced by the Rust CPU stage."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"CPR1", f"Bad magic in {path}: {magic}"

        id_len = struct.unpack("<I", f.read(4))[0]
        sample_id = f.read(id_len).decode("utf-8")

        text_len = struct.unpack("<I", f.read(4))[0]
        text = f.read(text_len).decode("utf-8")

        n_tokens = struct.unpack("<I", f.read(4))[0]
        text_tokens = torch.from_numpy(
            np.frombuffer(f.read(n_tokens * 4), dtype=np.uint32).astype(np.int64).copy()
        )

        n_audio = struct.unpack("<I", f.read(4))[0]
        audio = np.frombuffer(f.read(n_audio * 4), dtype=np.float32).copy()

        n_ref = struct.unpack("<I", f.read(4))[0]
        ref_audio = np.frombuffer(f.read(n_ref * 4), dtype=np.float32).copy()

    return {
        "id": sample_id,
        "text": text,
        "text_tokens": text_tokens,
        "audio_16k": audio,
        "ref_16k": ref_audio,
    }


def load_models(ckpt_dir: str, device: str):
    """Load VoiceEncoder and S3 tokenizer for GPU inference."""
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.s3gen import S3Gen
    from safetensors.torch import load_file

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(Path(ckpt_dir) / "ve.safetensors"))
    ve.to(device).eval()

    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(
        load_file(Path(ckpt_dir) / "s3gen_meanflow.safetensors"), strict=True
    )
    s3gen.to(device).eval()

    return ve, s3gen.tokenizer


@torch.no_grad()
def process_batch(batch: list, ve, s3_tokenizer, device: str) -> list:
    """Run batched GPU inference on a list of samples."""
    audios = [s["audio_16k"] for s in batch]
    refs = [s["ref_16k"] for s in batch]

    # Batched: speech tokens from target audio
    speech_tok_batch, speech_lens = s3_tokenizer.forward(audios, max_len=None)

    # Batched: conditioning tokens from ref audio
    cond_tok_batch, cond_lens = s3_tokenizer.forward(refs, max_len=SPEECH_COND_PROMPT_LEN)

    results = []
    for i, sample in enumerate(batch):
        # Speech tokens + stop token
        st = speech_tok_batch[i, : speech_lens[i]].cpu()
        st = torch.cat([st, torch.tensor([STOP_SPEECH_TOKEN], dtype=torch.long)])

        # Speaker embedding (per-sample, VoiceEncoder returns variable-size embeddings)
        ve_embed = ve.embeds_from_wavs([refs[i]], sample_rate=S3_SR)
        ve_embed = torch.from_numpy(ve_embed).float().mean(dim=0, keepdim=True)

        # Conditioning tokens, padded to SPEECH_COND_PROMPT_LEN
        cl = min(int(cond_lens[i]), SPEECH_COND_PROMPT_LEN)
        ct = cond_tok_batch[i, :cl].cpu()
        if ct.size(0) < SPEECH_COND_PROMPT_LEN:
            ct = torch.cat(
                [ct, torch.zeros(SPEECH_COND_PROMPT_LEN - ct.size(0), dtype=torch.long)]
            )

        results.append(
            {
                "id": sample["id"],
                "text": sample["text"],
                "text_tokens": sample["text_tokens"],
                "speech_tokens": st,
                "speaker_emb": ve_embed,
                "cond_tokens": ct,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Batched GPU preprocessing for Chatterbox")
    parser.add_argument("--bin_dir", required=True, help="Dir of .bin files from Rust stage")
    parser.add_argument("--output_dir", required=True, help="Output dir for .pt files")
    parser.add_argument("--ckpt_dir", default="./ckpts/turbo")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading GPU models...")
    ve, s3_tokenizer = load_models(args.ckpt_dir, args.device)

    # Scan for .bin files, skip already-processed
    bin_files = sorted(Path(args.bin_dir).glob("*.bin"))
    pending = [f for f in bin_files if not (output_dir / f"{f.stem}.pt").exists()]
    logger.info(f"Found {len(bin_files)} bins, {len(pending)} need GPU processing")

    t0 = time.time()
    n_done = 0
    stats_lens = []

    pbar = tqdm(total=len(pending), desc="GPU inference")
    for batch_start in range(0, len(pending), args.batch_size):
        batch_files = pending[batch_start : batch_start + args.batch_size]

        batch = []
        for bf in batch_files:
            try:
                batch.append(read_bin(bf))
            except Exception as e:
                logger.warning(f"Skip {bf.stem}: {e}")

        if not batch:
            continue

        results = process_batch(batch, ve, s3_tokenizer, args.device)

        for r in results:
            torch.save(r, output_dir / f"{r['id']}.pt")
            stats_lens.append(r["speech_tokens"].size(0))
            n_done += 1

        pbar.update(len(batch))

    pbar.close()
    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    logger.info(f"Done! {n_done} samples in {elapsed:.1f}s ({rate:.1f} samples/s)")
    if stats_lens:
        logger.info(
            f"Speech token lengths: min={min(stats_lens)}, max={max(stats_lens)}, "
            f"mean={sum(stats_lens) / len(stats_lens):.0f}"
        )


if __name__ == "__main__":
    main()
