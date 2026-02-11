"""
preprocess.py -- Offline preprocessing for Chatterbox-Turbo pause-tag fine-tuning.

Reads a JSONL manifest of (wav, text, ref_wav) and produces .pt files containing
pre-computed speech tokens, speaker embeddings, and conditioning tokens.

Optimizations over naive version:
  - Batched S3 tokenizer calls (target audio + conditioning) → 1 GPU call per batch
  - Batched VoiceEncoder via embeds_from_wavs (handles internal batching)
  - Threaded audio prefetching so CPU I/O overlaps with GPU compute

Usage:
    python scripts/preprocess.py \
        --manifest dataset/manifests/train.jsonl \
        --data_root dataset/ \
        --output_dir dataset/preprocessed/train \
        --batch_size 32 \
        --num_workers 8 \
        --device cuda
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
S3_SR = 16_000
ENC_COND_LEN = 15 * S3_SR       # 15s of 16kHz for T3 conditioning
SPEECH_COND_PROMPT_LEN = 375     # Turbo default
STOP_SPEECH_TOKEN = 6562


def load_models(ckpt_dir: str, device: str):
    """Load VoiceEncoder and S3Tokenizer (no T3 needed for preprocessing)."""
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.s3gen import S3Gen
    from safetensors.torch import load_file

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(Path(ckpt_dir) / "ve.safetensors"))
    ve.to(device).eval()

    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(load_file(Path(ckpt_dir) / "s3gen_meanflow.safetensors"), strict=True)
    s3gen.to(device).eval()
    s3_tokenizer = s3gen.tokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return ve, s3_tokenizer, tokenizer


def load_wav_16k(path: str) -> np.ndarray:
    """Load audio file and resample to 16kHz mono numpy array."""
    wav, _ = librosa.load(path, sr=S3_SR, mono=True)
    if len(wav) < S3_SR:
        wav = np.pad(wav, (0, S3_SR - len(wav)), mode="reflect")
    return wav


def load_sample_audio(sample: dict, data_root: Path) -> dict:
    """CPU work: load + resample target and reference audio. Runs in thread pool."""
    wav_path = str(data_root / sample["wav"])
    wav_16k = load_wav_16k(wav_path)

    ref_wav_path = str(data_root / sample.get("ref_wav", sample["wav"]))
    ref_16k = load_wav_16k(ref_wav_path)

    return {
        "id": sample["id"],
        "text": sample["text"],
        "wav_16k": wav_16k,
        "ref_16k": ref_16k,
    }


@torch.no_grad()
def process_batch(batch: list, ve, s3_tokenizer, text_tokenizer, device: str) -> list:
    """Batched GPU inference: S3 tokenizer + VoiceEncoder on a batch of loaded samples."""
    target_wavs = [s["wav_16k"] for s in batch]
    ref_wavs = [s["ref_16k"] for s in batch]
    ref_wavs_cond = [r[:ENC_COND_LEN] for r in ref_wavs]

    # Batched S3 tokenizer: target audio → speech tokens
    speech_tok_batch, speech_lens = s3_tokenizer.forward(target_wavs, max_len=None)

    # Batched S3 tokenizer: ref audio → conditioning tokens
    cond_tok_batch, cond_lens = s3_tokenizer.forward(ref_wavs_cond, max_len=SPEECH_COND_PROMPT_LEN)

    # Batched VoiceEncoder: ref audio → speaker embeddings
    ve_embeds = ve.embeds_from_wavs(ref_wavs, sample_rate=S3_SR)
    # ve_embeds is (N, n_partials, 256) → mean over partials → (N, 256)
    ve_embeds = torch.from_numpy(ve_embeds).float()

    results = []
    for i, sample in enumerate(batch):
        # Text tokens (CPU, fast)
        text_enc = text_tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=512)
        text_tokens = text_enc.input_ids.squeeze(0)

        # Speech tokens + stop token
        st = speech_tok_batch[i, :speech_lens[i]].cpu()
        st = torch.cat([st, torch.tensor([STOP_SPEECH_TOKEN], dtype=torch.long)])

        # Speaker embedding
        ve_emb = ve_embeds[i].unsqueeze(0)  # (1, 256)

        # Conditioning tokens, padded to SPEECH_COND_PROMPT_LEN
        cl = min(int(cond_lens[i]), SPEECH_COND_PROMPT_LEN)
        ct = cond_tok_batch[i, :cl].cpu()
        if ct.size(0) < SPEECH_COND_PROMPT_LEN:
            ct = torch.cat([ct, torch.zeros(SPEECH_COND_PROMPT_LEN - ct.size(0), dtype=torch.long)])

        results.append({
            "id": sample["id"],
            "text": sample["text"],
            "text_tokens": text_tokens,
            "speech_tokens": st,
            "speaker_emb": ve_emb,
            "cond_tokens": ct,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for Chatterbox T3 fine-tuning")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts/turbo")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8, help="Threads for audio loading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    logger.info("Loading models...")
    ve, s3_tokenizer, text_tokenizer = load_models(args.ckpt_dir, args.device)

    logger.info(f"Reading manifest: {args.manifest}")
    with open(args.manifest) as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]

    # Filter already-processed
    pending = [s for s in samples if not (output_dir / f"{s['id']}.pt").exists()]
    logger.info(f"Total: {len(samples)}, already done: {len(samples) - len(pending)}, pending: {len(pending)}")

    t0 = time.time()
    n_done = 0
    n_skip = 0
    speech_lens = []

    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    pbar = tqdm(total=len(pending), desc="Preprocessing")

    # Process in batches: submit audio loading for next batch while GPU processes current
    for batch_start in range(0, len(pending), args.batch_size):
        batch_samples = pending[batch_start : batch_start + args.batch_size]

        # Parallel audio loading on CPU threads
        futures = [pool.submit(load_sample_audio, s, data_root) for s in batch_samples]

        loaded = []
        for fut in futures:
            try:
                loaded.append(fut.result())
            except Exception as e:
                n_skip += 1
                logger.warning(f"Skip load: {e}")

        if not loaded:
            pbar.update(len(batch_samples))
            continue

        # Batched GPU inference
        try:
            results = process_batch(loaded, ve, s3_tokenizer, text_tokenizer, args.device)
            for r in results:
                torch.save(r, output_dir / f"{r['id']}.pt")
                speech_lens.append(r["speech_tokens"].size(0))
                n_done += 1
        except Exception as e:
            n_skip += len(loaded)
            logger.warning(f"Skip batch: {e}")

        pbar.update(len(batch_samples))

    pool.shutdown(wait=False)
    pbar.close()

    elapsed = time.time() - t0
    rate = n_done / elapsed if elapsed > 0 else 0
    logger.info(f"Done! {n_done} processed, {n_skip} skipped in {elapsed:.1f}s ({rate:.1f} samples/s)")
    if speech_lens:
        logger.info(f"Speech token lengths: min={min(speech_lens)}, max={max(speech_lens)}, "
                     f"mean={sum(speech_lens)/len(speech_lens):.0f}")


if __name__ == "__main__":
    main()
