"""
preprocess.py -- Offline preprocessing for Chatterbox-Turbo pause-tag fine-tuning.

Reads a JSONL manifest of (wav, text, ref_wav) and produces .pt files containing
pre-computed speech tokens, speaker embeddings, and conditioning tokens.

Usage:
    python scripts/preprocess.py \
        --manifest dataset/manifests/train.jsonl \
        --data_root dataset/ \
        --output_dir dataset/preprocessed/train \
        --device cuda

Manifest format (one JSON object per line):
    {"id": "utt_001", "wav": "wavs/utt_001.wav", "text": "Hello [0.5s] world.", "ref_wav": "wavs/utt_001.wav"}

If ref_wav is omitted, wav is used as the reference (self-conditioning).
"""

import argparse
import json
import logging
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
S3GEN_SR = 24_000
ENC_COND_LEN = 15 * S3_SR       # 15s of 16kHz for T3 conditioning
SPEECH_COND_PROMPT_LEN = 375     # Turbo default


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
    wav, sr = librosa.load(path, sr=S3_SR, mono=True)
    if len(wav) < S3_SR:
        wav = np.pad(wav, (0, S3_SR - len(wav)), mode="reflect")
    return wav


@torch.no_grad()
def preprocess_sample(sample: dict, data_root: Path, ve, s3_tokenizer, text_tokenizer, device: str):
    """
    Preprocess a single sample into tensors ready for training.

    Returns dict with:
        text_tokens:    LongTensor (T_text,)
        speech_tokens:  LongTensor (T_speech,)
        speaker_emb:    FloatTensor (1, 256)
        cond_tokens:    LongTensor (SPEECH_COND_PROMPT_LEN,)
    """
    # ── Text tokenization ────────────────────────────────────────────────
    text = sample["text"]
    text_enc = text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    text_tokens = text_enc.input_ids.squeeze(0)

    # ── Load target audio ────────────────────────────────────────────────
    wav_path = str(data_root / sample["wav"])
    wav_16k = load_wav_16k(wav_path)

    # ── Speech tokens from target audio ──────────────────────────────────
    speech_tokens, token_lens = s3_tokenizer.forward([wav_16k], max_len=None)
    speech_tokens = speech_tokens.squeeze(0)[:token_lens[0]].cpu()

    # Append stop token
    stop_speech = torch.tensor([6562], dtype=torch.long)
    speech_tokens = torch.cat([speech_tokens, stop_speech])

    # ── Reference audio for conditioning ─────────────────────────────────
    ref_wav_path = str(data_root / sample.get("ref_wav", sample["wav"]))
    ref_16k = load_wav_16k(ref_wav_path)

    # Speaker embedding
    ve_embed = ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)
    ve_embed = torch.from_numpy(ve_embed).float().mean(dim=0, keepdim=True)

    # Conditioning prompt tokens (first ENC_COND_LEN samples)
    ref_for_cond = ref_16k[:ENC_COND_LEN]
    cond_tokens, _ = s3_tokenizer.forward([ref_for_cond], max_len=SPEECH_COND_PROMPT_LEN)
    cond_tokens = cond_tokens.squeeze(0)[:SPEECH_COND_PROMPT_LEN].cpu()

    # Pad if shorter than expected
    if cond_tokens.size(0) < SPEECH_COND_PROMPT_LEN:
        pad_len = SPEECH_COND_PROMPT_LEN - cond_tokens.size(0)
        cond_tokens = torch.cat([cond_tokens, torch.zeros(pad_len, dtype=torch.long)])

    return {
        "id": sample["id"],
        "text": text,
        "text_tokens": text_tokens,
        "speech_tokens": speech_tokens,
        "speaker_emb": ve_embed,
        "cond_tokens": cond_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset for Chatterbox T3 fine-tuning")
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSONL manifest")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing wav files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save .pt files")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts/turbo", help="Path to pretrained Turbo weights")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    logger.info("Loading models...")
    ve, s3_tokenizer, text_tokenizer = load_models(args.ckpt_dir, args.device)

    logger.info(f"Reading manifest: {args.manifest}")
    with open(args.manifest) as f:
        samples = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"Preprocessing {len(samples)} samples...")
    stats = {"total": 0, "skipped": 0, "speech_token_lens": []}

    for sample in tqdm(samples, desc="Preprocessing"):
        out_path = output_dir / f"{sample['id']}.pt"
        if out_path.exists():
            stats["total"] += 1
            continue

        try:
            result = preprocess_sample(sample, data_root, ve, s3_tokenizer, text_tokenizer, args.device)
            torch.save(result, out_path)
            stats["total"] += 1
            stats["speech_token_lens"].append(result["speech_tokens"].size(0))
        except Exception as e:
            logger.warning(f"Skipping {sample['id']}: {e}")
            stats["skipped"] += 1

    logger.info(f"Done. Processed: {stats['total']}, Skipped: {stats['skipped']}")
    if stats["speech_token_lens"]:
        lens = stats["speech_token_lens"]
        logger.info(f"Speech token lengths: min={min(lens)}, max={max(lens)}, "
                     f"mean={sum(lens)/len(lens):.0f}")


if __name__ == "__main__":
    main()
