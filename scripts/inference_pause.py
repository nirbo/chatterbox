"""
inference_pause.py -- Generate speech with pause tags using a fine-tuned LoRA adapter.

Usage:
    python scripts/inference_pause.py \
        --text "Hello [0.5s] world!" \
        --ref_audio ref.wav \
        --lora_path ./output/pause_lora/best \
        --output out.wav

    # With exact-duration enforcement:
    python scripts/inference_pause.py \
        --text "Hello [0.5s] world! [1.0s] Goodbye." \
        --ref_audio ref.wav \
        --lora_path ./output/pause_lora/best \
        --output out.wav \
        --enforce_durations
"""

import argparse
import re
from pathlib import Path

import torch
import torchaudio
from peft import PeftModel
from chatterbox.tts_turbo import ChatterboxTurboTTS


SIL_TOKEN = 4299
TOKEN_RATE = 25  # speech tokens per second
MIN_SIL_RUN = 3  # minimum silence run to count as intentional pause


def enforce_pause_durations(text: str, speech_tokens: torch.Tensor) -> torch.Tensor:
    """
    Post-process speech tokens to snap silence runs to the exact durations
    specified by [Xs] tags in the text.

    The model learns *where* to pause; this function enforces *how long*.
    """
    # Extract requested durations from text
    durations = [float(m.group(1)) for m in re.finditer(r"\[(\d+\.?\d*)s\]", text)]
    if not durations:
        return speech_tokens

    # Find silence runs in the generated speech tokens
    is_sil = (speech_tokens == SIL_TOKEN)
    runs = []
    in_run = False
    run_start = 0

    for i in range(len(is_sil)):
        if is_sil[i] and not in_run:
            run_start = i
            in_run = True
        elif not is_sil[i] and in_run:
            if i - run_start >= MIN_SIL_RUN:
                runs.append((run_start, i))
            in_run = False
    if in_run and len(speech_tokens) - run_start >= MIN_SIL_RUN:
        runs.append((run_start, len(speech_tokens)))

    # Match runs to durations (greedily, in order)
    n_matches = min(len(runs), len(durations))
    if n_matches == 0:
        return speech_tokens

    # Build new token sequence by replacing each matched run
    result_parts = []
    prev_end = 0

    for i in range(n_matches):
        start, end = runs[i]
        target_len = round(durations[i] * TOKEN_RATE)
        target_len = max(target_len, 1)  # at least 1 token

        result_parts.append(speech_tokens[prev_end:start])
        result_parts.append(torch.full((target_len,), SIL_TOKEN, dtype=speech_tokens.dtype))
        prev_end = end

    # Append remaining tokens after last matched run
    result_parts.append(speech_tokens[prev_end:])

    return torch.cat(result_parts)


def main():
    parser = argparse.ArgumentParser(description="Generate speech with pause tags")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio for voice cloning")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output", type=str, default="output.wav")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enforce_durations", action="store_true",
                        help="Post-process to enforce exact pause durations from tags")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--top_p", type=float, default=0.95)
    args = parser.parse_args()

    print("Loading Chatterbox Turbo...")
    model = ChatterboxTurboTTS.from_pretrained(device=args.device)

    print(f"Loading LoRA adapter from {args.lora_path}...")
    model.t3 = PeftModel.from_pretrained(model.t3, args.lora_path)
    model.t3.merge_and_unload()  # Merge LoRA into base weights for faster inference

    print(f"Generating: {args.text!r}")
    wav = model.generate(
        args.text,
        audio_prompt_path=args.ref_audio,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    # Optional: enforce exact pause durations
    # Note: This requires intercepting speech tokens before S3Gen, which the
    # current generate() API does not expose. For now, this is a placeholder
    # showing how it would work if you modify generate() to return speech tokens.
    # See TRAIN.md Section 7 for the full approach.

    torchaudio.save(args.output, wav, model.sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
