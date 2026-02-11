# Fine-Tuning Chatterbox-Turbo T3 for Pause Tags

> Goal: Add `[0.5s]`, `[1.0s]`, etc. pause tags to Chatterbox-Turbo, so that
> `"Hello [0.5s] world!"` produces speech with a half-second silence gap.

---

## 1. Architecture Primer (What You're Actually Training)

Chatterbox is a **three-stage** pipeline. Only the first stage (T3) needs fine-tuning:

```
Text Tokens ──┐
               ├──► T3 (GPT-2 medium, 350M) ──► Speech Tokens (25 tok/s)
Speaker Emb ──┘           │
                          ▼
                     S3Gen (CFM, 2-step) ──► Mel ──► HiFTNet Vocoder ──► Audio
```

**T3** is an autoregressive transformer (GPT-2 medium: 24 layers, 1024 hidden, 16 heads)
that takes a concatenated sequence of `[conditioning | text_tokens | speech_tokens]` and
predicts the next speech token via cross-entropy. It produces **discrete audio codes** from
a VQ codebook of 6561 entries at 25 tokens/second.

**You do NOT need to touch S3Gen or the vocoder.** Those decode speech tokens to audio
deterministically. Your job is to teach T3 that when it sees `[0.5s]` in the text, it
should output ~12-13 silence tokens (token ID `4299`, at 25 tok/s x 0.5s).

### How Turbo's Existing Tags Work

Tags like `[laugh]`, `[sigh]` are **not** special tokens. They are plain text tokenized
by the GPT-2 BPE tokenizer (50,276 vocab). The model learned during pretraining that
these text subsequences should map to the corresponding audio patterns. Your pause tags
will work the same way.

### Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `S3_TOKEN_RATE` | 25 | Speech tokens per second |
| `S3GEN_SIL` | 4299 | Silence speech token ID |
| `start_speech_token` | 6561 | BOS for speech |
| `stop_speech_token` | 6562 | EOS for speech |
| `speech_tokens_dict_size` | 6563 | Speech vocab (0-6562) |
| `text_tokens_dict_size` | 50276 | GPT-2 BPE vocab |

---

## 2. The Training Approach

### What Exactly Happens

1. **Input**: Text with pause tags + reference audio (for speaker conditioning)
2. **Target**: Speech tokens where pause regions are replaced with silence token runs
3. **Loss**: Cross-entropy on predicted speech tokens (same as original training)

The `T3.loss()` method at `src/chatterbox/models/t3/t3.py:190` already implements this:
```python
def loss(self, *, t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens):
    out = self.forward(t3_cond=..., text_tokens=..., speech_tokens=..., training=True)
    loss_text = F.cross_entropy(out.text_logits, masked_text, ignore_index=-100)
    loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=-100)
    return loss_text, loss_speech
```

### Strategy: LoRA Fine-Tune (Recommended)

Full fine-tune of 350M params is overkill and risks catastrophic forgetting. Use LoRA on
the attention layers of the GPT-2 backbone. This keeps 99%+ of params frozen and trains
~2-4M new parameters.

---

## 3. Environment Setup

### 3.1 System Requirements

- **GPU**: RTX 5090 32GB (Blackwell SM120) -- more than enough
- **CUDA**: 12.8+ (for Blackwell/SM120 support)
- **Python**: 3.11+
- **Disk**: ~50GB for datasets, checkpoints, and logs

### 3.2 Install Dependencies

```bash
# Clone and install chatterbox in editable mode
cd /home/nir/ml-tools/chatterbox
pip install -e .

# Training dependencies (not included in chatterbox)
pip install peft>=0.14.0          # LoRA / parameter-efficient fine-tuning
pip install accelerate>=1.2.0     # Mixed-precision, gradient accumulation
pip install wandb                 # Experiment tracking (optional but recommended)
pip install datasets              # HuggingFace datasets (optional)
pip install webdataset            # For large-scale streaming datasets (optional)
```

### 3.3 Verify CUDA + GPU

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
print(f'PyTorch: {torch.__version__}')
"
```

### 3.4 Download Pretrained Weights

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS
# This downloads all components to HF cache
model = ChatterboxTurboTTS.from_pretrained(device="cuda")
```

Or manually:
```bash
huggingface-cli download ResembleAI/chatterbox-turbo --local-dir ./ckpts/turbo
```

---

## 4. Dataset Preparation

### 4.1 What You Need

Each training sample consists of:

| Field | Description |
|-------|-------------|
| `audio` | Waveform file (WAV, 16kHz mono preferred) |
| `text` | Transcript **with** pause tags, e.g. `"Hello [0.5s] world."` |
| `ref_audio` | Reference clip for speaker conditioning (5-15s, same speaker) |

### 4.2 Option A: Synthesize Pause Data from Existing TTS Data

If you have a clean TTS dataset (LJSpeech, LibriTTS, VCTK, your own):

1. **Force-align** text to audio using MFA (Montreal Forced Aligner) or WhisperX
2. **Identify natural pauses** (silences > 200ms between words)
3. **Insert pause tags** in the transcript at those locations
4. Optionally **augment** by splicing extra silence into existing audio and inserting
   corresponding tags

```python
"""
Pseudocode: Generate training pairs from aligned data.
"""
import torchaudio
from chatterbox.models.s3tokenizer import S3Tokenizer

s3tok = S3Tokenizer()  # or load from pretrained

def make_pause_sample(wav_path, alignment, tokenizer):
    wav, sr = torchaudio.load(wav_path)
    wav_16k = torchaudio.functional.resample(wav, sr, 16000)

    # Get speech tokens for entire utterance
    speech_tokens, _ = s3tok(wav_16k.unsqueeze(0).cuda())
    speech_tokens = speech_tokens.squeeze(0)  # (T,)

    text_with_tags = ""
    for i, word_info in enumerate(alignment):
        text_with_tags += word_info["word"]
        if i < len(alignment) - 1:
            gap = alignment[i+1]["start"] - word_info["end"]
            if gap >= 0.2:  # 200ms+ gap
                duration = round(gap, 1)
                text_with_tags += f" [{duration}s]"
        text_with_tags += " "

    return {
        "text": text_with_tags.strip(),
        "speech_tokens": speech_tokens,  # already has silence in right places
        "wav_16k": wav_16k,
    }
```

### 4.3 Option B: Splice Silence Directly into Speech Tokens

For more control, directly manipulate speech token sequences:

```python
SIL_TOKEN = 4299
TOKEN_RATE = 25  # tokens per second

def insert_pause_tokens(speech_tokens, pause_position_tok, pause_duration_s):
    """Insert silence tokens at a specific position in the speech token sequence."""
    n_sil = round(pause_duration_s * TOKEN_RATE)
    sil_block = torch.full((n_sil,), SIL_TOKEN, dtype=speech_tokens.dtype)
    return torch.cat([
        speech_tokens[:pause_position_tok],
        sil_block,
        speech_tokens[pause_position_tok:]
    ])
```

### 4.4 Dataset Format (On Disk)

Structure as a simple directory of JSON + WAV:

```
dataset/
  manifests/
    train.jsonl     # one JSON object per line
    val.jsonl
  wavs/
    utt_0001.wav
    utt_0002.wav
    ...
```

Each line in `train.jsonl`:
```json
{
  "id": "utt_0001",
  "wav": "wavs/utt_0001.wav",
  "text": "The quick brown fox [0.3s] jumped over the lazy dog.",
  "ref_wav": "wavs/utt_0001.wav",
  "speaker": "speaker_01"
}
```

### 4.5 Data Volume Guidelines

| Scenario | Data Needed | Expected Quality |
|----------|-------------|-----------------|
| Proof of concept | 100-500 samples | Learns the concept, timing rough |
| Usable | 2,000-5,000 samples | Good timing, some drift |
| Production | 10,000+ samples | Precise pause control |

Mix 50/50 with samples **without** pause tags to avoid forgetting normal speech.

---

## 5. Training Scripts

All scripts are in the `scripts/` directory. The pipeline has four stages:

### 5.1 Generate Pause-Tag Data

```bash
# From LJSpeech-style dataset (uses WhisperX for forced alignment):
python scripts/make_pause_data.py \
    --input_dir /path/to/LJSpeech-1.1 \
    --format ljspeech \
    --output_manifest dataset/manifests/train.jsonl \
    --output_wavs dataset/wavs \
    --augment \
    --device cuda

# From existing JSONL manifest:
python scripts/make_pause_data.py \
    --input_manifest /path/to/manifest.jsonl \
    --data_root /path/to/data \
    --format jsonl \
    --output_manifest dataset/manifests/train.jsonl \
    --output_wavs dataset/wavs \
    --augment \
    --device cuda
```

See `scripts/make_pause_data.py` for details. It:
- Force-aligns text to audio using WhisperX
- Detects natural pauses (>150ms gaps between words)
- Inserts `[Xs]` pause tags at those locations (quantized to 0.25s steps)
- Optionally augments by splicing extra silence
- Includes plain (no-tag) copies to prevent catastrophic forgetting

### 5.2 Offline Preprocessing

```bash
python scripts/preprocess.py \
    --manifest dataset/manifests/train.jsonl \
    --data_root dataset/ \
    --output_dir dataset/preprocessed/train \
    --ckpt_dir ./ckpts/turbo \
    --device cuda
```

See `scripts/preprocess.py`. Produces `.pt` files with pre-computed speech tokens,
speaker embeddings, and conditioning tokens. This avoids re-tokenizing audio every epoch.

### 5.3 Train

```bash
python scripts/train_pause.py \
    --train_dir dataset/preprocessed/train \
    --ckpt_dir ./ckpts/turbo \
    --output_dir ./output/pause_lora \
    --epochs 10 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --wandb  # optional
```

See `scripts/train_pause.py`. Key design decisions (informed by community projects):
- Uses T3's **native `loss()` method** for correct concat/splice logic
- LoRA targets `c_attn` and `c_proj` (GPT-2 attention layers)
- Sets `cond_prompt_speech_emb=None` so T3 computes it internally
- `emotion_adv=0.5` (Turbo default, no perceiver/CFG)
- bf16 autocast, gradient clipping, cosine LR schedule
- Saves best checkpoint by validation speech loss

### 5.4 Inference

```bash
python scripts/inference_pause.py \
    --text "Hello [0.5s] world!" \
    --ref_audio ref.wav \
    --lora_path ./output/pause_lora/best \
    --output out.wav
```

See `scripts/inference_pause.py`. Also includes `enforce_pause_durations()` for
exact-duration post-processing (hybrid approach).

### Dataset and Collation

See `scripts/dataset.py` for the `ChatterboxT3Dataset` class and `collate_fn`.
Pads text with GPT-2's pad token (50256), speech with 0.

---

## 7. Inference with Fine-Tuned Model

```python
from peft import PeftModel
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Load base model
model = ChatterboxTurboTTS.from_pretrained(device="cuda")

# Wrap T3 with LoRA adapter
model.t3 = PeftModel.from_pretrained(model.t3, "./output/pause_lora/epoch_9")
model.t3.merge_and_unload()  # Optional: merge LoRA into base for faster inference

# Generate with pause tag
wav = model.generate("Hello [0.5s] world!", audio_prompt_path="ref.wav")
```

---

## 8. What Can Go Wrong & Mitigations

| Problem | Cause | Fix |
|---------|-------|-----|
| Pauses too short/long | Not enough training data with varied durations | Augment with 0.1s-3.0s range |
| Forgets normal speech | Trained only on pause samples | Mix 50% normal samples without tags |
| Garbage audio after pause | S3Gen confused by silence runs | Keep pauses <= 3s; add natural silence transitions |
| LoRA doesn't converge | Rank too low or LR wrong | Try r=32, or lower LR to 1e-4 |
| OOM on 32GB | Sequences too long | Reduce max_speech_tokens, use gradient checkpointing |

---

## 9. Key Differences from LM Training (Things to Know)

If you're coming from text LLM fine-tuning:

1. **Token rate matters physically.** In LMs, token count is abstract. Here, 25 tokens =
   1 second of audio. Silence token 4299 repeated 25 times = exactly 1 second of silence.

2. **Two loss terms.** T3 has both `loss_text` (auxiliary, predicts text tokens) and
   `loss_speech` (primary, predicts audio tokens). Weight speech loss much higher (~10:1).

3. **Conditioning is critical.** Every sample needs a reference audio clip for speaker
   identity. Bad conditioning = bad speaker consistency.

4. **You must listen, not just watch loss.** Generate samples every N steps and check if
   pauses sound right. Loss alone does not tell you about perceptual quality.

5. **The S3Gen/vocoder is frozen.** You only train T3. Speech tokens to audio is
   deterministic given the same tokens, so if T3 outputs the right silence tokens, the
   audio will be correct.

---

## 10. Quick-Start Checklist

```
[ ] 1. Install chatterbox + training deps (Section 3.2)
[ ] 2. Download pretrained weights (Section 3.4)
[ ] 3. Prepare aligned dataset with pause tags (Section 4)
[ ] 4. Verify tokenizer handles your pause tags:
       tokenizer = AutoTokenizer.from_pretrained("./ckpts/turbo")
       print(tokenizer("[0.5s]"))  # should produce valid token IDs
[ ] 5. Write collate_fn (Section 6)
[ ] 6. Run training (Section 5.2)
[ ] 7. Listen to samples every few epochs
[ ] 8. Load LoRA adapter and test (Section 7)
```

---

## Appendix: File Reference

| File | Purpose |
|------|---------|
| `src/chatterbox/models/t3/t3.py` | T3 model with `loss()` and `inference_turbo()` |
| `src/chatterbox/models/t3/modules/t3_config.py` | All T3 hyperparameters |
| `src/chatterbox/models/t3/modules/cond_enc.py` | Conditioning encoder (T3Cond) |
| `src/chatterbox/models/t3/llama_configs.py` | GPT2_medium / Llama_520M configs |
| `src/chatterbox/models/s3tokenizer/` | Audio to speech tokens (VQ codec) |
| `src/chatterbox/models/s3gen/` | Speech tokens to mel to waveform |
| `src/chatterbox/models/s3gen/const.py` | `S3GEN_SIL = 4299` (silence token) |
| `src/chatterbox/models/voice_encoder/` | Speaker embedding extractor |
| `src/chatterbox/tts_turbo.py` | Full turbo inference pipeline |
