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

## 2. Training Strategy

### What Exactly Happens

1. **Input**: Text with pause tags + reference audio (for speaker conditioning)
2. **Target**: Speech tokens where pause regions contain silence token runs
3. **Loss**: `loss_speech + 0.1 * loss_text` (cross-entropy on predicted tokens)

### LoRA Fine-Tune

Full fine-tune of 350M params risks catastrophic forgetting. Use LoRA on
the attention layers of the GPT-2 backbone:

- **Target modules**: `c_attn`, `c_proj` (GPT-2 attention projections)
- **Rank**: r=16, alpha=32, dropout=0.05
- **Trainable params**: ~2-4M (vs 350M total)

---

## 3. Environment Setup

### Hardware Requirements

- **GPU**: 24+ GB VRAM (tested on RTX 5090 32GB, Blackwell SM120)
- **CUDA**: 12.8+ for Blackwell, 12.1+ for Ampere/Hopper
- **Disk**: ~50GB for dataset + checkpoints
- **Python**: 3.11+

### Install Dependencies

```bash
cd /path/to/chatterbox
python -m venv venv
source venv/bin/activate

# Install chatterbox in editable mode
pip install -e .

# Training dependencies
pip install -r scripts/requirements-train.txt
```

**`scripts/requirements-train.txt`:**
```
peft>=0.14.0
accelerate>=1.2.0
whisperx
librosa>=0.11.0
soundfile
wandb           # optional
torch>=2.6.0
torchaudio>=2.6.0
transformers>=4.46.3
safetensors>=0.5.3
```

### Download Pretrained Weights

```bash
huggingface-cli download ResembleAI/chatterbox-turbo --local-dir ./ckpts/turbo
```

Or from Python:
```python
from chatterbox.tts_turbo import ChatterboxTurboTTS
model = ChatterboxTurboTTS.from_pretrained(device="cuda")
```

---

## 4. Full Pipeline

```
Step 1: Download dataset    →  dataset/wavs/ + dataset/manifests/train_raw.jsonl
Step 2: Generate pause data →  dataset/manifests/train.jsonl (with [Xs] tags)
Step 3: Preprocess          →  dataset/preprocessed/train/*.pt
Step 4: Train LoRA          →  output/pause_lora/best/
Step 5: Inference           →  output.wav
```

---

## 5. Scripts Reference

### 5.1 `download_peoples_speech.py` — Download Dataset

Streams People's Speech from HuggingFace, filters by duration, saves as WAV + JSONL manifest.

```bash
python scripts/download_peoples_speech.py \
    --n_samples 30000 \
    --output_dir dataset \
    --min_duration 2.0 \
    --max_duration 20.0
```

**Output:**
- `dataset/wavs/ps_000000.wav` ... `ps_029999.wav` (~13 GB)
- `dataset/manifests/train_raw.jsonl`

**Manifest format (one JSON per line):**
```json
{"id": "ps_000000", "wav": "wavs/ps_000000.wav", "text": "the actual transcript"}
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n_samples` | 30000 | Number of utterances to download |
| `--output_dir` | dataset | Root output directory |
| `--min_duration` | 2.0 | Min utterance duration (seconds) |
| `--max_duration` | 20.0 | Max utterance duration (seconds) |
| `--split` | train | HuggingFace dataset split |

---

### 5.2 `make_pause_data.py` — Generate Pause-Tagged Data

Force-aligns text to audio using WhisperX (wav2vec2-based alignment, no ASR needed since
transcripts already exist), identifies natural pauses between words, and inserts `[Xs]` tags.

```bash
python scripts/make_pause_data.py \
    --input_manifest dataset/manifests/train_raw.jsonl \
    --data_root dataset/ \
    --format jsonl \
    --output_manifest dataset/manifests/train.jsonl \
    --output_wavs dataset/wavs \
    --augment \
    --workers 8 \
    --device cuda
```

**Performance:** ~57 samples/s with 8 workers on RTX 5090 (30k input → 68,701 output in ~9 min).

**How it works:**
1. Each worker loads its own wav2vec2 alignment model (~360MB VRAM each)
2. For each sample: load audio → force-align text → find word gaps ≥ 150ms → insert `[Xs]` tags
3. Durations quantized to 0.25s steps (0.25, 0.5, 0.75 ... up to 3.0s)
4. Augmentation (30% probability): splice 0.25-2.0s silence at random word boundaries
5. Plain copies (no tags) included to preserve normal speech ability

**Output:** `dataset/manifests/train.jsonl` with ~2.3x input samples:
```json
{"id": "ps_000000", "wav": "wavs/ps_000000.wav", "text": "Hello [0.5s] world.", "ref_wav": "wavs/ps_000000.wav"}
{"id": "ps_000000_plain", "wav": "wavs/ps_000000.wav", "text": "Hello world.", "ref_wav": "wavs/ps_000000.wav"}
{"id": "ps_000000_aug", "wav": "wavs/ps_000000_aug.wav", "text": "Hello [0.5s] world [1.0s] today.", "ref_wav": "wavs/ps_000000.wav"}
```

**Also works with LJSpeech format:**
```bash
python scripts/make_pause_data.py \
    --input_dir /path/to/LJSpeech-1.1 \
    --format ljspeech \
    --output_manifest dataset/manifests/train.jsonl \
    --output_wavs dataset/wavs \
    --augment --workers 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input_manifest` | - | Input JSONL manifest (for `--format jsonl`) |
| `--input_dir` | - | LJSpeech directory (for `--format ljspeech`) |
| `--data_root` | . | Root for resolving relative wav paths in JSONL |
| `--format` | - | `jsonl` or `ljspeech` **(required)** |
| `--output_manifest` | - | Output JSONL path **(required)** |
| `--output_wavs` | - | Output WAV directory **(required)** |
| `--device` | cuda | GPU device for alignment |
| `--workers` | 8 | Worker processes (each loads own model, ~360MB VRAM each) |
| `--augment` | false | Enable silence-splicing augmentation |
| `--include_plain` | true | Include copies without pause tags |
| `--skip_alignment` | false | Skip alignment entirely (pass text unchanged) |
| `--seed` | 42 | Random seed for augmentation |

**PyTorch 2.6+ note:** Uses a `torch.load` monkeypatch for pyannote/whisperx compatibility
with `weights_only=True` default. See `_patch_torch_load()` in the source.

---

### 5.3 `preprocess.py` — Tokenize (Batched GPU)

Converts audio + text into pre-computed tensors for training. Runs S3 tokenizer, VoiceEncoder,
and GPT-2 BPE in batched mode with threaded audio prefetching.

```bash
python scripts/preprocess.py \
    --manifest dataset/manifests/train.jsonl \
    --data_root dataset/ \
    --output_dir dataset/preprocessed/train \
    --batch_size 32 \
    --num_workers 8 \
    --device cuda
```

**Performance:** ~116 samples/s with batch_size=32 on RTX 5090 (68k samples in ~10 min, 2.9 GB output).

**Optimizations over naive approach (3x-4x faster):**
- Batched S3 tokenizer: 1 GPU call per batch for target audio AND conditioning audio
- Batched VoiceEncoder: 1 call per batch for speaker embeddings
- 8 threads prefetch audio (librosa load + resample) while GPU processes current batch
- Resumable: skips samples where `{id}.pt` already exists

**Output per sample** (`dataset/preprocessed/train/{id}.pt`):
```python
{
    "id":            str,                          # "ps_000000"
    "text":          str,                          # "Hello [0.5s] world."
    "text_tokens":   LongTensor (T_text,),         # GPT-2 BPE token IDs
    "speech_tokens": LongTensor (T_speech,),       # S3 VQ tokens + stop token (6562)
    "speaker_emb":   FloatTensor (1, 256),         # VoiceEncoder speaker embedding
    "cond_tokens":   LongTensor (375,),            # Conditioning speech tokens (from first 15s ref)
}
```

**Typical speech token stats:** min=52, max=426, mean=355 (for 2-20s utterances).

| Flag | Default | Description |
|------|---------|-------------|
| `--manifest` | - | JSONL manifest **(required)** |
| `--data_root` | - | Root for wav paths **(required)** |
| `--output_dir` | - | Output directory for .pt files **(required)** |
| `--ckpt_dir` | ./ckpts/turbo | Pretrained Turbo weights |
| `--device` | cuda | GPU device |
| `--batch_size` | 32 | GPU batch size for S3 tokenizer / VoiceEncoder |
| `--num_workers` | 8 | Threads for audio loading |

---

### 5.4 `dataset.py` — PyTorch Dataset + Collation

Loads preprocessed `.pt` files for training. Used by `train_pause.py`.

```python
from dataset import ChatterboxT3Dataset, collate_fn

dataset = ChatterboxT3Dataset(
    "dataset/preprocessed/train",
    max_speech_len=1024,   # truncate long utterances
    max_text_len=256,
)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
```

**Padding:**
- Text: padded with GPT-2 pad token (50256)
- Speech: padded with 0
- Speaker embeddings + conditioning tokens: stacked (fixed-size)

**Batch dict keys:** `text_tokens`, `text_token_lens`, `speech_tokens`, `speech_token_lens`, `speaker_emb`, `cond_tokens`

---

### 5.5 `train_pause.py` — LoRA Training Loop

Fine-tunes T3 with LoRA adapters on GPT-2 attention layers.

```bash
python scripts/train_pause.py \
    --train_dir dataset/preprocessed/train \
    --ckpt_dir ./ckpts/turbo \
    --output_dir ./output/pause_lora \
    --epochs 10 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4
```

**Effective batch size:** `batch_size * grad_accum` = 16

**Training details:**
- LoRA on `c_attn` + `c_proj`: r=16, alpha=32, dropout=0.05
- Optimizer: AdamW (lr=2e-4, weight_decay=0.01)
- Scheduler: Cosine annealing over total optimizer steps
- Mixed precision: bf16 via `torch.amp`
- Gradient clipping: max_norm=1.0
- Auto 5% val split if no `--val_dir` provided
- Best model saved by lowest val speech loss
- W&B logging with `--wandb` flag

**Output:**
- `output/pause_lora/best/` — best checkpoint (by val speech loss)
- `output/pause_lora/epoch_N/` — per-epoch checkpoints

| Flag | Default | Description |
|------|---------|-------------|
| `--train_dir` | - | Preprocessed .pt directory **(required)** |
| `--val_dir` | None | Separate val set (auto-split 5% if omitted) |
| `--ckpt_dir` | ./ckpts/turbo | Pretrained Turbo weights |
| `--output_dir` | ./output/pause_lora | LoRA checkpoint output |
| `--epochs` | 10 | Training epochs |
| `--batch_size` | 4 | Per-device batch size |
| `--grad_accum` | 4 | Gradient accumulation steps |
| `--lr` | 2e-4 | Learning rate |
| `--max_speech_len` | 1024 | Max speech tokens per sample |
| `--max_text_len` | 256 | Max text tokens per sample |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha |
| `--lora_dropout` | 0.05 | LoRA dropout |
| `--val_split` | 0.05 | Auto val split ratio |
| `--log_every` | 10 | Log every N steps |
| `--save_every_epoch` | 1 | Save checkpoint every N epochs |
| `--wandb` | false | Enable W&B logging |
| `--wandb_project` | chatterbox-pause-tags | W&B project name |
| `--seed` | 42 | Random seed |

---

### 5.6 `inference_pause.py` — Generate Speech with Pauses

```bash
python scripts/inference_pause.py \
    --text "Hello [0.5s] world! [1.0s] This has pauses." \
    --ref_audio ref.wav \
    --lora_path ./output/pause_lora/best \
    --output out.wav
```

**How it works:**
1. Loads Chatterbox Turbo base model
2. Loads LoRA adapter and merges into base weights (faster inference)
3. Generates speech — model produces silence token (4299) runs at `[Xs]` positions
4. Saves 24kHz WAV

**Optional duration enforcement:**
```bash
python scripts/inference_pause.py \
    --text "Hello [0.5s] world!" \
    --ref_audio ref.wav \
    --lora_path ./output/pause_lora/best \
    --output out.wav \
    --enforce_durations
```

The `--enforce_durations` flag post-processes speech tokens to snap silence runs to the
exact durations from tags (model learns *where*, post-processing enforces *how long*).

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | - | Input text with `[Xs]` tags **(required)** |
| `--ref_audio` | - | Reference audio for voice cloning **(required)** |
| `--lora_path` | - | LoRA adapter directory **(required)** |
| `--output` | output.wav | Output WAV path |
| `--device` | cuda | GPU device |
| `--enforce_durations` | false | Snap silences to exact tag durations |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_k` | 1000 | Top-k sampling |
| `--top_p` | 0.95 | Nucleus sampling |

**Programmatic usage:**
```python
from peft import PeftModel
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="cuda")
model.t3 = PeftModel.from_pretrained(model.t3, "./output/pause_lora/best")
model.t3.merge_and_unload()

wav = model.generate("Hello [0.5s] world!", audio_prompt_path="ref.wav")
```

---

## 6. File Structure

```
scripts/
├── download_peoples_speech.py  # Step 1: Download dataset from HuggingFace
├── make_pause_data.py          # Step 2: Force-align + insert pause tags
├── preprocess.py               # Step 3: Tokenize audio → .pt (batched GPU)
├── dataset.py                  # PyTorch Dataset + collate_fn
├── train_pause.py              # Step 4: LoRA training loop
├── inference_pause.py          # Step 5: Generate speech with pauses
└── requirements-train.txt      # pip dependencies

dataset/
├── wavs/                       # Raw audio files (WAV)
├── manifests/
│   ├── train_raw.jsonl         # From download (no pause tags)
│   └── train.jsonl             # From make_pause_data (with [Xs] tags)
└── preprocessed/
    └── train/                  # .pt files from preprocess.py (~43KB each)

ckpts/
└── turbo/                      # Pretrained Turbo weights
    ├── t3_turbo_v1.safetensors
    ├── s3gen_meanflow.safetensors
    ├── ve.safetensors
    ├── t3_turbo_v1.yaml
    ├── tokenizer.json
    └── tokenizer_config.json

output/
└── pause_lora/                 # Training output
    ├── best/                   # Best LoRA checkpoint
    └── epoch_N/                # Per-epoch checkpoints
```

---

## 7. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Pauses too short/long | Not enough varied training data | Augment with `--augment` flag, range 0.25-3.0s |
| Forgets normal speech | Only trained on pause samples | `--include_plain` adds tag-free copies (default: on) |
| Garbage audio after pause | Silence runs too long for S3Gen | Cap pauses at 3.0s in training data |
| LoRA doesn't converge | Rank too low or LR wrong | Try r=32 or lower LR to 1e-4 |
| OOM during training | Sequences too long | Reduce `--max_speech_len` or `--batch_size` |
| `torch.load` errors (pyannote) | PyTorch 2.6+ `weights_only=True` default | `make_pause_data.py` has built-in monkeypatch |
| WhisperX language detection slow | Auto-detecting per sample | Fixed: hardcoded `language="en"` |
| Preprocessing fills disk | N/A (fixed) | Batched GPU preprocessing outputs ~43KB/sample, not raw audio |

---

## 8. Tips from LM Fine-Tuning

If you're coming from text LLM fine-tuning:

1. **Token rate is physical.** 25 tokens = 1 second of audio. Silence token 4299 repeated 25x = exactly 1s silence.
2. **Two loss terms.** `loss_speech` (primary) + `0.1 * loss_text` (auxiliary). Speech loss is what matters.
3. **Conditioning is critical.** Every sample needs reference audio for speaker identity. Bad ref = bad consistency.
4. **Listen, don't just watch loss.** Generate samples every few epochs. Loss alone doesn't measure perceptual quality.
5. **S3Gen/vocoder is frozen.** You only train T3. Token-to-audio is deterministic.

---

## 9. Quick Start (Copy-Paste)

```bash
source venv/bin/activate

# 1. Download 30k samples (~13 GB, ~30 min)
python scripts/download_peoples_speech.py --n_samples 30000 --output_dir dataset

# 2. Generate pause-tagged data (~9 min with 8 workers)
python scripts/make_pause_data.py \
    --input_manifest dataset/manifests/train_raw.jsonl \
    --data_root dataset/ --format jsonl \
    --output_manifest dataset/manifests/train.jsonl \
    --output_wavs dataset/wavs --augment --workers 8

# 3. Preprocess / tokenize (~10 min)
python scripts/preprocess.py \
    --manifest dataset/manifests/train.jsonl \
    --data_root dataset/ \
    --output_dir dataset/preprocessed/train \
    --batch_size 32 --num_workers 8

# 4. Train LoRA
python scripts/train_pause.py \
    --train_dir dataset/preprocessed/train \
    --output_dir ./output/pause_lora \
    --epochs 10 --batch_size 4 --grad_accum 4

# 5. Inference
python scripts/inference_pause.py \
    --text "Hello [0.5s] world!" \
    --ref_audio dataset/wavs/ps_000000.wav \
    --lora_path ./output/pause_lora/best \
    --output out.wav
```

---

## Appendix: Source Code Reference

| File | Purpose |
|------|---------|
| `src/chatterbox/models/t3/t3.py` | T3 model: `loss()`, `inference_turbo()` |
| `src/chatterbox/models/t3/modules/t3_config.py` | T3 hyperparameters |
| `src/chatterbox/models/t3/modules/cond_enc.py` | Conditioning encoder (T3Cond dataclass) |
| `src/chatterbox/models/t3/llama_configs.py` | GPT2_medium / Llama_520M configs |
| `src/chatterbox/models/s3tokenizer/` | Audio → speech tokens (VQ codec) |
| `src/chatterbox/models/s3gen/` | Speech tokens → mel → waveform |
| `src/chatterbox/models/s3gen/const.py` | `S3GEN_SIL = 4299` (silence token) |
| `src/chatterbox/models/voice_encoder/` | Speaker embedding extractor (256-d) |
| `src/chatterbox/tts_turbo.py` | Full Turbo inference pipeline |
