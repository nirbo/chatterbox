"""
dataset.py -- PyTorch Dataset and collation for Chatterbox T3 fine-tuning.

Loads preprocessed .pt files from the preprocess.py output directory.
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ChatterboxT3Dataset(Dataset):
    """
    Loads pre-computed .pt files containing:
        text_tokens:    LongTensor (T_text,)
        speech_tokens:  LongTensor (T_speech,)
        speaker_emb:    FloatTensor (1, 256)
        cond_tokens:    LongTensor (375,)
    """

    def __init__(self, preprocessed_dir: str, max_speech_len: int = 1024, max_text_len: int = 256):
        self.files = sorted(Path(preprocessed_dir).glob("*.pt"))
        self.max_speech_len = max_speech_len
        self.max_text_len = max_text_len

        # Pre-cache single-element tensors (avoids re-creation every __getitem__)
        self._bot = torch.tensor([BOT_TOKEN], dtype=torch.long)
        self._eot = torch.tensor([EOT_TOKEN], dtype=torch.long)
        self._start_speech = torch.tensor([START_SPEECH_TOKEN], dtype=torch.long)

        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {preprocessed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)

        # Truncate before wrapping so max_len accounts for special tokens
        text_tokens = data["text_tokens"][:self.max_text_len - 2]
        speech_tokens = data["speech_tokens"][:self.max_speech_len - 1]
        speaker_emb = data["speaker_emb"]       # (1, 256)
        cond_tokens = data["cond_tokens"]        # (375,)

        # Wrap text with BOT (255) and EOT (0) — required by T3._ensure_BOT_EOT
        text_tokens = torch.cat([self._bot, text_tokens, self._eot])

        # Prepend start_speech (6561) — stop_speech (6562) already appended by preprocessing
        speech_tokens = torch.cat([self._start_speech, speech_tokens])

        return {
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "speaker_emb": speaker_emb.squeeze(0),  # (256,)
            "cond_tokens": cond_tokens,
        }


# T3-internal special tokens (from T3Config)
BOT_TOKEN = 255           # start_text_token: prepended to text
EOT_TOKEN = 0             # stop_text_token: appended to text
START_SPEECH_TOKEN = 6561 # prepended to speech
STOP_SPEECH_TOKEN = 6562  # appended to speech (already done in preprocessing)

# Padding tokens
TEXT_PAD_TOKEN = 50256     # GPT-2 eos_token_id
SPEECH_PAD_TOKEN = 0


def collate_fn(batch):
    """
    Pad variable-length text and speech tokens to batch max.
    Returns a dict ready for the training loop.
    """
    text_lens = torch.tensor([s["text_tokens"].size(0) for s in batch])
    speech_lens = torch.tensor([s["speech_tokens"].size(0) for s in batch])

    # Pad sequences
    text_padded = pad_sequence(
        [s["text_tokens"] for s in batch],
        batch_first=True,
        padding_value=TEXT_PAD_TOKEN,
    )
    speech_padded = pad_sequence(
        [s["speech_tokens"] for s in batch],
        batch_first=True,
        padding_value=SPEECH_PAD_TOKEN,
    )

    # Stack fixed-size tensors
    speaker_embs = torch.stack([s["speaker_emb"] for s in batch])  # (B, 256)
    cond_tokens = torch.stack([s["cond_tokens"] for s in batch])   # (B, 375)

    return {
        "text_tokens": text_padded,
        "text_token_lens": text_lens,
        "speech_tokens": speech_padded,
        "speech_token_lens": speech_lens,
        "speaker_emb": speaker_embs,
        "cond_tokens": cond_tokens,
    }
