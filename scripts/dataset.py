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

        if not self.files:
            raise FileNotFoundError(f"No .pt files found in {preprocessed_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)

        text_tokens = data["text_tokens"][:self.max_text_len]
        speech_tokens = data["speech_tokens"][:self.max_speech_len]
        speaker_emb = data["speaker_emb"]       # (1, 256)
        cond_tokens = data["cond_tokens"]        # (375,)

        return {
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "speaker_emb": speaker_emb.squeeze(0),  # (256,)
            "cond_tokens": cond_tokens,
        }


# Turbo uses GPT-2 tokenizer, pad_token_id = eos_token_id = 50256
TEXT_PAD_TOKEN = 50256
SPEECH_PAD_TOKEN = 0
STOP_SPEECH_TOKEN = 6562


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
