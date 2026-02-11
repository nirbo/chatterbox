"""
train_pause.py -- LoRA fine-tuning of Chatterbox-Turbo T3 for pause tags.

Teaches the model to insert silence token runs when it sees [Xs] in text input.

Usage:
    python scripts/train_pause.py \
        --train_dir dataset/preprocessed/train \
        --val_dir dataset/preprocessed/val \
        --ckpt_dir ./ckpts/turbo \
        --output_dir ./output/pause_lora \
        --epochs 10 \
        --batch_size 4 \
        --lr 2e-4

Requires: pip install peft accelerate wandb (wandb optional)
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond

from dataset import ChatterboxT3Dataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_t3_turbo(ckpt_dir: str, device: str) -> T3:
    """Load pretrained Turbo T3 with the correct hyperparameters."""
    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False

    t3 = T3(hp)
    state = load_file(Path(ckpt_dir) / "t3_turbo_v1.safetensors")
    t3.load_state_dict(state)

    # Turbo deletes the backbone's native word embedding (T3 uses its own)
    del t3.tfmr.wte
    t3.to(device)
    return t3


def apply_lora(t3: T3, lora_r: int = 16, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """Wrap T3 with LoRA adapters on GPT-2 attention layers."""
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        # GPT-2 attention projection layers
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    model = get_peft_model(t3, config)
    model.print_trainable_parameters()
    return model


def build_t3_cond(batch: dict, device: str) -> T3Cond:
    """Construct T3Cond from a batch. Matches the official prepare_conditionals pattern."""
    speaker_emb = batch["speaker_emb"].to(device)        # (B, 256)
    cond_tokens = batch["cond_tokens"].to(device)         # (B, 375)

    return T3Cond(
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=cond_tokens,
        # Let T3.prepare_conditioning compute cond_prompt_speech_emb from tokens
        cond_prompt_speech_emb=None,
        emotion_adv=0.5 * torch.ones(speaker_emb.size(0), 1, 1, device=device),
        clap_emb=None,
    )


def compute_loss(t3_model, batch: dict, device: str):
    """
    Forward pass through T3 and compute cross-entropy loss.
    Uses T3's native loss() which handles concat + splice correctly.
    """
    text_tokens = batch["text_tokens"].to(device)
    text_token_lens = batch["text_token_lens"].to(device)
    speech_tokens = batch["speech_tokens"].to(device)
    speech_token_lens = batch["speech_token_lens"].to(device)

    t3_cond = build_t3_cond(batch, device)

    # peft wraps the module; get the underlying T3 which has .loss()
    base = t3_model.get_base_model() if hasattr(t3_model, "get_base_model") else t3_model

    loss_text, loss_speech = base.loss(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        text_token_lens=text_token_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_token_lens,
    )

    # Speech loss is the primary objective; text loss is auxiliary
    total_loss = loss_speech + 0.1 * loss_text
    return total_loss, loss_speech.item(), loss_text.item()


@torch.no_grad()
def validate(t3_model, val_loader, device):
    """Run validation and return average losses."""
    t3_model.eval()
    total_speech_loss = 0.0
    total_text_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        _, speech_loss, text_loss = compute_loss(t3_model, batch, device)
        total_speech_loss += speech_loss
        total_text_loss += text_loss
        n_batches += 1

    t3_model.train()
    if n_batches == 0:
        return 0.0, 0.0
    return total_speech_loss / n_batches, total_text_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune Chatterbox-Turbo T3 for pause tags")
    parser.add_argument("--train_dir", type=str, required=True, help="Preprocessed training data directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Preprocessed validation data (auto-split if None)")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts/turbo")
    parser.add_argument("--output_dir", type=str, default="./output/pause_lora")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_speech_len", type=int, default=1024, help="Max speech tokens per sample")
    parser.add_argument("--max_text_len", type=int, default=256, help="Max text tokens per sample")

    # LoRA hyperparameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every_epoch", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.05, help="Validation split if no val_dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="chatterbox-pause-tags")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B ──────────────────────────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Dataset ──────────────────────────────────────────────────────────
    logger.info("Loading dataset...")
    full_dataset = ChatterboxT3Dataset(
        args.train_dir,
        max_speech_len=args.max_speech_len,
        max_text_len=args.max_text_len,
    )

    if args.val_dir:
        val_dataset = ChatterboxT3Dataset(
            args.val_dir,
            max_speech_len=args.max_speech_len,
            max_text_len=args.max_text_len,
        )
        train_dataset = full_dataset
    else:
        val_size = max(1, int(len(full_dataset) * args.val_split))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # ── Model ────────────────────────────────────────────────────────────
    logger.info("Loading T3 Turbo...")
    t3 = build_t3_turbo(args.ckpt_dir, args.device)

    logger.info("Applying LoRA...")
    t3_lora = apply_lora(t3, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, t3_lora.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = torch.amp.GradScaler("cuda")

    # ── Training ─────────────────────────────────────────────────────────
    logger.info("Starting training...")
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t3_lora.train()
        epoch_speech_loss = 0.0
        epoch_text_loss = 0.0
        n_steps = 0

        for step, batch in enumerate(train_loader):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                total_loss, speech_loss, text_loss = compute_loss(t3_lora, batch, args.device)
                scaled_loss = total_loss / args.grad_accum

            scaler.scale(scaled_loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(t3_lora.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_speech_loss += speech_loss
            epoch_text_loss += text_loss
            n_steps += 1

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    f"[Epoch {epoch+1}/{args.epochs}][Step {step}/{len(train_loader)}] "
                    f"speech_loss={speech_loss:.4f} text_loss={text_loss:.4f} lr={lr:.2e}"
                )
                if args.wandb:
                    wandb.log({
                        "train/speech_loss": speech_loss,
                        "train/text_loss": text_loss,
                        "train/total_loss": total_loss.item(),
                        "train/lr": lr,
                        "global_step": global_step,
                    })

        # ── Epoch summary ────────────────────────────────────────────────
        avg_speech = epoch_speech_loss / max(n_steps, 1)
        avg_text = epoch_text_loss / max(n_steps, 1)
        logger.info(f"Epoch {epoch+1} train avg: speech_loss={avg_speech:.4f} text_loss={avg_text:.4f}")

        # ── Validation ───────────────────────────────────────────────────
        val_speech, val_text = validate(t3_lora, val_loader, args.device)
        logger.info(f"Epoch {epoch+1} val: speech_loss={val_speech:.4f} text_loss={val_text:.4f}")

        if args.wandb:
            wandb.log({
                "val/speech_loss": val_speech,
                "val/text_loss": val_text,
                "epoch": epoch + 1,
            })

        # ── Save checkpoint ──────────────────────────────────────────────
        if (epoch + 1) % args.save_every_epoch == 0:
            save_path = output_dir / f"epoch_{epoch+1}"
            t3_lora.save_pretrained(save_path)
            logger.info(f"Saved LoRA checkpoint: {save_path}")

        if val_speech < best_val_loss:
            best_val_loss = val_speech
            best_path = output_dir / "best"
            t3_lora.save_pretrained(best_path)
            logger.info(f"New best model (val_speech_loss={val_speech:.4f}): {best_path}")

    logger.info(f"Training complete. Best val speech loss: {best_val_loss:.4f}")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
