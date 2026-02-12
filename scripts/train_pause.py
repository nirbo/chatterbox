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
import shutil
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from rich.console import Console

from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond

from dataset import ChatterboxT3Dataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()

# Colorblind-friendly palette (dark-mode)
C_EPOCH = "bold cyan"
C_STEP = "bright_blue"
C_SPEECH = "#FF9E64"       # warm orange
C_TEXT = "#7AA2F7"          # soft blue
C_LR = "#BB9AF7"            # lavender
C_VAL = "#73DACA"           # teal
C_SAVE = "#E0AF68"          # gold
C_BEST = "#9ECE6A"          # lime green
C_DIM = "dim"


C_SPEED = "#F7768E"          # soft pink — tok/sec
C_ETA = "#C0CAF5"            # pale slate — ETA


def log_step(epoch, epochs, step, total_steps, speech_loss, text_loss, lr,
             tok_per_sec=None, eta_str=None):
    parts = (
        f"  [{C_EPOCH}]E {epoch}/{epochs}[/]"
        f"  [{C_STEP}]step {step}/{total_steps}[/]"
        f"  [{C_SPEECH}]speech[/] [{C_DIM}]=[/][{C_SPEECH}]{speech_loss:.4f}[/]"
        f"  [{C_TEXT}]text[/] [{C_DIM}]=[/][{C_TEXT}]{text_loss:.4f}[/]"
        f"  [{C_LR}]lr[/] [{C_DIM}]=[/][{C_LR}]{lr:.6f}[/]"
    )
    if tok_per_sec is not None:
        parts += f"  [{C_SPEED}]tok/s[/] [{C_DIM}]=[/][{C_SPEED}]{tok_per_sec:,.0f}[/]"
    if eta_str is not None:
        parts += f"  [{C_ETA}]ETA[/] [{C_DIM}]=[/][{C_ETA}]{eta_str}[/]"
    console.print(parts, highlight=False)


def log_epoch_summary(epoch, kind, speech_loss, text_loss):
    tag_color = C_VAL if kind == "val" else C_EPOCH
    console.print(
        f"  [{tag_color}]Epoch {epoch} {kind}[/]"
        f"  [{C_SPEECH}]speech[/] [{C_DIM}]=[/][{C_SPEECH}]{speech_loss:.4f}[/]"
        f"  [{C_TEXT}]text[/] [{C_DIM}]=[/][{C_TEXT}]{text_loss:.4f}[/]",
        highlight=False,
    )


def log_save(msg):
    console.print(f"  [{C_SAVE}]{msg}[/]", highlight=False)


def log_best(val_loss, path):
    console.print(
        f"  [{C_BEST}]New best[/]"
        f"  [{C_SPEECH}]val_speech[/] [{C_DIM}]=[/][{C_SPEECH}]{val_loss:.4f}[/]"
        f"  [{C_DIM}]->[/] [{C_SAVE}]{path}[/]",
        highlight=False,
    )


def save_checkpoint(path, t3_lora, optimizer, scheduler, scaler, epoch, global_step, best_val_loss):
    """Save full training state for resumption."""
    path = Path(path)
    # LoRA weights
    t3_lora.save_pretrained(path)
    # Training state
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_cpu": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
    }, path / "train_state.pt")


def load_checkpoint(path, t3_lora, optimizer, scheduler, scaler):
    """Restore full training state from checkpoint."""
    path = Path(path)
    # LoRA weights
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file as load_safetensors
    adapter_path = path / "adapter_model.safetensors"
    if adapter_path.exists():
        state = load_safetensors(str(adapter_path))
        set_peft_model_state_dict(t3_lora, state)
    # Training state
    state_path = path / "train_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"No train_state.pt in {path}")
    ckpt = torch.load(state_path, weights_only=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    torch.random.set_rng_state(ckpt["rng_cpu"])
    torch.cuda.set_rng_state(ckpt["rng_cuda"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["best_val_loss"]


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
    Bypasses T3.loss() which has a shape bug in F.cross_entropy (passes B,T,C
    but PyTorch expects B,C,T for 3D). We call forward() + reshape to 2D instead.
    """
    text_tokens = batch["text_tokens"].to(device)
    text_token_lens = batch["text_token_lens"].to(device)
    speech_tokens = batch["speech_tokens"].to(device)
    speech_token_lens = batch["speech_token_lens"].to(device)

    t3_cond = build_t3_cond(batch, device)

    # Call through the model wrapper (PeftModel / torch.compile) — not get_base_model()
    # which would bypass the compile graph. PEFT forwards all kwargs to T3.forward().
    out = t3_model(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        text_token_lens=text_token_lens,
        speech_tokens=speech_tokens,
        speech_token_lens=speech_token_lens,
        training=True,
    )

    # Masking logic from T3.loss() (t3.py:214-219)
    IGNORE_ID = -100
    len_text = text_tokens.size(1)
    len_speech = speech_tokens.size(1)
    mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]
    mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]
    masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
    masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)

    # Reshape to 2D: (B*T, C) logits + (B*T,) targets — avoids 3D shape ambiguity
    loss_text = F.cross_entropy(
        out.text_logits.reshape(-1, out.text_logits.size(-1)),
        masked_text.reshape(-1),
        ignore_index=IGNORE_ID,
    )
    loss_speech = F.cross_entropy(
        out.speech_logits.reshape(-1, out.speech_logits.size(-1)),
        masked_speech.reshape(-1),
        ignore_index=IGNORE_ID,
    )

    # Speech loss is the primary objective; text loss is auxiliary
    total_loss = loss_speech + 0.1 * loss_text
    return total_loss, loss_speech.detach(), loss_text.detach()


@torch.no_grad()
def validate(t3_model, val_loader, device):
    """Run validation and return average losses."""
    t3_model.eval()
    sum_speech = torch.zeros(1, device=device)
    sum_text = torch.zeros(1, device=device)
    n_batches = 0

    for batch in val_loader:
        _, speech_loss, text_loss = compute_loss(t3_model, batch, device)
        sum_speech += speech_loss
        sum_text += text_loss
        n_batches += 1

    t3_model.train()
    if n_batches == 0:
        return 0.0, 0.0
    # Single GPU sync at the end
    return (sum_speech / n_batches).item(), (sum_text / n_batches).item()


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
    parser.add_argument("--compile", type=str, default=None, metavar="MODE",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                        help="torch.compile mode (omit to disable)")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH",
                        help="Resume from checkpoint dir (e.g. ./output/pause_lora/epoch_3)")
    parser.add_argument("--save_every_steps", type=int, default=0, metavar="N",
                        help="Save a step checkpoint every N optimizer steps (0 = disabled)")
    parser.add_argument("--max_checkpoints", type=int, default=3, metavar="K",
                        help="Max step-checkpoints to keep (FIFO rotation)")
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
        collate_fn=collate_fn, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4,
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}, lr: {args.lr}")

    # ── Model ────────────────────────────────────────────────────────────
    logger.info("Loading T3 Turbo...")
    t3 = build_t3_turbo(args.ckpt_dir, args.device)

    logger.info("Applying LoRA...")
    t3_lora = apply_lora(t3, lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    if args.compile:
        logger.info(f"Compiling model with mode={args.compile}...")
        t3_lora = torch.compile(t3_lora, mode=args.compile, dynamic=True)

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, t3_lora.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = torch.amp.GradScaler("cuda")

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if args.resume:
        logger.info(f"Resuming from {args.resume}...")
        start_epoch, global_step, best_val_loss = load_checkpoint(
            args.resume, t3_lora, optimizer, scheduler, scaler)
        start_epoch += 1  # resume from NEXT epoch
        logger.info(f"Resumed: epoch={start_epoch}, global_step={global_step}, "
                     f"best_val_loss={best_val_loss:.4f}")

    # ── Training ─────────────────────────────────────────────────────────
    logger.info("Starting training...")
    total_train_steps = len(train_loader) * args.epochs
    train_t0 = time.time()
    step_ckpt_queue = deque()

    for epoch in range(start_epoch, args.epochs):
        t3_lora.train()
        epoch_speech_loss = torch.zeros(1, device=args.device)
        epoch_text_loss = torch.zeros(1, device=args.device)
        n_steps = 0
        log_t0 = time.time()
        log_tokens = torch.zeros(1, device=args.device)

        for step, batch in enumerate(train_loader):
            batch_tokens = batch["speech_token_lens"].sum() + batch["text_token_lens"].sum()

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

                # Step-based checkpointing
                if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                    step_path = output_dir / f"step_{global_step}"
                    save_checkpoint(step_path, t3_lora, optimizer, scheduler, scaler,
                                    epoch, global_step, best_val_loss)
                    step_ckpt_queue.append(step_path)
                    log_save(f"Step checkpoint: {step_path}")
                    # FIFO rotation: remove oldest if over limit
                    if len(step_ckpt_queue) > args.max_checkpoints:
                        oldest = step_ckpt_queue.popleft()
                        if oldest.exists():
                            shutil.rmtree(oldest)
                            log_save(f"Removed old checkpoint: {oldest}")

            epoch_speech_loss += speech_loss
            epoch_text_loss += text_loss
            n_steps += 1
            log_tokens += batch_tokens

            if step % args.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                # Single GPU sync point: pull losses + tokens for display
                sp_val = speech_loss.item()
                tx_val = text_loss.item()
                tok_count = log_tokens.item()
                # Throughput
                now = time.time()
                dt = now - log_t0
                tok_per_sec = tok_count / dt if dt > 0 else 0
                # ETA
                done_steps = epoch * len(train_loader) + step + 1
                elapsed = now - train_t0
                steps_per_sec = done_steps / elapsed if elapsed > 0 else 1
                remaining = (total_train_steps - done_steps) / steps_per_sec
                h, m = divmod(int(remaining), 3600)
                m, s = divmod(m, 60)
                eta_str = f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"
                # Reset per-window counters
                log_t0 = now
                log_tokens.zero_()

                log_step(epoch + 1, args.epochs, step, len(train_loader),
                         sp_val, tx_val, lr, tok_per_sec, eta_str)
                if args.wandb:
                    wandb.log({
                        "train/speech_loss": sp_val,
                        "train/text_loss": tx_val,
                        "train/total_loss": sp_val + 0.1 * tx_val,
                        "train/lr": lr,
                        "train/tok_per_sec": tok_per_sec,
                        "global_step": global_step,
                    })

        # ── Epoch summary ────────────────────────────────────────────────
        avg_speech = (epoch_speech_loss / max(n_steps, 1)).item()
        avg_text = (epoch_text_loss / max(n_steps, 1)).item()
        log_epoch_summary(epoch + 1, "train", avg_speech, avg_text)

        # ── Validation ───────────────────────────────────────────────────
        val_speech, val_text = validate(t3_lora, val_loader, args.device)
        log_epoch_summary(epoch + 1, "val", val_speech, val_text)

        if args.wandb:
            wandb.log({
                "val/speech_loss": val_speech,
                "val/text_loss": val_text,
                "epoch": epoch + 1,
            })

        # ── Save checkpoint ──────────────────────────────────────────────
        if (epoch + 1) % args.save_every_epoch == 0:
            save_path = output_dir / f"epoch_{epoch+1}"
            save_checkpoint(save_path, t3_lora, optimizer, scheduler, scaler,
                            epoch, global_step, best_val_loss)
            log_save(f"Saved checkpoint: {save_path}")

        if val_speech < best_val_loss:
            best_val_loss = val_speech
            best_path = output_dir / "best"
            save_checkpoint(best_path, t3_lora, optimizer, scheduler, scaler,
                            epoch, global_step, best_val_loss)
            log_best(val_speech, best_path)

    console.print(f"  [{C_BEST}]Training complete. Best val speech loss: {best_val_loss:.4f}[/]",
                  highlight=False)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
