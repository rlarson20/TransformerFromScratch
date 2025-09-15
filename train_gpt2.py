#!/usr/bin/env python3
"""
Training script for GPT-2 on Apple Silicon with WikiText-2
Optimized for MPS backend with memory management
"""

import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer

from models.GPT2 import GPT2ForCausalLM


class WikiTextDataset(Dataset):
    """Simple dataset wrapper for tokenized WikiText"""

    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize all texts and concatenate
        print("Tokenizing dataset...")
        all_tokens = []
        for text in texts:
            if text.strip():  # Skip empty strings
                tokens = tokenizer.encode(text.strip())
                all_tokens.extend(tokens)

        # Split into chunks of max_length
        self.examples = []
        for i in range(0, len(all_tokens) - max_length, max_length):
            chunk = all_tokens[i : i + max_length + 1]  # +1 for labels
            if len(chunk) == max_length + 1:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))

        print(f"Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def setup_device():
    """Setup optimal device for Apple Silicon"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def load_data(tokenizer, max_length=512, max_train_samples=None):
    """Load and preprocess WikiText-2 dataset"""
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    train_texts = dataset["train"]["text"]
    val_texts = dataset["validation"]["text"]

    # Limit samples for faster experimentation
    if max_train_samples:
        train_texts = train_texts[:max_train_samples]
        val_texts = val_texts[: max_train_samples // 10]

    train_dataset = WikiTextDataset(train_texts, tokenizer, max_length)
    val_dataset = WikiTextDataset(val_texts, tokenizer, max_length)

    return train_dataset, val_dataset


def create_model(vocab_size: int, device: torch.device) -> GPT2ForCausalLM:
    """Create small GPT-2 model for local training"""
    model = GPT2ForCausalLM(
        num_layers=6,
        vocab_size=vocab_size,
        hidden_size=384,
        num_heads=6,
        context_size=512,
        expand_size=1536,  # 4 * hidden_size
        embed_drop=0.1,
        attn_drop=0.1,
        out_drop=0.1,
        ffn_drop=0.1,
        head_norm=True,
        tie_weights=True,
        bias=True,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    return model.to(device)


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device, non_blocking=True)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)
        loss = outputs["loss"]

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Log progress
        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch} [{batch_idx:3d}/{len(dataloader):3d}] "
                f"Loss: {loss.item():.4f} "
                f"Time: {elapsed:.1f}s"
            )

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        outputs = model(batch)
        total_loss += outputs["loss"].item()

    return total_loss / len(dataloader)


@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt="The", max_length=100):
    """Generate text sample from model"""
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Generate
    for _ in range(max_length):
        # Get logits for last token
        outputs = model(tokens)
        logits = outputs["logits"][0, -1, :]

        # Sample next token (temperature=0.8)
        probs = torch.softmax(logits / 0.8, dim=-1)
        next_token = torch.multinomial(probs, 1)

        # Append to sequence
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

        # Stop if we hit context limit
        if tokens.size(1) >= model.context_size:
            break

    # Decode and return
    generated_text = tokenizer.decode(tokens[0].tolist())
    return generated_text


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )


def main():
    # Configuration
    config = {
        "batch_size": 8,  # Small batch for Apple Silicon memory
        "learning_rate": 3e-4,
        "epochs": 10,
        "max_length": 512,
        "max_train_samples": 5000,  # Limit for faster training
        "save_every": 2,
        "eval_every": 1,
        "checkpoint_dir": "./checkpoints",
    }

    # Setup
    device = setup_device()

    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_dataset, val_dataset = load_data(
        tokenizer, config["max_length"], config["max_train_samples"]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True if device.type == "mps" else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True if device.type == "mps" else False,
    )

    # Create model
    model = create_model(tokenizer.vocab_size, device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"] * len(train_loader)
    )

    # Training loop
    print(f"\nStarting training for {config['epochs']} epochs...")
    best_val_loss = float("inf")

    for epoch in range(1, config["epochs"] + 1):
        print(f"\n--- Epoch {epoch}/{config['epochs']} ---")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        if epoch % config["eval_every"] == 0:
            val_loss = evaluate(model, val_loader, device)
            perplexity = math.exp(val_loss)
            print(f"Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    val_loss,
                    f"{config['checkpoint_dir']}/best_model.pt",
                )
                print("Saved new best model!")

            # Generate sample
            sample = generate_sample(model, tokenizer, device, "The")
            print(f"Sample generation:\n{sample}\n")

        # Save checkpoint
        if epoch % config["save_every"] == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                f"{config['checkpoint_dir']}/checkpoint_epoch_{epoch}.pt",
            )

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
