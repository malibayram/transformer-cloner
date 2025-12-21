"""
2-STAGE TRAINING PIPELINE FOR GEMMA3-141M ON B200
==================================================
Stage 1: ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0 (40GB - diverse web text)
Stage 2: alibayram/tr-books (high-quality book text)

This approach:
- Stage 1: Learns broad Turkish language patterns from diverse data
- Stage 2: Refines quality and style from curated book text
- Future Stage 3: SFT on question-answering datasets

Usage:
    python train_gemma3_141m.py --stage 0              # Run stage 0 (Cosmos)
    python train_gemma3_141m.py --stage 1              # Run stage 1 (Books)
    python train_gemma3_141m.py --stage 0 --test-mode  # Quick test with 10 steps
"""

import os
import argparse
import wandb
from huggingface_hub import login
from transformers import (
    GemmaTokenizerFast,
    Gemma3ForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch

# ========================
# CLI Arguments
# ========================
parser = argparse.ArgumentParser(description="Gemma3-141M Pretraining")
parser.add_argument("--stage", type=int, default=0, choices=[0, 1], 
                    help="Stage to run: 0=Cosmos, 1=Books")
parser.add_argument("--test-mode", action="store_true", 
                    help="Quick test run with 10 steps per stage (~5 min)")
args = parser.parse_args()

# Test mode uses minimal steps for quick validation
TEST_MODE_STEPS = 10

# ========================
# Configuration
# ========================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = "gemma3-141m-verda"
WANDB_ENTITY = "alibayram-ytu"

# ================================================
# TRAINING STAGES CONFIGURATION (Pretraining Only)
# SFT stages moved to train_sft.py
# ================================================
STAGES = [
    {
        "name": "stage1-cosmos",
        "dataset": "ytu-ce-cosmos/Cosmos-Turkish-Corpus-v1.0",
        "max_steps": 5000,  # Reduced for <24h total pipeline
        "learning_rate": 5e-5,
        "block_size": 2048,
        "batch_size": 32,  # A100 optimized
        "sft": False,
        "description": "Large diverse Turkish corpus (40GB)",
    },
    {
        "name": "stage2-books",
        "dataset": "alibayram/tr-books",
        "max_steps": 4000,  # Reduced for <24h total pipeline
        "learning_rate": 2e-5,
        "block_size": 2048,
        "batch_size": 32,  # A100 optimized
        "sft": False,
        "description": "High-quality Turkish books",
    },
]

# Select which stage to run from CLI
CURRENT_STAGE = args.stage

# ================================================
# B200 SETTINGS (192GB VRAM)
# ================================================
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch = batch_size * 2
MAX_STEPS = 5000  # Will be overridden by stage config
# Set num_workers to 0 to avoid potential hangs with StreamingTextDataset
# This makes data loading run in main process, preventing deadlock/pickling issues
NUM_WORKERS = 0  # Reduced to avoid empty batch errors with small datasets
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.1

# ========================
# Get Stage Configuration
# ========================
stage = STAGES[CURRENT_STAGE]
DATASET_NAME = stage["dataset"]
# Override max_steps in test mode
MAX_STEPS = TEST_MODE_STEPS if args.test_mode else stage["max_steps"]
LEARNING_RATE = stage["learning_rate"]
BLOCK_SIZE = stage["block_size"]
PER_DEVICE_TRAIN_BATCH_SIZE = stage["batch_size"]
PER_DEVICE_EVAL_BATCH_SIZE = stage["batch_size"]
WANDB_RUN_NAME = f"gemma3-141m-{stage['name']}" + ("-test" if args.test_mode else "")
OUTPUT_DIR = f"gemma3-141m-{stage['name']}" + ("-test" if args.test_mode else "")

# ========================
# Initialize
# ========================
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = WANDB_ENTITY
else:
    os.environ["WANDB_DISABLED"] = "true"

login(token=HF_TOKEN)

# ========================
# Load Dataset
# ========================
print("=" * 70)
print(f"🎯 STAGE {CURRENT_STAGE + 1}: {stage['name'].upper()}")
print("=" * 70)
print(f"📚 Dataset: {DATASET_NAME}")
print(f"   {stage['description']}")

# Check dataset size to decide on streaming
dataset = load_dataset(DATASET_NAME, split="train", streaming=True)
# Safe shuffle buffer for A100 (prevents OOM with large documents)
dataset = dataset.shuffle(buffer_size=100, seed=42)
USE_STREAMING = True  # Always stream for large datasets

print(f"   Mode: Streaming (memory efficient)")

# ========================
# Load Model & Tokenizer
# ========================
BASE_MODEL = "alibayram/gemma3-141m-cloned-mean"

# Determine model ID based on stage, test mode, and existing checkpoints
# In test mode, use -test suffix for model paths
model_id = None
suffix = "-test" if args.test_mode else ""

if CURRENT_STAGE == 0:
    # For Stage 0: Check if we have an existing checkpoint to resume from
    resume_path = f"alibayram/gemma3-141m-stage1-cosmos{suffix}"
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Check if model exists and is accessible
        info = api.model_info(resume_path)
        model_id = resume_path
        print(f"🔄 Resuming Stage 0 from existing Hub model: {resume_path}")
    except Exception as e:
        print(f"⚠️ Could not find/access {resume_path}: {e}")
        model_id = BASE_MODEL
        print(f"🆕 Starting Stage 0 from scratch (base model): {BASE_MODEL}")
else:
    # For Stage 1+: Try local checkpoint first, then Hub
    prev_stage = STAGES[CURRENT_STAGE - 1]
    local_path = f"gemma3-141m-{prev_stage['name']}{suffix}"
    hub_path = f"alibayram/gemma3-141m-{prev_stage['name']}{suffix}"
    
    import os
    if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "model.safetensors")):
        model_id = local_path
        print(f"📂 Found local checkpoint from previous stage: {local_path}")
    else:
        # Try HuggingFace Hub
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # Check if model exists
            api.model_info(hub_path)
            model_id = hub_path
            print(f"☁️ Using Hub model from previous stage: {hub_path}")
        except Exception as e:
            # Fallback to base model (unlikely to work well but safe fallback)
            print(f"⚠️ Previous stage model not found ({e}), using last known good model")
            # If resume path exists (e.g. stage 1 exists but stage 2 doesn't), use that
            # Otherwise base
            model_id = BASE_MODEL
    
print(f"🔧 Loading model from: {model_id}")

# Always load tokenizer from base model (avoids tokenizer save issues)
tokenizer = GemmaTokenizerFast.from_pretrained(BASE_MODEL, use_fast=True)

model = Gemma3ForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # Use SDPA instead of Flash Attention 2 (more stable with streaming)
    device_map="auto",
)

# CRITICAL: Align model embeddings with tokenizer vocabulary size
# This prevents CUDA device-side assert errors when token IDs exceed embedding table size
original_vocab_size = model.config.vocab_size
tokenizer_vocab_size = len(tokenizer)
if original_vocab_size != tokenizer_vocab_size:
    print(f"⚠️ Vocabulary size mismatch: model={original_vocab_size}, tokenizer={tokenizer_vocab_size}")
    print(f"   Resizing model embeddings to match tokenizer...")
    model.resize_token_embeddings(tokenizer_vocab_size)
    print(f"   ✅ Embeddings resized: {original_vocab_size} → {tokenizer_vocab_size}")
else:
    print(f"   ✅ Vocabulary size matches: {tokenizer_vocab_size}")

model.gradient_checkpointing_enable()

print(f"   Model: {model.num_parameters():,} parameters")
print(f"   Dtype: {model.dtype}")

# ========================
# Streaming Dataset
# ========================
from torch.utils.data import IterableDataset

class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset for large corpora (pretraining)"""
    
    def __init__(self, hf_dataset, tokenizer, block_size):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        # Loop indefinitely to implementation infinite streaming
        while True:
            # Shuffle handled by dataset.shuffle(buffer_size=100)
            for example in self.dataset:
                text = example.get("text") or example.get("content") or example.get("book_content") or ""
                if not text:
                    continue
                
                text = text + self.tokenizer.eos_token
                
                # Chunk text to prevent OOM on large documents (e.g. books)
                # Process in chunks of ~8KB characters (approx 2k tokens)
                chunk_len = 8192
                for i in range(0, len(text), chunk_len):
                    chunk = text[i : i + chunk_len]
                    tokens = self.tokenizer(
                        chunk, 
                        add_special_tokens=False,
                        return_attention_mask=False
                    )["input_ids"]
                    buffer.extend(tokens)
                
                    while len(buffer) >= self.block_size:
                        input_ids = torch.tensor(buffer[:self.block_size])
                        yield {
                            "input_ids": input_ids,
                            "attention_mask": torch.ones_like(input_ids),
                            "labels": input_ids.clone(),
                        }
                        buffer = buffer[self.block_size:]


# Create dataset (pretraining only - SFT is handled by train_sft.py)
print(f"📦 Creating streaming dataset with {BLOCK_SIZE} token blocks...")
train_dataset = StreamingTextDataset(dataset, tokenizer, BLOCK_SIZE)

# ========================
# Training Arguments
# ========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Batch settings
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,  # Effective batch = 32*2 = 64
    
    # Learning rate
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    
    # Duration - use max_steps for streaming datasets
    max_steps=MAX_STEPS,
    
    # B200 Optimizations
    bf16=True,
    tf32=True,
    dataloader_num_workers=NUM_WORKERS,
    dataloader_pin_memory=False,
    
    # Gradient checkpointing
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    torch_compile=False,
    
    # Data
    remove_unused_columns=False,
    
    # Logging
    logging_steps=100,
    eval_strategy="no",  # No eval for streaming
    save_steps=1000,  # Save every 1000 steps
    save_total_limit=5,  # Keep more checkpoints
    
    # wandb
    report_to="wandb" if WANDB_API_KEY else "none",
    run_name=WANDB_RUN_NAME,
    
    # Hub - push on EVERY save
    push_to_hub=True,
    hub_model_id=f"alibayram/{OUTPUT_DIR}",
    hub_strategy="every_save",  # Push to HF on every checkpoint
    
    # Optimizer
    optim="adamw_torch_fused",
    max_grad_norm=1.0,
    
    # Seed
    seed=42,
    data_seed=42,
)

# ========================
# Data Collator & Trainer
# ========================
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# ========================
# Print Summary
# ========================
print("\n" + "=" * 70)
print(f"🚀 TRAINING CONFIGURATION - STAGE {CURRENT_STAGE + 1}")
print("=" * 70)
print(f"   Stage: {stage['name']}")
print(f"   Dataset: {DATASET_NAME}")
print(f"   Loading from: {model_id}")
print("-" * 70)
print(f"   Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"   Block size: {BLOCK_SIZE} tokens")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Max steps: {MAX_STEPS:,}")
print(f"   Precision: BF16 + TF32 + Flash Attention 2")
print("=" * 70)
print(f"\n📤 Will push to: alibayram/gemma3-141m-{stage['name']}")
print("")

# ========================
# Train (with resume support)
# ========================
import os
checkpoint_dir = OUTPUT_DIR
last_checkpoint = None
if os.path.isdir(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(checkpoint_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"🔄 Resuming from checkpoint: {last_checkpoint}")

trainer.train(resume_from_checkpoint=last_checkpoint)

# ========================
# Push & Cleanup
# ========================
print(f"📤 Pushing to Hugging Face Hub as: gemma3-141m-{stage['name']}...")
trainer.push_to_hub()

if WANDB_API_KEY:
    wandb.finish()
    print("✅ wandb run finished!")

print(f"\n🎉 Stage {CURRENT_STAGE + 1} ({stage['name']}) complete!")
print(f"   Model saved to: alibayram/gemma3-141m-{stage['name']}")

if CURRENT_STAGE < len(STAGES) - 1:
    next_stage = STAGES[CURRENT_STAGE + 1]
    print(f"\n➡️  To run Stage {CURRENT_STAGE + 2}:")
    print(f"   1. Change CURRENT_STAGE = {CURRENT_STAGE + 1} in the script")
    print(f"   2. Run the script again")
    print(f"   Next stage: {next_stage['name']} - {next_stage['description']}")
else:
    print("\n✨ All stages complete! Your model is fully trained.")
