#!/usr/bin/env python3
"""
SFT Training Script using TRL SFTTrainer
=========================================
Trains Gemma3-141M on Turkish instruction datasets.
Each stage builds on the previous stage's model.

Usage:
    python train_sft.py --stage 0     # Run specific stage
    python train_sft.py --all         # Run all stages sequentially
    python train_sft.py --from-stage 2  # Start from stage 2
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# ================================================
# SFT STAGES CONFIGURATION (Increased for full training)
# ================================================
SFT_STAGES = [
    {
        "name": "sft-alpaca",
        "dataset": "cenfis/alpaca-turkish-combined",
        "max_steps": 2500,  # Reduced for <24h total
        "description": "Turkish Alpaca SFT (82K instruction-following)",
    },
    {
        "name": "sft-medical",
        "dataset": "matrixportalx/turkish_medical_alpaca",
        "max_steps": 1500,  # Reduced for <24h total
        "description": "Turkish Medical Q&A (21.6K)",
    },
    {
        "name": "sft-wiki",
        "dataset": "matrixportalx/wikipedia-alpaca-v2",
        "max_steps": 3000,  # Reduced for <24h total
        "description": "Turkish Wikipedia Q&A (319K)",
    },
    {
        "name": "sft-instructions",
        "dataset": "merve/turkish_instructions",
        "max_steps": 2000,  # Reduced for <24h total
        "description": "Turkish Instructions (51.6K)",
    },
]

# ================================================
# MODEL CONFIGURATION
# ================================================
# Continue from the latest successfully pushed model
# Robust Fallback: Stage 2 -> Stage 1 -> Base
STAGE2_MODEL = "alibayram/gemma3-141m-stage2-books"
STAGE1_MODEL = "alibayram/gemma3-141m-stage1-cosmos"
BASE_MODEL_NAME = "alibayram/gemma3-141m-cloned-mean"

def get_best_available_model():
    """Finds the best available checkpoint to start SFT from."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        # 1. Try Stage 2 (Books)
        try:
            api.model_info(STAGE2_MODEL)
            print(f"✅ Found Stage 2 model: {STAGE2_MODEL}")
            return STAGE2_MODEL
        except:
            print(f"⚠️ Stage 2 model not found ({STAGE2_MODEL})")
            
        # 2. Try Stage 1 (Cosmos)
        try:
            api.model_info(STAGE1_MODEL)
            print(f"✅ Found Stage 1 model: {STAGE1_MODEL}")
            return STAGE1_MODEL
        except:
            print(f"⚠️ Stage 1 model not found ({STAGE1_MODEL})")
            
        # 3. Fallback to Base
        print(f"🔙 Falling back to base model: {BASE_MODEL_NAME}")
        return BASE_MODEL_NAME
    except Exception as e:
        print(f"⚠️ Error checking Hub ({e}), using base model")
        return BASE_MODEL_NAME

PRETRAINED_MODEL = get_best_available_model()
BASE_TOKENIZER = "alibayram/gemma3-141m-cloned-mean"
HUB_PREFIX = "alibayram/gemma3-141m"

# ================================================
# A100 OPTIMIZED TRAINING SETTINGS
# ================================================
BATCH_SIZE = 8           # Per device (A100 80GB can handle more)
GRADIENT_ACCUMULATION = 8  # Effective batch size = 64
LEARNING_RATE = 2e-5     # Slightly higher for SFT
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# Turkish system prompt for instruction-following
SYSTEM_PROMPT = "Talimatları dikkatle takip eden ve yüksek kaliteli yanıtlar üreten yardımcı bir asistansın."

def format_instruction_turkish(example):
    """Format instruction examples into Turkish prompt format"""
    # Support both English and Turkish column names
    # Note: some datasets have leading spaces in column names (e.g., ' giriş', ' çıktı')
    instruction = (example.get("instruction") or example.get("talimat") or 
                   example.get(" talimat") or "")
    input_text = (example.get("input") or example.get("giriş") or 
                  example.get(" giriş") or "")
    output = (example.get("output") or example.get("çıktı") or 
              example.get(" çıktı") or "")
    
    if not instruction or not output:
        return {"text": ""}
    
    # Build user message
    if input_text and input_text.strip():
        user_msg = f"Talimat:\n{instruction}\n\nGirdi:\n{input_text}"
    else:
        user_msg = f"Talimat:\n{instruction}"
    
    # Build full prompt with Turkish format
    text = f"<bos>{SYSTEM_PROMPT}\n\n{user_msg}\n\nYanıt:\n{output}<eos>"
    return {"text": text}


def get_model_path(stage_idx, test_mode=False):
    """
    Get model path for stage - each stage uses previous stage's model.
    SFT Stage 0 uses the pretrained model.
    SFT Stage 1+ uses the previous SFT stage output.
    In test mode, uses -test suffix for model paths.
    """
    suffix = "-test" if test_mode else ""
    
    if stage_idx == 0:
        # First SFT stage uses the pretrained model
        # In test mode, try test version first, fallback to production
        if test_mode:
            test_pretrained = f"{PRETRAINED_MODEL}-test"
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.model_info(test_pretrained)
                print(f"✅ Using test pretrained model: {test_pretrained}")
                return test_pretrained
            except:
                print(f"⚠️ Test pretrained not found, using: {PRETRAINED_MODEL}")
        return PRETRAINED_MODEL
    
    # Subsequent stages use previous SFT stage
    prev_stage = SFT_STAGES[stage_idx - 1]
    local_path = f"gemma3-141m-{prev_stage['name']}{suffix}"
    hub_path = f"{HUB_PREFIX}-{prev_stage['name']}{suffix}"
    
    # Check local first
    if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "model.safetensors")):
        print(f"📂 Found local model: {local_path}")
        return local_path
    
    # Check Hub
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        # Check if model exists
        api.model_info(hub_path)
        print(f"☁️ Using Hub model: {hub_path}")
        return hub_path
    except Exception as e:
        print(f"⚠️ Could not find previous stage model: {e}")
    
    # Fallback to pretrained (not ideal but prevents crash)
    print(f"⚠️ Falling back to pretrained model: {PRETRAINED_MODEL}")
    return PRETRAINED_MODEL


# Test mode uses minimal steps for quick validation
TEST_MODE_STEPS = 10

def run_stage(stage_idx, test_mode=False):
    """Run a single SFT stage"""
    stage = SFT_STAGES[stage_idx]
    suffix = "-test" if test_mode else ""
    output_dir = f"gemma3-141m-{stage['name']}{suffix}"
    hub_model_id = f"{HUB_PREFIX}-{stage['name']}{suffix}"
    max_steps = TEST_MODE_STEPS if test_mode else stage['max_steps']
    
    print("=" * 70)
    mode_str = "(TEST MODE)" if test_mode else ""
    print(f"🎯 SFT STAGE {stage_idx + 1}/{len(SFT_STAGES)}: {stage['name'].upper()} {mode_str}")
    print("=" * 70)
    print(f"📚 Dataset: {stage['dataset']}")
    print(f"   {stage['description']}")
    print(f"📊 Max steps: {max_steps}")
    print(f"📦 Output: {output_dir}")
    print(f"☁️ Hub: {hub_model_id}")
    
    # Load and prepare dataset
    print("\n📥 Loading dataset...")
    dataset = load_dataset(stage["dataset"], split="train")
    print(f"   Raw samples: {len(dataset)}")
    
    dataset = dataset.map(format_instruction_turkish, num_proc=4)
    dataset = dataset.filter(lambda x: len(x["text"]) > 10)
    print(f"   After formatting: {len(dataset)}")
    
    # Get model path (uses previous stage's model, with -test suffix in test mode)
    model_path = get_model_path(stage_idx, test_mode=test_mode)
    print(f"\n🔧 Loading model from: {model_path}")
    
    # Load tokenizer from base (consistent tokenizer across stages)
    tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    
    # CRITICAL: Align model embeddings with tokenizer vocabulary size
    # This prevents CUDA device-side assert errors when token IDs exceed embedding table size
    original_vocab_size = model.config.vocab_size
    tokenizer_vocab_size = len(tokenizer)
    if original_vocab_size != tokenizer_vocab_size:
        print(f"⚠️ Vocabulary size mismatch: model={original_vocab_size}, tokenizer={tokenizer_vocab_size}")
        print(f"   Resizing model embeddings to match tokenizer...")
        model.resize_token_embeddings(tokenizer_vocab_size)
        print(f"   ✅ Embeddings resized: {original_vocab_size} → {tokenizer_vocab_size}")
    
    # Enable TF32 for faster training on A100
    torch.set_float32_matmul_precision('high')
    
    print(f"   Model loaded: {model.num_parameters():,} parameters")
    
    # SFT Config (A100 optimized, TRL 0.26.2 compatible)
    sft_config = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",  # Field containing formatted text
        
        # Batch settings
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        
        # Learning settings
        learning_rate=LEARNING_RATE,
        max_steps=max_steps,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        
        # Precision
        bf16=True,
        
        # Logging
        logging_steps=10,
        logging_first_step=True,
        
        # Saving
        save_steps=500,
        save_total_limit=3,
        
        # Hub
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="every_save",
        
        # Reporting
        report_to="wandb",
        run_name=f"gemma3-141m-{stage['name']}",
        
        # Other
        seed=42,
        dataloader_num_workers=4,
    )
    
    print("\n🚀 Starting training...")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )
    
    # Train
    trainer.train()
    
    # Save and push final model
    print("\n💾 Saving final model...")
    trainer.save_model()
    trainer.push_to_hub()
    
    print(f"\n✅ Stage {stage_idx + 1} complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Pushed to Hub: {hub_model_id}")
    
    # Clean up memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Turkish LLM")
    parser.add_argument("--stage", type=int, default=None, help="Run specific stage (0-3)")
    parser.add_argument("--all", action="store_true", help="Run all stages sequentially")
    parser.add_argument("--from-stage", type=int, default=0, help="Start from specific stage")
    parser.add_argument("--test-mode", action="store_true", 
                        help="Quick test run with 10 steps per stage (~5 min each)")
    args = parser.parse_args()
    
    print("=" * 70)
    mode_str = "(TEST MODE)" if args.test_mode else ""
    print(f"🚀 GEMMA3-141M TURKISH SFT TRAINING {mode_str}")
    print("=" * 70)
    print(f"📋 Total SFT stages: {len(SFT_STAGES)}")
    for i, s in enumerate(SFT_STAGES):
        steps = TEST_MODE_STEPS if args.test_mode else s['max_steps']
        print(f"   {i}: {s['name']} - {s['description']} ({steps} steps)")
    print()
    
    if args.all:
        print(f"▶️ Running all stages from {args.from_stage}...")
        for i in range(args.from_stage, len(SFT_STAGES)):
            success = run_stage(i, test_mode=args.test_mode)
            if not success:
                print(f"❌ Stage {i} failed, stopping.")
                break
        print("\n🎉 ALL SFT STAGES COMPLETE!")
        
    elif args.stage is not None:
        print(f"▶️ Running stage {args.stage}...")
        run_stage(args.stage, test_mode=args.test_mode)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
