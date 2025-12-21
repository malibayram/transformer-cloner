#!/usr/bin/env python3
"""
Cross-Tokenizer Cloning Test Script

Tests cloning models with a custom tokenizer (alibayram/gemma3-tr-v32k)
and validates text generation works correctly.

This tests the core `clone()` functionality which maps embeddings from
the original tokenizer to a new tokenizer.

Usage:
    python scripts/test_cross_tokenizer_cloning.py
    python scripts/test_cross_tokenizer_cloning.py --hf-token YOUR_TOKEN
"""

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import login

# Add src to path for local development
sys.path.insert(0, "src")

from transformer_cloner import TransformerCloner, EmbeddingStrategy


@dataclass
class TestResult:
    """Result of testing a single model."""
    model_name: str
    model_id: str
    load_success: bool
    clone_success: bool
    generation_success: bool
    save_success: bool = False
    error_message: Optional[str] = None
    load_time: float = 0.0
    clone_time: float = 0.0
    output_path: Optional[str] = None
    param_count: int = 0


# Models to test with the custom tokenizer
MODELS_TO_TEST = [
    {
        "name": "SmolLM2-135M-Instruct",
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "description": "SmolLM2 Instruct (135M params)",
    },
    {
        "name": "Qwen3-0.6B",
        "model_id": "Qwen/Qwen3-0.6B",
        "description": "Qwen3 small (0.6B params)",
    },
    {
        "name": "Llama-3.2-1B",
        "model_id": "meta-llama/Llama-3.2-1B",
        "description": "Llama 3.2 (1B params, gated)",
    },
    {
        "name": "Ministral-3-14B",
        "model_id": "mistralai/Ministral-3-14B-Instruct-2512",
        "description": "Ministral 3 (14B params, gated)",
    },
    {
        "name": "Gemma-3-270m",
        "model_id": "google/gemma-3-270m",
        "description": "Google Gemma 3 (270M params)",
    },
]

# Target tokenizer
TARGET_TOKENIZER = "alibayram/gemma3-tr-v32k"

# Test prompts - include Turkish text to test the Turkish tokenizer
TEST_PROMPTS = [
    # English
    {"name": "English greeting", "text": "Hello, how are you today?"},
    # Turkish
    {"name": "Turkish greeting", "text": "Merhaba, nasılsın bugün?"},
    # Turkish sentence
    {"name": "Turkish sentence", "text": "Türkiye'nin başkenti Ankara'dır."},
    # Code (universal)
    {"name": "Python code", "text": "def merhaba():\n    return"},
    # Mixed
    {"name": "Mixed text", "text": "The answer is: Cevap şudur:"},
]


def setup_hf_auth(token: Optional[str] = None) -> Optional[str]:
    """Setup HuggingFace authentication."""
    if token:
        print("🔑 Using provided HuggingFace token...")
        login(token=token)
        return token
    
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        print("🔑 Using HuggingFace token from environment variable...")
        login(token=env_token)
        return env_token
    
    print("\n🔐 HuggingFace Authentication")
    print("Some models require authentication.")
    print("You can get your token from: https://huggingface.co/settings/tokens\n")
    
    token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    if token:
        login(token=token)
        return token
    
    print("⚠️  No token provided. Gated models may fail to load.")
    return None


def test_generation(model, tokenizer, prompts: list, max_new_tokens: int = 20) -> tuple[int, int]:
    """Test generation with multiple prompts. Returns (success_count, total_count)."""
    success = 0
    total = len(prompts)
    
    for prompt_info in prompts:
        name = prompt_info["name"]
        text = prompt_info["text"]
        
        try:
            inputs = tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            
            # Check for OOV tokens
            max_id = input_ids.max().item()
            vocab_size = model.config.vocab_size
            
            if max_id >= vocab_size:
                print(f"      ⚠️  {name}: Token ID {max_id} >= vocab {vocab_size}")
                continue
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
                )
            
            generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            print(f"      ✅ {name}: {repr(generated[:40])}{'...' if len(generated) > 40 else ''}")
            success += 1
        except Exception as e:
            print(f"      ❌ {name}: {str(e)[:50]}")
    
    return success, total


def test_model(
    model_info: dict,
    target_tokenizer_id: str,
    output_dir: str,
    hf_token: Optional[str],
    strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN,
) -> TestResult:
    """Test cloning a model with a custom tokenizer."""
    name = model_info["name"]
    model_id = model_info["model_id"]
    
    result = TestResult(
        model_name=name,
        model_id=model_id,
        load_success=False,
        clone_success=False,
        generation_success=False,
    )
    
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"Source: {model_id}")
    print(f"Target Tokenizer: {target_tokenizer_id}")
    print(f"{'='*70}")
    
    # Step 1: Load model and create cloner
    print("\n[1/4] Loading model and tokenizers...")
    try:
        start_time = time.time()
        cloner = TransformerCloner(
            org_model_id=model_id,
            target_tokenizer_id=target_tokenizer_id,
            token=hf_token,
        )
        result.load_time = time.time() - start_time
        result.load_success = True
        
        orig_vocab = len(cloner.org_tokenizer)
        target_vocab = len(cloner.target_tokenizer)
        print(f"   ✅ Loaded in {result.load_time:.1f}s")
        print(f"   Original vocab: {orig_vocab}, Target vocab: {target_vocab}")
    except Exception as e:
        result.error_message = f"Load failed: {str(e)}"
        print(f"   ❌ Failed: {e}")
        if "gated" in str(e).lower() or "auth" in str(e).lower():
            print("   ℹ️  This model requires authentication")
        return result
    
    # Step 2: Clone with new tokenizer
    print("\n[2/4] Cloning with new tokenizer...")
    try:
        start_time = time.time()
        cloned_model = cloner.clone(strategy=strategy, verbose=True)
        result.clone_time = time.time() - start_time
        result.clone_success = True
        
        param_count = cloned_model.num_parameters()
        result.param_count = param_count
        print(f"   ✅ Cloned in {result.clone_time:.1f}s")
        print(f"   Model params: {param_count:,} ({param_count/1e6:.1f}M)")
        print(f"   Model vocab size: {cloned_model.config.vocab_size}")
    except Exception as e:
        result.error_message = f"Clone failed: {str(e)}"
        print(f"   ❌ Clone failed: {e}")
        traceback.print_exc()
        return result
    
    # Step 3: Test generation
    print("\n[3/4] Testing generation...")
    try:
        success, total = test_generation(
            cloned_model, 
            cloner.target_tokenizer, 
            TEST_PROMPTS,
        )
        result.generation_success = success == total
        print(f"   Generation: {success}/{total} prompts succeeded")
    except Exception as e:
        print(f"   ❌ Generation test failed: {e}")
        result.error_message = f"Generation failed: {str(e)}"
    
    # Step 4: Save model
    print("\n[4/4] Saving model...")
    safe_name = name.replace("/", "-").replace(" ", "-").lower()
    tokenizer_name = target_tokenizer_id.split("/")[-1]
    output_name = f"{safe_name}-{tokenizer_name}"
    local_path = os.path.join(output_dir, output_name)
    
    try:
        os.makedirs(local_path, exist_ok=True)
        cloned_model.save_pretrained(local_path)
        cloner.target_tokenizer.save_pretrained(local_path)
        result.save_success = True
        result.output_path = local_path
        print(f"   ✅ Saved to: {local_path}")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")
        if not result.error_message:
            result.error_message = f"Save failed: {str(e)}"
    
    # Cleanup
    del cloned_model
    del cloner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def print_summary(results: list[TestResult]):
    """Print summary table."""
    print("\n")
    print("=" * 85)
    print("CROSS-TOKENIZER CLONING TEST RESULTS")
    print(f"Target Tokenizer: {TARGET_TOKENIZER}")
    print("=" * 85)
    
    print(f"\n{'Model':<25} {'Params':<12} {'Load':<6} {'Clone':<7} {'Gen':<6} {'Save':<6} {'Notes'}")
    print("-" * 100)
    
    for r in results:
        load = "✅" if r.load_success else "❌"
        clone = "✅" if r.clone_success else ("❌" if r.load_success else "-")
        gen = "✅" if r.generation_success else ("❌" if r.clone_success else "-")
        save = "✅" if r.save_success else ("❌" if r.clone_success else "-")
        params = f"{r.param_count/1e6:.1f}M" if r.param_count > 0 else "-"
        
        if r.save_success:
            notes = f"OK ({r.load_time:.1f}s + {r.clone_time:.1f}s)"
        elif r.error_message:
            notes = r.error_message[:30] + "..." if len(r.error_message) > 30 else r.error_message
        else:
            notes = ""
        
        print(f"{r.model_name:<25} {params:<12} {load:<6} {clone:<7} {gen:<6} {save:<6} {notes}")
    
    print("-" * 100)
    
    total = len(results)
    cloned = sum(1 for r in results if r.clone_success)
    gen_ok = sum(1 for r in results if r.generation_success)
    
    print(f"\nSummary: {cloned}/{total} cloned, {gen_ok}/{total} generation OK")


def main():
    parser = argparse.ArgumentParser(
        description="Test cloning models with a custom tokenizer",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./cloned_models_tr",
        help="Directory to save cloned models (default: ./cloned_models_tr)",
    )
    parser.add_argument(
        "--hf-token", "-t",
        type=str,
        default=None,
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--skip-auth",
        action="store_true",
        help="Skip authentication prompt",
    )
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="mean",
        choices=["mean", "sum", "first", "last", "weighted", "max", "min"],
        help="Embedding strategy (default: mean)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Test only a specific model by name (e.g., 'SmolLM2-135M-Instruct')",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    
    args = parser.parse_args()
    
    # Handle --list-models
    if args.list_models:
        print("Available models:")
        for m in MODELS_TO_TEST:
            print(f"  - {m['name']}: {m['description']}")
        return 0
    
    # Filter models if --model is specified
    models_to_run = MODELS_TO_TEST
    if args.model:
        models_to_run = [m for m in MODELS_TO_TEST if m['name'].lower() == args.model.lower()]
        if not models_to_run:
            print(f"❌ Model '{args.model}' not found")
            print("Available models:")
            for m in MODELS_TO_TEST:
                print(f"  - {m['name']}")
            return 1
    
    print("=" * 85)
    print("CROSS-TOKENIZER CLONING TEST")
    print("=" * 85)
    print(f"\nTarget Tokenizer: {TARGET_TOKENIZER}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Strategy: {args.strategy}")
    print(f"\nModels to test ({len(models_to_run)}):")
    for m in models_to_run:
        print(f"  - {m['name']}: {m['description']}")
    
    print(f"\nTest prompts ({len(TEST_PROMPTS)}):")
    for p in TEST_PROMPTS:
        print(f"  - {p['name']}: {repr(p['text'][:30])}...")
    
    # Auth
    hf_token = None
    if not args.skip_auth:
        hf_token = setup_hf_auth(args.hf_token)
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get strategy enum
    strategy = EmbeddingStrategy(args.strategy)
    
    # Test each model
    results = []
    for model_info in models_to_run:
        try:
            result = test_model(
                model_info=model_info,
                target_tokenizer_id=TARGET_TOKENIZER,
                output_dir=args.output_dir,
                hf_token=hf_token,
                strategy=strategy,
            )
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            results.append(TestResult(
                model_name=model_info["name"],
                model_id=model_info["model_id"],
                load_success=False,
                clone_success=False,
                generation_success=False,
                error_message=str(e),
            ))
    
    print_summary(results)
    
    all_ok = all(r.clone_success and r.generation_success for r in results)
    if all_ok:
        print("\n🎉 All models cloned and generated successfully!")
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
