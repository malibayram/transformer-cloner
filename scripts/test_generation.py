#!/usr/bin/env python3
"""
Text Generation Test Script for Cloned Models

Tests that cloned/vocab-pruned models can properly tokenize various text inputs
and generate coherent outputs. Uses previously cloned models if they exist,
otherwise clones them first.

Usage:
    python scripts/test_generation.py
    python scripts/test_generation.py --models-dir ./cloned_models
"""

import argparse
import os
import sys
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for local development
sys.path.insert(0, "src")


# Test prompts covering various text types
TEST_PROMPTS = [
    # Simple English
    {
        "name": "Simple greeting",
        "text": "Hello, how are you today?",
        "description": "Basic English greeting",
    },
    # Code
    {
        "name": "Python code",
        "text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "description": "Python function completion",
    },
    # Numbers and math
    {
        "name": "Math expression",
        "text": "Calculate: 2 + 2 = 4, 3 + 3 = 6, 4 + 4 =",
        "description": "Simple arithmetic pattern",
    },
    # Mixed language (if model supports)
    {
        "name": "Technical text",
        "text": "The transformer architecture uses self-attention mechanisms to",
        "description": "Technical ML description",
    },
    # Short prompt
    {
        "name": "Single word",
        "text": "The",
        "description": "Minimal single-word prompt",
    },
    # Question
    {
        "name": "Question",
        "text": "What is the capital of France?",
        "description": "Simple factual question",
    },
    # List
    {
        "name": "List completion",
        "text": "Top programming languages:\n1. Python\n2. JavaScript\n3.",
        "description": "List continuation",
    },
    # Special characters
    {
        "name": "Special chars",
        "text": "Email: test@example.com, URL: https://",
        "description": "Text with special characters",
    },
]


def find_cloned_models(models_dir: str) -> dict[str, str]:
    """Find all cloned model directories."""
    models = {}
    if not os.path.exists(models_dir):
        return models
    
    for name in os.listdir(models_dir):
        path = os.path.join(models_dir, name)
        if os.path.isdir(path):
            # Check if it has model files
            if os.path.exists(os.path.join(path, "config.json")):
                models[name] = path
    
    return models


def test_generation(
    model_path: str,
    model_name: str,
    max_new_tokens: int = 20,
) -> dict:
    """Test text generation with a cloned model."""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"Path: {model_path}")
    print(f"{'='*70}")
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "load_success": False,
        "tests": [],
        "error": None,
    }
    
    # Load model and tokenizer
    print("\n📦 Loading model and tokenizer...")
    try:
        # Try to load VocabPrunedTokenizer if vocab_mapping.json exists
        vocab_mapping_path = os.path.join(model_path, "vocab_mapping.json")
        legacy_mapping_path = os.path.join(model_path, "vocab_id_mapping.json")
        
        if os.path.exists(vocab_mapping_path):
            # New format: use VocabPrunedTokenizer
            from transformer_cloner.vocab_pruned_tokenizer import VocabPrunedTokenizer
            tokenizer = VocabPrunedTokenizer.from_pretrained(model_path)
            print(f"   Using VocabPrunedTokenizer (proper token remapping)")
        elif os.path.exists(legacy_mapping_path):
            # Legacy format: load mapping manually and create wrapper
            import json
            from transformer_cloner.vocab_pruned_tokenizer import VocabPrunedTokenizer
            
            original_tokenizer = AutoTokenizer.from_pretrained(model_path)
            with open(legacy_mapping_path) as f:
                id_mapping = {int(k): v for k, v in json.load(f).items()}
            
            tokenizer = VocabPrunedTokenizer(
                original_tokenizer=original_tokenizer,
                id_mapping=id_mapping,
                unk_token_id=0,
            )
            print(f"   Using VocabPrunedTokenizer (legacy format)")
        else:
            # No mapping file - use regular tokenizer (will likely fail for pruned models)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"   ⚠️  No vocab mapping found - using original tokenizer")
        
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.eval()
        
        # Get vocab size from model config (pruned size)
        model_vocab_size = model.config.vocab_size
        tokenizer_vocab_size = len(tokenizer) if hasattr(tokenizer, '__len__') else getattr(tokenizer, 'vocab_size', model_vocab_size)
        
        print(f"   ✅ Loaded successfully")
        print(f"   Model vocab size: {model_vocab_size}")
        print(f"   Tokenizer vocab size: {tokenizer_vocab_size}")
        print(f"   Model params: {model.num_parameters():,}")
        results["load_success"] = True
        results["vocab_size"] = model_vocab_size
    except Exception as e:
        print(f"   ❌ Failed to load: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
        return results
    
    # Test each prompt
    print(f"\n🔤 Testing {len(TEST_PROMPTS)} different prompts...")
    print("-" * 70)
    
    for i, prompt_info in enumerate(TEST_PROMPTS, 1):
        prompt_name = prompt_info["name"]
        prompt_text = prompt_info["text"]
        description = prompt_info["description"]
        
        test_result = {
            "name": prompt_name,
            "prompt": prompt_text,
            "description": description,
            "tokenize_success": False,
            "generate_success": False,
            "output": None,
            "error": None,
        }
        
        print(f"\n[{i}/{len(TEST_PROMPTS)}] {prompt_name}: {description}")
        print(f"    Input: {repr(prompt_text[:50])}{'...' if len(prompt_text) > 50 else ''}")
        
        # Test tokenization
        try:
            inputs = tokenizer(prompt_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            num_tokens = input_ids.shape[1]
            
            # Check for out-of-vocab tokens
            max_token_id = input_ids.max().item()
            if max_token_id >= model_vocab_size:
                print(f"    ⚠️  Token ID {max_token_id} >= model vocab size {model_vocab_size}")
                test_result["error"] = f"OOV token: {max_token_id}"
            else:
                print(f"    ✅ Tokenized: {num_tokens} tokens, max_id={max_token_id}")
                test_result["tokenize_success"] = True
                test_result["num_tokens"] = num_tokens
        except Exception as e:
            print(f"    ❌ Tokenization failed: {e}")
            test_result["error"] = f"Tokenize error: {str(e)}"
            results["tests"].append(test_result)
            continue
        
        # Test generation
        try:
            with torch.no_grad():
                # Use input_ids directly - VocabPrunedTokenizer already remaps to valid range
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic for testing
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id or 0,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode output
                generated_ids = outputs[0][input_ids.shape[1]:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                print(f"    ✅ Generated: {repr(generated_text[:60])}{'...' if len(generated_text) > 60 else ''}")
                test_result["generate_success"] = True
                test_result["output"] = generated_text
                test_result["output_tokens"] = len(generated_ids)
        except Exception as e:
            print(f"    ❌ Generation failed: {e}")
            test_result["error"] = f"Generate error: {str(e)}"
        
        results["tests"].append(test_result)
    
    # Summary for this model
    print("\n" + "-" * 70)
    tokenize_ok = sum(1 for t in results["tests"] if t["tokenize_success"])
    generate_ok = sum(1 for t in results["tests"] if t["generate_success"])
    total = len(results["tests"])
    
    print(f"Summary: {tokenize_ok}/{total} tokenized, {generate_ok}/{total} generated")
    
    if generate_ok == total:
        print("✅ All tests passed!")
    elif generate_ok > 0:
        print("⚠️  Some tests failed")
    else:
        print("❌ All tests failed")
    
    # Clean up
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def print_final_summary(all_results: list[dict]):
    """Print final summary of all model tests."""
    print("\n")
    print("=" * 80)
    print("GENERATION TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<35} {'Load':<6} {'Tokenize':<10} {'Generate':<10} {'Status'}")
    print("-" * 80)
    
    for r in all_results:
        name = r["model_name"][:33]
        load = "✅" if r["load_success"] else "❌"
        
        if r["load_success"]:
            tok_ok = sum(1 for t in r["tests"] if t["tokenize_success"])
            gen_ok = sum(1 for t in r["tests"] if t["generate_success"])
            total = len(r["tests"])
            tok = f"{tok_ok}/{total}"
            gen = f"{gen_ok}/{total}"
            
            if gen_ok == total:
                status = "✅ All passed"
            elif gen_ok > 0:
                status = "⚠️ Partial"
            else:
                status = "❌ Failed"
        else:
            tok = "-"
            gen = "-"
            status = f"❌ {r.get('error', 'Load failed')[:20]}"
        
        print(f"{name:<35} {load:<6} {tok:<10} {gen:<10} {status}")
    
    print("-" * 80)
    
    # Overall stats
    total_models = len(all_results)
    loaded = sum(1 for r in all_results if r["load_success"])
    all_passed = sum(1 for r in all_results 
                     if r["load_success"] and 
                     all(t["generate_success"] for t in r["tests"]))
    
    print(f"\nOverall: {all_passed}/{total_models} models passed all generation tests")


def main():
    parser = argparse.ArgumentParser(
        description="Test text generation with cloned models",
    )
    parser.add_argument(
        "--models-dir", "-d",
        type=str,
        default="./cloned_models",
        help="Directory containing cloned models (default: ./cloned_models)",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=20,
        help="Maximum new tokens to generate (default: 20)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Test only a specific model by name (e.g., 'smollm2-135m-instruct-vocab8000')",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CLONED MODEL GENERATION TEST")
    print("=" * 80)
    print(f"\nModels directory: {os.path.abspath(args.models_dir)}")
    print(f"Max new tokens: {args.max_tokens}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    
    # Find cloned models
    models = find_cloned_models(args.models_dir)
    
    if not models:
        print(f"\n❌ No cloned models found in {args.models_dir}")
        print("Run test_multi_model_cloning.py first to create cloned models.")
        return 1
    
    print(f"\n📁 Found {len(models)} cloned models:")
    for name, path in sorted(models.items()):
        print(f"   - {name}")
    
    # Filter to specific model if requested
    if args.model:
        if args.model in models:
            models = {args.model: models[args.model]}
        else:
            print(f"\n❌ Model '{args.model}' not found")
            print(f"Available: {', '.join(sorted(models.keys()))}")
            return 1
    
    # Test each model
    all_results = []
    
    for model_name, model_path in sorted(models.items()):
        try:
            result = test_generation(
                model_path=model_path,
                model_name=model_name,
                max_new_tokens=args.max_tokens,
            )
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error testing {model_name}: {e}")
            all_results.append({
                "model_name": model_name,
                "model_path": model_path,
                "load_success": False,
                "tests": [],
                "error": str(e),
            })
    
    # Print final summary
    print_final_summary(all_results)
    
    # Return exit code
    all_ok = all(
        r["load_success"] and all(t["generate_success"] for t in r["tests"])
        for r in all_results
    )
    
    if all_ok:
        print("\n🎉 All models passed all generation tests!")
        return 0
    else:
        print("\n⚠️  Some tests failed - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
