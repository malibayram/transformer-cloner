#!/usr/bin/env python3
"""
Multi-Model Cloning Compatibility Test Script

Tests the transformer-cloner package against various model architectures
to verify compatibility. Supports HuggingFace authentication for gated models
and can optionally push cloned models to HuggingFace Hub.

Usage:
    python scripts/test_multi_model_cloning.py

Requirements:
    - transformer-cloner package installed
    - HuggingFace Hub authentication for gated models
"""

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
from huggingface_hub import HfApi, login

# Add src to path for local development
sys.path.insert(0, "src")

from transformer_cloner import TransformerCloner


@dataclass
class TestResult:
    """Result of testing a single model."""
    model_name: str
    model_id: str
    load_success: bool
    token_map_success: bool
    clone_success: bool
    save_success: bool = False
    push_success: bool = False
    error_message: Optional[str] = None
    model_config: Optional[dict] = None
    load_time: float = 0.0
    clone_time: float = 0.0
    output_path: Optional[str] = None
    param_count: int = 0


# Models to test - user-specified list
MODELS_TO_TEST = [
    {
        "name": "SmolLM2-135M-Instruct",
        "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "description": "SmolLM2 Instruct model (135M params)",
    },
    {
        "name": "Qwen3-0.6B",
        "model_id": "Qwen/Qwen3-0.6B",
        "description": "Qwen3 small model (0.6B params)",
    },
    {
        "name": "Phi-1",
        "model_id": "microsoft/phi-1",
        "description": "Microsoft Phi-1 (1.3B params)",
    },
    {
        "name": "Llama-3.2-1B",
        "model_id": "meta-llama/Llama-3.2-1B",
        "description": "Meta Llama 3.2 (1B params, gated)",
    },
]


def get_model_config_summary(model) -> dict:
    """Extract key config parameters from a model."""
    config = model.config
    return {
        "hidden_size": getattr(config, "hidden_size", "N/A"),
        "num_layers": getattr(config, "num_hidden_layers", "N/A"),
        "num_heads": getattr(config, "num_attention_heads", "N/A"),
        "vocab_size": getattr(config, "vocab_size", "N/A"),
        "intermediate_size": getattr(config, "intermediate_size", "N/A"),
        "model_type": getattr(config, "model_type", "N/A"),
    }


def setup_hf_auth(token: Optional[str] = None) -> Optional[str]:
    """Setup HuggingFace authentication."""
    if token:
        print("🔑 Using provided HuggingFace token...")
        login(token=token)
        return token
    
    # Check environment variable
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env_token:
        print("🔑 Using HuggingFace token from environment variable...")
        login(token=env_token)
        return env_token
    
    # Interactive login
    print("\n🔐 HuggingFace Authentication")
    print("Some models (like Llama) require authentication.")
    print("You can get your token from: https://huggingface.co/settings/tokens")
    print()
    
    token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    if token:
        login(token=token)
        return token
    
    print("⚠️  No token provided. Gated models may fail to load.")
    return None


def test_model(
    model_info: dict,
    vocab_size: int,
    output_dir: Optional[str],
    push_to_hub: bool,
    hub_org: Optional[str],
    hf_token: Optional[str],
) -> TestResult:
    """Test a single model for cloning compatibility."""
    name = model_info["name"]
    model_id = model_info["model_id"]
    
    result = TestResult(
        model_name=name,
        model_id=model_id,
        load_success=False,
        token_map_success=False,
        clone_success=False,
    )
    
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Model ID: {model_id}")
    print(f"Target vocab size: {vocab_size}")
    print(f"{'='*60}")
    
    # Step 1: Try to load the model
    print("\n[1/3] Loading model and tokenizer...")
    try:
        start_time = time.time()
        cloner = TransformerCloner(
            org_model_id=model_id,
            target_tokenizer_id=model_id,  # Use same tokenizer for testing
            token=hf_token,  # Pass HF token for gated models
        )
        result.load_time = time.time() - start_time
        result.load_success = True
        result.model_config = get_model_config_summary(cloner.org_model)
        print(f"   ✅ Loaded successfully in {result.load_time:.1f}s")
        print(f"   Config: {result.model_config}")
    except Exception as e:
        result.error_message = f"Load failed: {str(e)}"
        print(f"   ❌ Failed to load: {e}")
        if "gated" in str(e).lower() or "auth" in str(e).lower() or "access" in str(e).lower():
            print("   ℹ️  This model requires HuggingFace authentication")
        return result
    
    # Note: Skipping build_token_id_map() - not needed for clone_with_vocab_pruning
    # which uses direct 1:1 embedding mapping for the kept tokens
    result.token_map_success = True  # Mark as success (not applicable)
    
    # Step 2: Test vocab-pruned cloning
    print(f"\n[2/3] Cloning with vocab pruning (vocab_size={vocab_size})...")
    try:
        start_time = time.time()
        cloned_model, pruned_tokenizer, id_mapping = cloner.clone_with_vocab_pruning(
            vocab_size=vocab_size,
            verbose=True,
        )
        result.clone_time = time.time() - start_time
        result.clone_success = True
        
        # Calculate size reduction
        orig_vocab = result.model_config["vocab_size"]
        new_vocab = cloned_model.config.vocab_size
        reduction = (1 - new_vocab / orig_vocab) * 100 if orig_vocab != "N/A" else 0
        
        print(f"   ✅ Clone successful in {result.clone_time:.1f}s")
        print(f"   Original vocab: {orig_vocab} → Pruned vocab: {new_vocab} ({reduction:.1f}% reduction)")
        
        # Calculate model size
        param_count = cloned_model.num_parameters()
        result.param_count = param_count
        print(f"   Cloned model parameters: {param_count:,} ({param_count/1e6:.1f}M)")
        
        # Verify the cloned model can do a forward pass
        print("   Testing forward pass...")
        with torch.no_grad():
            input_ids = torch.tensor([[0, 1, 2, 3, 4]])
            try:
                outputs = cloned_model(input_ids)
                print(f"   ✅ Forward pass successful, output shape: {outputs.logits.shape}")
            except Exception as e:
                print(f"   ⚠️  Forward pass failed: {e}")
        
    except Exception as e:
        result.error_message = f"Clone failed: {str(e)}"
        print(f"   ❌ Failed to clone: {e}")
        traceback.print_exc()
        return result
    
    # Step 4: Save or push to hub
    print("\n[3/3] Saving model...")
    
    # Generate output name
    safe_name = name.replace("/", "-").replace(" ", "-").lower()
    pruned_name = f"{safe_name}-vocab{vocab_size}"
    
    if push_to_hub and hf_token:
        # Push to HuggingFace Hub
        hub_repo = f"{hub_org}/{pruned_name}" if hub_org else pruned_name
        print(f"   📤 Pushing to HuggingFace Hub: {hub_repo}")
        try:
            cloned_model.push_to_hub(hub_repo, token=hf_token)
            pruned_tokenizer.original_tokenizer.push_to_hub(hub_repo, token=hf_token)
            result.push_success = True
            result.output_path = f"https://huggingface.co/{hub_repo}"
            print(f"   ✅ Pushed to Hub: {result.output_path}")
        except Exception as e:
            print(f"   ❌ Failed to push to Hub: {e}")
            result.error_message = f"Push failed: {str(e)}"
    
    if output_dir:
        # Save locally
        local_path = os.path.join(output_dir, pruned_name)
        print(f"   💾 Saving locally to: {local_path}")
        try:
            os.makedirs(local_path, exist_ok=True)
            cloned_model.save_pretrained(local_path)
            # Save the VocabPrunedTokenizer with proper mapping format
            pruned_tokenizer.save_pretrained(local_path)
            
            result.save_success = True
            result.output_path = local_path
            print(f"   ✅ Saved to: {local_path}")
        except Exception as e:
            print(f"   ❌ Failed to save locally: {e}")
            if not result.error_message:
                result.error_message = f"Save failed: {str(e)}"
    
    # Clean up to free memory
    del cloned_model
    del cloner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def print_summary(results: list[TestResult], push_to_hub: bool):
    """Print a summary table of all test results."""
    print("\n")
    print("=" * 90)
    print("MULTI-MODEL CLONING TEST RESULTS")
    print("=" * 90)
    
    # Header
    if push_to_hub:
        print(f"\n{'Model':<25} {'Params':<12} {'Load':<6} {'Clone':<7} {'Push':<6} {'Notes'}")
    else:
        print(f"\n{'Model':<25} {'Params':<12} {'Load':<6} {'Clone':<7} {'Save':<6} {'Notes'}")
    print("-" * 100)
    
    for r in results:
        load = "✅" if r.load_success else "❌"
        clone = "✅" if r.clone_success else ("❌" if r.load_success else "-")
        params = f"{r.param_count/1e6:.1f}M" if r.param_count > 0 else "-"
        
        if push_to_hub:
            save_push = "✅" if r.push_success else ("❌" if r.clone_success else "-")
        else:
            save_push = "✅" if r.save_success else ("❌" if r.clone_success else "-")
        
        notes = ""
        if r.clone_success and (r.save_success or r.push_success):
            notes = f"OK ({r.load_time:.1f}s load, {r.clone_time:.1f}s clone)"
        elif r.output_path:
            notes = r.output_path[:40]
        elif r.error_message:
            notes = r.error_message[:40] + "..." if len(r.error_message) > 40 else r.error_message
        
        print(f"{r.model_name:<25} {params:<12} {load:<6} {clone:<7} {save_push:<6} {notes}")
    
    print("-" * 100)
    
    # Summary stats
    total = len(results)
    loaded = sum(1 for r in results if r.load_success)
    cloned = sum(1 for r in results if r.clone_success)
    saved = sum(1 for r in results if r.save_success or r.push_success)
    
    print(f"\nSummary: {cloned}/{total} cloned, {saved}/{total} saved/pushed, {loaded}/{total} loaded")
    
    # List successful outputs
    successful = [r for r in results if r.output_path]
    if successful:
        print("\n✅ Saved/Pushed models:")
        for r in successful:
            print(f"   - {r.model_name}: {r.output_path}")
    
    # List any failures with details
    failures = [r for r in results if not r.clone_success]
    if failures:
        print("\n⚠️  Failed models:")
        for r in failures:
            print(f"   - {r.model_name}: {r.error_message}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test transformer-cloner with multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--vocab-size", "-v",
        type=int,
        default=8000,
        help="Target vocabulary size for pruned models (default: 8000)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./cloned_models",
        help="Directory to save cloned models (default: ./cloned_models)",
    )
    parser.add_argument(
        "--push-to-hub", "-p",
        action="store_true",
        help="Push cloned models to HuggingFace Hub instead of saving locally",
    )
    parser.add_argument(
        "--hub-org",
        type=str,
        default=None,
        help="HuggingFace organization/username for pushing (default: your username)",
    )
    parser.add_argument(
        "--hf-token", "-t",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--skip-auth",
        action="store_true",
        help="Skip interactive authentication prompt",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TRANSFORMER-CLONER MULTI-MODEL COMPATIBILITY TEST")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  - Target vocab size: {args.vocab_size}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"  - Push to Hub: {args.push_to_hub}")
    if args.hub_org:
        print(f"  - Hub organization: {args.hub_org}")
    
    print(f"\nModels to test ({len(MODELS_TO_TEST)}):")
    for m in MODELS_TO_TEST:
        print(f"  - {m['name']}: {m['description']}")
    
    # Setup HuggingFace authentication
    hf_token = None
    if not args.skip_auth:
        hf_token = setup_hf_auth(args.hf_token)
    
    # Create output directory if saving locally
    if not args.push_to_hub:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n📁 Output directory: {os.path.abspath(args.output_dir)}")
    
    results = []
    
    for model_info in MODELS_TO_TEST:
        try:
            result = test_model(
                model_info=model_info,
                vocab_size=args.vocab_size,
                output_dir=args.output_dir if not args.push_to_hub else None,
                push_to_hub=args.push_to_hub,
                hub_org=args.hub_org,
                hf_token=hf_token,
            )
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error testing {model_info['name']}: {e}")
            results.append(TestResult(
                model_name=model_info["name"],
                model_id=model_info["model_id"],
                load_success=False,
                token_map_success=False,
                clone_success=False,
                error_message=str(e),
            ))
    
    print_summary(results, args.push_to_hub)
    
    # Return exit code based on results
    if all(r.clone_success for r in results):
        print("\n🎉 All models passed!")
        return 0
    else:
        print("\n⚠️  Some models failed - see above for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
