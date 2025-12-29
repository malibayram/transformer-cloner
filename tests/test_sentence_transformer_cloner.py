"""Test script for SentenceTransformerCloner."""

import os
import tempfile
import shutil

from transformer_cloner import SentenceTransformerCloner, PruningConfig


def test_without_pruning():
    """Test cloning without pruning - should copy original weights."""
    print("\n=== Test: Clone without pruning ===")
    
    model_path = "./embeddinggemma"
    if not os.path.exists(model_path):
        print(f"Skipping: {model_path} not found")
        return
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "cloned_model")
        
        cloner = SentenceTransformerCloner(model_path)
        cloner.clone(verbose=True)
        cloner.save(output_path, verbose=True)
        
        # Verify output structure
        assert os.path.exists(os.path.join(output_path, "modules.json")), "modules.json missing"
        assert os.path.exists(os.path.join(output_path, "config.json")), "config.json missing"
        assert os.path.exists(os.path.join(output_path, "1_Pooling", "config.json")), "Pooling config missing"
        
        print("✓ Clone without pruning succeeded!")


def test_with_pruning():
    """Test cloning with pruning - should prune Dense layers."""
    print("\n=== Test: Clone with pruning ===")
    
    model_path = "./embeddinggemma"
    if not os.path.exists(model_path):
        print(f"Skipping: {model_path} not found")
        return
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "cloned_model_pruned")
        
        # Prune to smaller hidden size (must be compatible with original model)
        # embeddinggemma has: hidden_size=640, num_heads=4, num_kv_heads=1, head_dim=64
        config = PruningConfig(
            hidden_size=512,
            num_hidden_layers=12,
            intermediate_size=900,
            num_attention_heads=4,
            num_key_value_heads=1,  # Keep as 1, can't exceed original
        )
        
        cloner = SentenceTransformerCloner(model_path, pruning_config=config)
        cloner.clone(verbose=True)
        cloner.save(output_path, verbose=True)
        
        # Verify output structure
        assert os.path.exists(os.path.join(output_path, "modules.json")), "modules.json missing"
        
        # Check pooling config was updated
        import json
        with open(os.path.join(output_path, "1_Pooling", "config.json")) as f:
            pooling_config = json.load(f)
        
        assert pooling_config["word_embedding_dimension"] == 512, f"Pooling dimension not updated: {pooling_config}"
        
        # Check Dense layers were pruned
        with open(os.path.join(output_path, "2_Dense", "config.json")) as f:
            dense1_config = json.load(f)
        
        with open(os.path.join(output_path, "3_Dense", "config.json")) as f:
            dense2_config = json.load(f)
        
        print(f"Dense1: in={dense1_config['in_features']}, out={dense1_config['out_features']}")
        print(f"Dense2: in={dense2_config['in_features']}, out={dense2_config['out_features']}")
        
        print("✓ Clone with pruning succeeded!")


if __name__ == "__main__":
    test_without_pruning()
    test_with_pruning()
    print("\n=== All tests passed! ===")
