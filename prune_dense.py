import torch
import json
import os
from safetensors.torch import load_file, save_file

def prune_dense_layer(dense_path, input_pruning=False, output_pruning=False):
    print(f"Processing {dense_path}...")
    model_path = os.path.join(dense_path, "model.safetensors")
    config_path = os.path.join(dense_path, "config.json")

    # Load Config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load Weights
    tensors = load_file(model_path)
    weight = tensors["linear.weight"]
    # Usually Dense layers in SentenceTransformer don't have bias if bias=False, but let's check
    bias = tensors.get("linear.bias", None)

    print(f"Original weight shape: {weight.shape}")
    
    new_weight = weight
    
    # Prune Input (Columns corresponding to input features)
    # Weight shape for Linear is [out_features, in_features]
    if input_pruning:
        current_in = weight.shape[1]
        target_in = 640
        if current_in > target_in:
            print(f"Pruning input from {current_in} to {target_in}")
            new_weight = new_weight[:, :target_in]
            config["in_features"] = target_in
        else:
            print(f"Input already {current_in}, skipping prune.")

    # Prune Output (Rows corresponding to output features)
    if output_pruning:
        current_out = weight.shape[0]
        target_out = 640
        if current_out > target_out:
            print(f"Pruning output from {current_out} to {target_out}")
            new_weight = new_weight[:target_out, :]
            config["out_features"] = target_out
            
            if bias is not None:
                print(f"Pruning bias from {bias.shape[0]} to {target_out}")
                bias = bias[:target_out]
        else:
            print(f"Output already {current_out}, skipping prune.")

    print(f"New weight shape: {new_weight.shape}")
    
    # Ensure contiguous for safetensors
    new_weight = new_weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    
    # Save Weights
    new_tensors = {"linear.weight": new_weight}
    if bias is not None:
        new_tensors["linear.bias"] = bias
        
    save_file(new_tensors, model_path)
    print(f"Saved pruned model to {model_path}")

    # Save Config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved updated config to {config_path}")
    print("-" * 20)

def main():
    base_dir = "embeddinggemm"
    
    # 2_Dense: Input comes from Gemma (640), so prune input from 768 -> 640
    # Current config: in=768, out=3072
    dense_2_path = os.path.join(base_dir, "2_Dense")
    if os.path.exists(dense_2_path):
        prune_dense_layer(dense_2_path, input_pruning=True, output_pruning=False)
    else:
        print(f"Warning: {dense_2_path} not found.")

    # 3_Dense: This layer takes 3072 and outputs back to embedding dimension.
    # The original was 768. We want final output to be 640 to match new embedding dim?
    # Wait, 3_Dense is usually the projection back.
    # Let's check the structure.
    # Gemma (640) -> 2_Dense (768->3072) -> Activation -> 3_Dense (3072->768) -> Normalize
    # If we change Gemma to output 640...
    # 2_Dense needs input 640. So [3072, 640] weight. (Input pruning)
    # 3_Dense output needs to be 640 if we want the final embedding to be 640.
    # The user asked "prune Dence models from 768 to 640".
    # So 3_Dense should have output 640. So [640, 3072] weight. (Output pruning)
    
    dense_3_path = os.path.join(base_dir, "3_Dense")
    if os.path.exists(dense_3_path):
        prune_dense_layer(dense_3_path, input_pruning=False, output_pruning=True)
    else:
        print(f"Warning: {dense_3_path} not found.")

if __name__ == "__main__":
    main()
