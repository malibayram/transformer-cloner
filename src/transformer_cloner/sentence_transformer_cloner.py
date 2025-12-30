"""SentenceTransformer cloner for cloning SentenceTransformer models with pruning support."""

import json
import os
import shutil
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors


def _resolve_model_path(model_id: str, token: Optional[str] = None) -> str:
    """
    Resolve a model path - download from HuggingFace Hub if not a local path.
    
    Args:
        model_id: Local path or HuggingFace model ID (e.g., "google/embeddinggemma-300m")
        token: Optional HuggingFace API token for gated models
    
    Returns:
        Local path to the model directory
    """
    # Check if it's already a local path
    if os.path.isdir(model_id):
        return model_id
    
    # Download from HuggingFace Hub
    local_path = snapshot_download(
        repo_id=model_id,
        token=token,
    )
    return local_path


class SentenceTransformerCloner:
    """
    Clone a SentenceTransformer model with a new tokenizer and/or architecture pruning.

    Uses TransformerCloner for the core transformer model and handles
    Dense layers, Pooling configuration, and other modules.

    Example usage:
        >>> from transformer_cloner import SentenceTransformerCloner, PruningConfig
        >>> 
        >>> # Clone with a new tokenizer (maps embeddings to new vocab)
        >>> cloner = SentenceTransformerCloner(
        ...     "path/to/model",
        ...     target_tokenizer_id="path/to/new/tokenizer"
        ... )
        >>> cloner.clone()
        >>> cloner.save("./cloned_model")
        >>>
        >>> # Clone with new tokenizer AND architecture pruning
        >>> config = PruningConfig(hidden_size=512)
        >>> cloner = SentenceTransformerCloner(
        ...     "path/to/model",
        ...     target_tokenizer_id="path/to/new/tokenizer",
        ...     pruning_config=config
        ... )
        >>> cloner.clone()
        >>> cloner.save("./cloned_pruned_model")
    """

    def __init__(
        self,
        model_path: str,
        target_tokenizer_id: Optional[str] = None,
        pruning_config: Optional["PruningConfig"] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize the SentenceTransformerCloner.

        Args:
            model_path: Path to the SentenceTransformer model (local or HuggingFace ID)
            target_tokenizer_id: Path or HuggingFace ID for the target tokenizer.
                                 If None, uses the original model's tokenizer.
            pruning_config: Optional pruning configuration for reducing model dimensions
            token: Optional HuggingFace API token for accessing gated models
        """
        from transformer_cloner.pruning_config import PruningConfig
        
        self.model_id = model_path  # Keep original for reference
        self.model_path = _resolve_model_path(model_path, token)  # Resolved local path
        self.target_tokenizer_id = target_tokenizer_id
        self.pruning_config = pruning_config
        self.token = token
        
        # Will be populated after cloning
        self.cloned_transformer = None
        self.cloned_tokenizer = None
        self.cloned_modules: dict[str, dict] = {}  # module_name -> {"config": ..., "weights": ...}
        self.modules_info: list[dict] = []  # From modules.json
        self.original_hidden_size: int = 0
        self.new_hidden_size: int = 0
        
    def _load_modules_json(self) -> list[dict]:
        """Load and parse modules.json from the model directory."""
        modules_path = os.path.join(self.model_path, "modules.json")
        if not os.path.exists(modules_path):
            raise FileNotFoundError(f"modules.json not found at {modules_path}")
        
        with open(modules_path, "r") as f:
            return json.load(f)
    
    def _load_dense_module(self, module_path: str) -> tuple[dict, dict]:
        """
        Load a Dense module's config and weights.
        
        Returns:
            Tuple of (config_dict, state_dict)
        """
        config_path = os.path.join(module_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load weights
        safetensors_path = os.path.join(module_path, "model.safetensors")
        pytorch_path = os.path.join(module_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            state_dict = load_safetensors(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        else:
            raise FileNotFoundError(f"No model weights found in {module_path}")
        
        return config, state_dict
    
    def _load_pooling_config(self, module_path: str) -> dict:
        """Load Pooling module config."""
        config_path = os.path.join(module_path, "config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _prune_dense_weights(
        self,
        state_dict: dict,
        config: dict,
        new_in_features: Optional[int] = None,
        new_out_features: Optional[int] = None,
    ) -> tuple[dict, dict]:
        """
        Prune Dense layer weights to new dimensions.
        
        Args:
            state_dict: Original state dict with linear.weight and optionally linear.bias
            config: Original config dict
            new_in_features: New input dimension (None to keep original)
            new_out_features: New output dimension (None to keep original)
        
        Returns:
            Tuple of (new_state_dict, new_config)
        """
        orig_in = config["in_features"]
        orig_out = config["out_features"]
        
        target_in = new_in_features if new_in_features else orig_in
        target_out = new_out_features if new_out_features else orig_out
        
        # Prune weight matrix [out_features, in_features]
        weight = state_dict["linear.weight"]
        new_weight = weight[:target_out, :target_in].clone()
        
        new_state_dict = {"linear.weight": new_weight}
        
        # Prune bias if present
        if "linear.bias" in state_dict and config.get("bias", True):
            bias = state_dict["linear.bias"]
            new_state_dict["linear.bias"] = bias[:target_out].clone()
        
        # Update config
        new_config = config.copy()
        new_config["in_features"] = target_in
        new_config["out_features"] = target_out
        
        return new_state_dict, new_config
    
    def clone(self, verbose: bool = True) -> "SentenceTransformerCloner":
        """
        Clone the SentenceTransformer model.
        
        If pruning_config is set, prunes the transformer and Dense layers.
        Otherwise, uses original model weights.
        
        Args:
            verbose: Whether to print progress
        
        Returns:
            Self for method chaining
        """
        from transformer_cloner.cloner import TransformerCloner
        from transformer_cloner.pruning_config import PruningConfig
        
        if verbose:
            print(f"Cloning SentenceTransformer from: {self.model_path}")
        
        # Load modules.json
        self.modules_info = self._load_modules_json()
        
        if verbose:
            print(f"Found {len(self.modules_info)} modules:")
            for m in self.modules_info:
                print(f"  - {m['name']}: {m['type']}")
        
        # Process each module
        for module_info in self.modules_info:
            module_type = module_info["type"]
            module_name = module_info["name"]
            module_path = os.path.join(self.model_path, module_info["path"])
            
            # Extract the class name from the module type (e.g., "sentence_transformers.models.Dense" -> "Dense")
            type_class = module_type.split(".")[-1]
            
            if type_class == "Transformer":
                self._clone_transformer(module_path, verbose)
            elif type_class == "Pooling":
                self._clone_pooling(module_path, module_name, verbose)
            elif type_class == "Dense":
                self._clone_dense(module_path, module_name, verbose)
            elif type_class == "Normalize":
                self._clone_normalize(module_name, verbose)
            elif type_class == "LayerNorm":
                self._clone_layernorm(module_path, module_name, verbose)
            elif type_class == "Dropout":
                self._clone_dropout(module_path, module_name, verbose)
            else:
                # For any other module type, copy as-is
                if verbose:
                    print(f"Copying module: {module_name} ({type_class})")
                self._copy_module(module_path, module_name)
        
        if verbose:
            print("Cloning complete!")
        
        return self
    
    def _clone_transformer(self, module_path: str, verbose: bool) -> None:
        """Clone the transformer module using TransformerCloner."""
        from transformers import (AutoConfig, AutoModelForCausalLM,
                                  AutoTokenizer)

        from transformer_cloner.cloner import TransformerCloner
        
        if verbose:
            print("Cloning Transformer module...")
        
        # Load the transformer model's config to get hidden size
        config = AutoConfig.from_pretrained(module_path, token=self.token)
        self.original_hidden_size = config.hidden_size
        
        if self.pruning_config and self.pruning_config.hidden_size:
            self.new_hidden_size = self.pruning_config.hidden_size
        else:
            self.new_hidden_size = self.original_hidden_size
        
        # Determine which tokenizer to use
        target_tokenizer = self.target_tokenizer_id if self.target_tokenizer_id else module_path
        has_new_tokenizer = self.target_tokenizer_id is not None
        
        if has_new_tokenizer or self.pruning_config:
            # Use TransformerCloner when we have a new tokenizer or pruning config
            cloner = TransformerCloner(
                org_model_id=module_path,
                target_tokenizer_id=target_tokenizer,
                token=self.token,
            )
            
            if verbose:
                print(f"  Original vocab size: {len(cloner.org_tokenizer)}")
                print(f"  Target vocab size: {len(cloner.target_tokenizer)}")
            
            if self.pruning_config:
                # Clone with pruning and optionally new tokenizer
                self.cloned_transformer, self.cloned_tokenizer, _ = cloner.clone_with_vocab_pruning(
                    vocab_size=len(cloner.target_tokenizer),
                    pruning_config=self.pruning_config,
                    verbose=verbose,
                )
            else:
                # Clone with new tokenizer only (no architecture pruning)
                self.cloned_transformer = cloner.clone(verbose=verbose)
                self.cloned_tokenizer = cloner.target_tokenizer
        else:
            # No new tokenizer and no pruning - load original model and tokenizer
            self.cloned_transformer = AutoModelForCausalLM.from_pretrained(
                module_path, token=self.token
            )
            self.cloned_tokenizer = AutoTokenizer.from_pretrained(
                module_path, token=self.token
            )
        
        if verbose:
            print(f"  Original hidden_size: {self.original_hidden_size}")
            print(f"  New hidden_size: {self.new_hidden_size}")
    
    def _clone_pooling(self, module_path: str, module_name: str, verbose: bool) -> None:
        """Clone the Pooling module, updating word_embedding_dimension if pruning."""
        if verbose:
            print(f"Processing Pooling module: {module_name}")
        
        config = self._load_pooling_config(module_path)
        
        if self.pruning_config and self.pruning_config.hidden_size:
            old_dim = config["word_embedding_dimension"]
            config["word_embedding_dimension"] = self.new_hidden_size
            if verbose:
                print(f"  Updated word_embedding_dimension: {old_dim} -> {self.new_hidden_size}")
        
        self.cloned_modules[module_name] = {
            "config": config,
            "weights": None,  # Pooling has no weights
            "type": "Pooling",
        }
    
    def _clone_dense(self, module_path: str, module_name: str, verbose: bool) -> None:
        """Clone a Dense module, pruning weights if needed."""
        if verbose:
            print(f"Processing Dense module: {module_name}")
        
        config, state_dict = self._load_dense_module(module_path)
        
        if self.pruning_config and self.pruning_config.hidden_size:
            orig_in = config["in_features"]
            orig_out = config["out_features"]
            
            # Determine which dimension to prune based on the Dense layer's role
            # First Dense in the chain: in_features should match transformer hidden_size
            # Last Dense in the chain: out_features should match transformer hidden_size
            
            new_in = None
            new_out = None
            
            # If in_features matches original hidden_size, prune it
            if orig_in == self.original_hidden_size:
                new_in = self.new_hidden_size
            
            # If out_features matches original hidden_size, prune it
            if orig_out == self.original_hidden_size:
                new_out = self.new_hidden_size
            
            if new_in or new_out:
                state_dict, config = self._prune_dense_weights(
                    state_dict, config, new_in, new_out
                )
                if verbose:
                    print(f"  Pruned: ({orig_in}, {orig_out}) -> ({config['in_features']}, {config['out_features']})")
            else:
                if verbose:
                    print(f"  Keeping original dimensions: ({orig_in}, {orig_out})")
        else:
            if verbose:
                print(f"  Keeping original: in={config['in_features']}, out={config['out_features']}")
        
        self.cloned_modules[module_name] = {
            "config": config,
            "weights": state_dict,
            "type": "Dense",
        }
    
    def _clone_normalize(self, module_name: str, verbose: bool) -> None:
        """Clone the Normalize module (no weights, just marker)."""
        if verbose:
            print(f"Processing Normalize module: {module_name}")
        
        self.cloned_modules[module_name] = {
            "config": None,
            "weights": None,
            "type": "Normalize",
        }
    
    def _clone_layernorm(self, module_path: str, module_name: str, verbose: bool) -> None:
        """Clone the LayerNorm module, updating dimension if pruning."""
        if verbose:
            print(f"Processing LayerNorm module: {module_name}")
        
        config_path = os.path.join(module_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Load weights
        state_dict = self._load_module_weights(module_path)
        
        # Update dimension if pruning
        if self.pruning_config and self.pruning_config.hidden_size:
            orig_dim = config["dimension"]
            if orig_dim == self.original_hidden_size:
                config["dimension"] = self.new_hidden_size
                # Prune the LayerNorm weights (weight and bias)
                if state_dict:
                    if "norm.weight" in state_dict:
                        state_dict["norm.weight"] = state_dict["norm.weight"][:self.new_hidden_size].clone()
                    if "norm.bias" in state_dict:
                        state_dict["norm.bias"] = state_dict["norm.bias"][:self.new_hidden_size].clone()
                if verbose:
                    print(f"  Updated dimension: {orig_dim} -> {self.new_hidden_size}")
            else:
                if verbose:
                    print(f"  Keeping dimension: {orig_dim}")
        else:
            if verbose:
                print(f"  Keeping dimension: {config['dimension']}")
        
        self.cloned_modules[module_name] = {
            "config": config,
            "weights": state_dict,
            "type": "LayerNorm",
        }
    
    def _clone_dropout(self, module_path: str, module_name: str, verbose: bool) -> None:
        """Clone the Dropout module (config only, no weights)."""
        if verbose:
            print(f"Processing Dropout module: {module_name}")
        
        config_path = os.path.join(module_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.cloned_modules[module_name] = {
            "config": config,
            "weights": None,
            "type": "Dropout",
        }
    
    def _load_module_weights(self, module_path: str) -> dict:
        """Load module weights from safetensors or pytorch format."""
        safetensors_path = os.path.join(module_path, "model.safetensors")
        pytorch_path = os.path.join(module_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            return load_safetensors(safetensors_path)
        elif os.path.exists(pytorch_path):
            return torch.load(pytorch_path, map_location="cpu", weights_only=True)
        return {}
    
    def _copy_module(self, module_path: str, module_name: str) -> None:
        """Copy an unknown module type as-is."""
        self.cloned_modules[module_name] = {
            "source_path": module_path,
            "type": "copy",
        }
    
    def save(self, output_path: str, safe_serialization: bool = True, verbose: bool = True) -> None:
        """
        Save the cloned SentenceTransformer model.
        
        Args:
            output_path: Directory to save the model
            safe_serialization: Use safetensors format (default True)
            verbose: Whether to print progress
        """
        if self.cloned_transformer is None:
            raise ValueError("Must call clone() before save()")
        
        os.makedirs(output_path, exist_ok=True)
        
        if verbose:
            print(f"Saving cloned model to: {output_path}")
        
        # Save transformer model and tokenizer
        self.cloned_transformer.save_pretrained(
            output_path, safe_serialization=safe_serialization
        )
        
        # Save tokenizer - handle VocabPrunedTokenizer if present
        if hasattr(self.cloned_tokenizer, 'save_pretrained'):
            self.cloned_tokenizer.save_pretrained(output_path)
        
        if verbose:
            print("  Saved transformer model and tokenizer")
        
        # Save modules.json
        modules_json = []
        for module_info in self.modules_info:
            modules_json.append(module_info)
        
        with open(os.path.join(output_path, "modules.json"), "w") as f:
            json.dump(modules_json, f, indent=2)
        
        # Save each cloned module
        for module_name, module_data in self.cloned_modules.items():
            module_type = module_data["type"]
            
            # Find the module info to get the path
            module_path = None
            for m in self.modules_info:
                if m["name"] == module_name:
                    module_path = m["path"]
                    break
            
            if module_path is None:
                continue
            
            full_module_path = os.path.join(output_path, module_path)
            os.makedirs(full_module_path, exist_ok=True)
            
            if module_type == "Pooling":
                self._save_pooling_module(full_module_path, module_data, verbose)
            elif module_type == "Dense":
                self._save_dense_module(full_module_path, module_data, safe_serialization, verbose)
            elif module_type == "Normalize":
                self._save_normalize_module(full_module_path, verbose)
            elif module_type == "LayerNorm":
                self._save_layernorm_module(full_module_path, module_data, safe_serialization, verbose)
            elif module_type == "Dropout":
                self._save_dropout_module(full_module_path, module_data, verbose)
            elif module_type == "copy":
                self._copy_module_dir(module_data["source_path"], full_module_path, verbose)
        
        # Copy other config files from original
        self._copy_config_files(output_path, verbose)
        
        if verbose:
            print("Save complete!")
    
    def _save_pooling_module(self, output_path: str, module_data: dict, verbose: bool) -> None:
        """Save a Pooling module."""
        config = module_data["config"]
        config_path = os.path.join(output_path, "config.json")
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        if verbose:
            print(f"  Saved Pooling config to {output_path}")
    
    def _save_dense_module(
        self, output_path: str, module_data: dict, safe_serialization: bool, verbose: bool
    ) -> None:
        """Save a Dense module."""
        config = module_data["config"]
        weights = module_data["weights"]
        
        # Save config
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Save weights
        if safe_serialization:
            weights_path = os.path.join(output_path, "model.safetensors")
            save_safetensors(weights, weights_path)
        else:
            weights_path = os.path.join(output_path, "pytorch_model.bin")
            torch.save(weights, weights_path)
        
        if verbose:
            print(f"  Saved Dense module to {output_path}")
    
    def _save_normalize_module(self, output_path: str, verbose: bool) -> None:
        """Save a Normalize module (empty directory marker)."""
        # Normalize module typically has no files, but we create the directory
        if verbose:
            print(f"  Created Normalize module at {output_path}")
    
    def _save_layernorm_module(
        self, output_path: str, module_data: dict, safe_serialization: bool, verbose: bool
    ) -> None:
        """Save a LayerNorm module."""
        config = module_data["config"]
        weights = module_data["weights"]
        
        # Save config
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Save weights if present
        if weights:
            if safe_serialization:
                weights_path = os.path.join(output_path, "model.safetensors")
                save_safetensors(weights, weights_path)
            else:
                weights_path = os.path.join(output_path, "pytorch_model.bin")
                torch.save(weights, weights_path)
        
        if verbose:
            print(f"  Saved LayerNorm module to {output_path}")
    
    def _save_dropout_module(self, output_path: str, module_data: dict, verbose: bool) -> None:
        """Save a Dropout module (config only, no weights)."""
        config = module_data["config"]
        
        # Save config
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        if verbose:
            print(f"  Saved Dropout module to {output_path}")
    
    def _copy_module_dir(self, source_path: str, dest_path: str, verbose: bool) -> None:
        """Copy a module directory as-is."""
        if os.path.exists(source_path):
            shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            if verbose:
                print(f"  Copied module from {source_path}")
    
    def _copy_config_files(self, output_path: str, verbose: bool) -> None:
        """Copy additional config files from the original model."""
        config_files = [
            "config_sentence_transformers.json",
            "sentence_bert_config.json",
        ]
        
        for filename in config_files:
            source = os.path.join(self.model_path, filename)
            if os.path.exists(source):
                dest = os.path.join(output_path, filename)
                shutil.copy2(source, dest)
                if verbose:
                    print(f"  Copied {filename}")
