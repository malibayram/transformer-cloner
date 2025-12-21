"""Main TransformerCloner class for model cloning and pruning."""

import json
import os
import re
import tempfile
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_cloner.embedding_strategy import EmbeddingStrategy
from transformer_cloner.pruning_config import PruningConfig


class TransformerCloner:
    """
    Clone a transformer model by mapping token IDs from a new tokenizer
    to the original tokenizer's token IDs and creating new embeddings.

    Example usage:
        >>> cloner = TransformerCloner(
        ...     org_model_id="google/gemma-3-270m-it",
        ...     target_tokenizer_id="your-username/custom-tokenizer",
        ... )
        >>> cloned_model = cloner.clone(strategy=EmbeddingStrategy.MEAN)
        >>> cloned_model.save_pretrained("path/to/save")
    """

    def __init__(
        self,
        org_model_id: str,
        target_tokenizer_id: str,
        token: Optional[str] = None,
    ):
        """
        Initialize the TransformerCloner.

        Args:
            org_model_id: HuggingFace model ID or path for the original model
            target_tokenizer_id: HuggingFace tokenizer ID or path for the target tokenizer
            token: Optional HuggingFace API token for accessing gated models
        """
        self.org_model_id = org_model_id
        self.target_tokenizer_id = target_tokenizer_id
        self.token = token

        # Load tokenizers
        self.org_tokenizer = AutoTokenizer.from_pretrained(org_model_id, token=token)
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_id, token=token)

        # Load original model
        self.org_model = AutoModelForCausalLM.from_pretrained(org_model_id, token=token)

        # Token ID mapping (target_id -> list of source_ids)
        self.token_id_map: dict[int, list[int]] = {}

    def build_token_id_map(
        self, batch_size: int = 5000, verbose: bool = True
    ) -> dict[int, list[int]]:
        """
        Build a mapping from target tokenizer IDs to original tokenizer IDs.
        Uses batch encoding for much faster processing.

        Args:
            batch_size: Number of tokens to process per batch
            verbose: Whether to print progress

        Returns:
            Dictionary mapping target token IDs to lists of source token IDs
        """
        target_vocab_keys = list(self.target_tokenizer.vocab.keys())
        total_tokens = len(target_vocab_keys)

        if verbose:
            print(f"Building token ID map for {total_tokens} tokens...")

        self.token_id_map = {}

        # Process in batches for much faster encoding
        for batch_start in range(0, total_tokens, batch_size):
            batch_end = min(batch_start + batch_size, total_tokens)
            batch_tokens = target_vocab_keys[batch_start:batch_end]

            # Batch encode all tokens at once
            batch_encoded = self.org_tokenizer(
                batch_tokens,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )

            # Map each token
            for i, vocab in enumerate(batch_tokens):
                target_id = self.target_tokenizer.vocab[vocab]
                # Skip the first token (usually BOS token)
                source_ids = batch_encoded["input_ids"][i][1:]
                self.token_id_map[target_id] = source_ids

            if verbose:
                print(f"Processed {batch_end}/{total_tokens} tokens")

        if verbose:
            print(f"Token ID map built with {len(self.token_id_map)} entries")

        return self.token_id_map

    def _combine_embeddings(
        self,
        source_embeddings: torch.Tensor,
        strategy: EmbeddingStrategy,
    ) -> torch.Tensor:
        """
        Combine multiple source embeddings into a single embedding.

        Args:
            source_embeddings: Tensor of shape (num_sources, embedding_dim)
            strategy: Strategy for combining embeddings

        Returns:
            Combined embedding tensor of shape (embedding_dim,)
        """
        if source_embeddings.shape[0] == 1:
            return source_embeddings[0].clone()

        if strategy == EmbeddingStrategy.MEAN:
            return source_embeddings.mean(dim=0)

        elif strategy == EmbeddingStrategy.SUM:
            return source_embeddings.sum(dim=0)

        elif strategy == EmbeddingStrategy.FIRST:
            return source_embeddings[0].clone()

        elif strategy == EmbeddingStrategy.LAST:
            return source_embeddings[-1].clone()

        elif strategy == EmbeddingStrategy.WEIGHTED:
            num_sources = source_embeddings.shape[0]
            weights = torch.arange(
                num_sources, 0, -1, dtype=torch.float32, device=source_embeddings.device
            )
            weights = weights / weights.sum()
            return (source_embeddings * weights.unsqueeze(1)).sum(dim=0)

        elif strategy == EmbeddingStrategy.MAX:
            return source_embeddings.max(dim=0).values

        elif strategy == EmbeddingStrategy.MIN:
            return source_embeddings.min(dim=0).values

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _map_single_embedding(
        self,
        target_model: AutoModelForCausalLM,
        target_id: int,
        source_ids: list[int],
        strategy: EmbeddingStrategy,
    ) -> None:
        """Map embeddings for a single token."""
        with torch.no_grad():
            source_embeddings = self.org_model.model.embed_tokens.weight[source_ids]
            combined_embedding = self._combine_embeddings(source_embeddings, strategy)
            target_model.model.embed_tokens.weight[target_id] = combined_embedding

    def clone(
        self,
        strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN,
        verbose: bool = True,
    ) -> AutoModelForCausalLM:
        """
        Clone the model with new tokenizer embeddings.

        Args:
            strategy: Strategy for combining embeddings when a target token
                     maps to multiple source tokens
            verbose: Whether to print progress

        Returns:
            Cloned model with new embeddings
        """
        if not self.token_id_map:
            self.build_token_id_map(verbose=verbose)

        if verbose:
            print(f"Cloning model with strategy: {strategy.value}")

        target_vocab_size = len(self.target_tokenizer)

        # Get model config and update vocab size and special token IDs
        config = self.org_model.config
        config.vocab_size = target_vocab_size
        config.eos_token_id = self.target_tokenizer.eos_token_id
        config.bos_token_id = self.target_tokenizer.bos_token_id
        config.pad_token_id = self.target_tokenizer.pad_token_id

        # Create new model with updated config
        target_model = AutoModelForCausalLM.from_config(config)
        target_model.resize_token_embeddings(target_vocab_size)

        assert target_model.config.vocab_size == len(self.target_tokenizer), (
            f"Model vocab size ({target_model.config.vocab_size}) != "
            f"tokenizer vocab size ({len(self.target_tokenizer)})"
        )

        if verbose:
            print(
                f"Model vocab size: {target_model.config.vocab_size}, "
                f"Tokenizer vocab size: {len(self.target_tokenizer)}"
            )
            print("Copying weights from original model...")

        with torch.no_grad():
            org_state_dict = self.org_model.state_dict()
            target_state_dict = target_model.state_dict()

            for name, param in org_state_dict.items():
                if "embed_tokens" not in name and "lm_head" not in name:
                    if name in target_state_dict:
                        target_state_dict[name].copy_(param)

            target_model.load_state_dict(target_state_dict)

        if verbose:
            print("Mapping embeddings...")

        errors = []
        for i in range(target_vocab_size):
            if i not in self.token_id_map:
                errors.append(i)
                continue

            source_ids = self.token_id_map[i]
            if not source_ids:
                errors.append(i)
                continue

            try:
                self._map_single_embedding(target_model, i, source_ids, strategy)
            except Exception as e:
                errors.append(i)
                if verbose:
                    print(f"Error mapping token {i}: {e}")

            if verbose and (i + 1) % 1000 == 0:
                print(f"Mapped {i + 1}/{target_vocab_size} embeddings")

        if verbose:
            if errors:
                print(f"Warning: {len(errors)} tokens could not be mapped: {errors[:10]}...")
            print("Model cloning complete!")

        if self.org_model.config.tie_word_embeddings:
            target_model.tie_weights()

        return target_model

    def clone_with_lm_head(
        self,
        strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN,
        verbose: bool = True,
    ) -> AutoModelForCausalLM:
        """
        Clone the model with new tokenizer embeddings and also update lm_head.

        Some models have separate lm_head weights (not tied with embeddings).
        This method handles both cases.

        Args:
            strategy: Strategy for combining embeddings
            verbose: Whether to print progress

        Returns:
            Cloned model with new embeddings and lm_head
        """
        target_model = self.clone(strategy=strategy, verbose=verbose)

        if not self.org_model.config.tie_word_embeddings:
            if verbose:
                print("Mapping lm_head weights...")

            target_vocab_size = len(self.target_tokenizer)

            with torch.no_grad():
                for i in range(target_vocab_size):
                    if i not in self.token_id_map:
                        continue

                    source_ids = self.token_id_map[i]
                    if not source_ids:
                        continue

                    source_weights = self.org_model.lm_head.weight[source_ids]
                    combined_weight = self._combine_embeddings(source_weights, strategy)
                    target_model.lm_head.weight[i] = combined_weight

            if verbose:
                print("lm_head mapping complete!")

        return target_model

    def create_pruned_tokenizer(
        self,
        keep_token_ids: list[int],
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> tuple[AutoTokenizer, dict[int, int]]:
        """
        Create a new tokenizer with only the specified tokens.

        Args:
            keep_token_ids: List of token IDs to keep from the original tokenizer
            save_path: Optional path to save the pruned tokenizer
            verbose: Whether to print progress

        Returns:
            Tuple of (new_tokenizer, id_mapping) where id_mapping maps old_id -> new_id
        """
        if verbose:
            print(f"Creating pruned tokenizer with {len(keep_token_ids)} tokens...")

        org_vocab = self.org_tokenizer.vocab

        new_vocab = {}
        id_mapping = {}

        special_token_ids = set()
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
            token_id = getattr(self.org_tokenizer, attr, None)
            if token_id is not None:
                special_token_ids.add(token_id)

        keep_token_ids = list(set(keep_token_ids) | special_token_ids)

        new_id = 0
        for old_id in sorted(keep_token_ids):
            if old_id < len(org_vocab):
                token_str = None
                for tok, tid in org_vocab.items():
                    if tid == old_id:
                        token_str = tok
                        break

                if token_str is not None:
                    new_vocab[token_str] = new_id
                    id_mapping[old_id] = new_id
                    new_id += 1

        if verbose:
            print(f"New vocab size: {len(new_vocab)}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.org_tokenizer.save_pretrained(tmp_dir)

            tokenizer_json_path = os.path.join(tmp_dir, "tokenizer.json")
            if os.path.exists(tokenizer_json_path):
                with open(tokenizer_json_path, "r") as f:
                    tokenizer_data = json.load(f)

                if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
                    tokenizer_data["model"]["vocab"] = new_vocab

                with open(tokenizer_json_path, "w") as f:
                    json.dump(tokenizer_data, f)

            new_tokenizer = AutoTokenizer.from_pretrained(tmp_dir)

        if self.org_tokenizer.bos_token_id in id_mapping:
            new_tokenizer.bos_token_id = id_mapping[self.org_tokenizer.bos_token_id]
        if self.org_tokenizer.eos_token_id in id_mapping:
            new_tokenizer.eos_token_id = id_mapping[self.org_tokenizer.eos_token_id]
        if self.org_tokenizer.pad_token_id in id_mapping:
            new_tokenizer.pad_token_id = id_mapping[self.org_tokenizer.pad_token_id]

        if save_path:
            new_tokenizer.save_pretrained(save_path)
            if verbose:
                print(f"Pruned tokenizer saved to {save_path}")

        return new_tokenizer, id_mapping

    def clone_with_vocab_pruning(
        self,
        keep_token_ids: Optional[list[int]] = None,
        vocab_size: Optional[int] = None,
        pruning_config: Optional[PruningConfig] = None,
        verbose: bool = True,
    ) -> tuple[AutoModelForCausalLM, "VocabPrunedTokenizer", dict[int, int]]:
        """
        Clone model with a reduced embedding table (fewer tokens in the model).

        Returns a VocabPrunedTokenizer that automatically remaps token IDs
        so you can use the model with natural text input.

        Uses direct 1:1 embedding mapping since we're keeping original tokens.
        No embedding combining needed.

        Args:
            keep_token_ids: Specific token IDs to keep. If None, uses vocab_size.
            vocab_size: Number of tokens to keep (uses first N tokens).
                       Ignored if keep_token_ids is provided.
            pruning_config: Optional architecture pruning config
            verbose: Whether to print progress

        Returns:
            Tuple of (cloned_model, vocab_pruned_tokenizer, id_mapping)
            - vocab_pruned_tokenizer: Wrapper that remaps tokens automatically
            - id_mapping: Maps old_id -> new_id for the embedding table
        """
        from transformer_cloner.vocab_pruned_tokenizer import VocabPrunedTokenizer
        
        if keep_token_ids is None:
            if vocab_size is None:
                raise ValueError("Either keep_token_ids or vocab_size must be provided")
            keep_token_ids = list(range(vocab_size))

        # Ensure special tokens are included
        special_token_ids = set()
        for attr in ["bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"]:
            token_id = getattr(self.org_tokenizer, attr, None)
            if token_id is not None:
                special_token_ids.add(token_id)

        keep_token_ids = sorted(set(keep_token_ids) | special_token_ids)

        if verbose:
            print(f"Cloning with vocab pruning: {len(keep_token_ids)} tokens")

        # Build id mapping (old_id -> new_id)
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(keep_token_ids)}
        new_vocab_size = len(keep_token_ids)

        if verbose:
            print(f"New vocab size: {new_vocab_size}")

        # Create config
        if pruning_config:
            validation_errors = pruning_config.validate(self.org_model.config)
            if validation_errors:
                error_msg = "Invalid pruning configuration:\n" + "\n".join(
                    f"  - {e}" for e in validation_errors
                )
                raise ValueError(error_msg)
            config = self.org_model.config.__class__(**self.org_model.config.to_dict())
            if pruning_config.hidden_size:
                config.hidden_size = pruning_config.hidden_size
            if pruning_config.num_hidden_layers:
                config.num_hidden_layers = pruning_config.num_hidden_layers
            if pruning_config.intermediate_size:
                config.intermediate_size = pruning_config.intermediate_size
            if pruning_config.num_attention_heads:
                config.num_attention_heads = pruning_config.num_attention_heads
            if pruning_config.num_key_value_heads and hasattr(config, "num_key_value_heads"):
                config.num_key_value_heads = pruning_config.num_key_value_heads
            if pruning_config.head_dim and hasattr(config, "head_dim"):
                config.head_dim = pruning_config.head_dim
        else:
            config = self.org_model.config.__class__(**self.org_model.config.to_dict())

        config.vocab_size = new_vocab_size
        
        # Update special token IDs to their new positions
        if self.org_tokenizer.eos_token_id in id_mapping:
            config.eos_token_id = id_mapping[self.org_tokenizer.eos_token_id]
        if self.org_tokenizer.bos_token_id in id_mapping:
            config.bos_token_id = id_mapping[self.org_tokenizer.bos_token_id]
        if self.org_tokenizer.pad_token_id in id_mapping:
            config.pad_token_id = id_mapping[self.org_tokenizer.pad_token_id]

        if verbose:
            print(f"Creating model with vocab_size={new_vocab_size}")

        target_model = AutoModelForCausalLM.from_config(config)
        target_model.resize_token_embeddings(new_vocab_size)

        if verbose:
            print("Copying weights from original model...")

        org_hidden_size = self.org_model.config.hidden_size
        new_hidden_size = config.hidden_size
        new_num_layers = config.num_hidden_layers

        with torch.no_grad():
            org_state_dict = self.org_model.state_dict()
            target_state_dict = target_model.state_dict()

            for name, target_param in target_state_dict.items():
                if "embed_tokens" in name or "lm_head" in name:
                    continue

                if self._should_skip_layer(
                    name, new_num_layers, self.org_model.config.num_hidden_layers
                ):
                    continue

                source_name = self._map_layer_name(
                    name, new_num_layers, self.org_model.config.num_hidden_layers
                )
                if source_name not in org_state_dict:
                    continue

                source_param = org_state_dict[source_name]

                if source_param.shape != target_param.shape:
                    pruned_param = self._prune_weight(
                        source_param,
                        target_param.shape,
                        name,
                        org_hidden_size,
                        new_hidden_size,
                        self.org_model.config.intermediate_size,
                        config.intermediate_size,
                        self.org_model.config.num_attention_heads,
                        config.num_attention_heads,
                        getattr(
                            self.org_model.config,
                            "num_key_value_heads",
                            self.org_model.config.num_attention_heads,
                        ),
                        getattr(config, "num_key_value_heads", config.num_attention_heads),
                        getattr(
                            self.org_model.config,
                            "head_dim",
                            org_hidden_size // self.org_model.config.num_attention_heads,
                        ),
                        getattr(
                            config, "head_dim", new_hidden_size // config.num_attention_heads
                        ),
                    )
                    if pruned_param is not None:
                        target_state_dict[name].copy_(pruned_param)
                else:
                    target_state_dict[name].copy_(source_param)

            target_model.load_state_dict(target_state_dict)

        if verbose:
            print("Mapping embeddings (direct 1:1)...")

        with torch.no_grad():
            for old_id, new_id in id_mapping.items():
                if new_hidden_size < org_hidden_size:
                    target_model.model.embed_tokens.weight[new_id] = (
                        self.org_model.model.embed_tokens.weight[old_id, :new_hidden_size].clone()
                    )
                else:
                    target_model.model.embed_tokens.weight[new_id] = (
                        self.org_model.model.embed_tokens.weight[old_id].clone()
                    )

        if verbose:
            print(f"Mapped {len(id_mapping)} embeddings directly")

        if self.org_model.config.tie_word_embeddings:
            target_model.tie_weights()

        # Create wrapper tokenizer
        # Use first token as UNK (usually <bos> or special token that exists)
        unk_new_id = id_mapping.get(
            self.org_tokenizer.unk_token_id,
            id_mapping.get(self.org_tokenizer.pad_token_id, 0)
        )
        
        pruned_tokenizer = VocabPrunedTokenizer(
            original_tokenizer=self.org_tokenizer,
            id_mapping=id_mapping,
            unk_token_id=unk_new_id,
        )

        if verbose:
            print("Vocab-pruned model cloning complete!")
            print(f"Created VocabPrunedTokenizer with {len(id_mapping)} tokens")

        return target_model, pruned_tokenizer, id_mapping

    def clone_pruned(
        self,
        pruning_config: PruningConfig,
        strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN,
        verbose: bool = True,
    ) -> AutoModelForCausalLM:
        """
        Clone the model with pruned architecture (smaller hidden size, fewer layers, etc.)

        This creates a smaller model by slicing weights from the original model.
        Useful for creating distillation targets or smaller models for fine-tuning.

        Args:
            pruning_config: Configuration specifying the target architecture dimensions
            strategy: Strategy for combining embeddings
            verbose: Whether to print progress

        Returns:
            Pruned and cloned model

        Raises:
            ValueError: If pruning_config is invalid for this model
        """
        validation_errors = pruning_config.validate(self.org_model.config)
        if validation_errors:
            error_msg = "Invalid pruning configuration:\n" + "\n".join(
                f"  - {e}" for e in validation_errors
            )
            raise ValueError(error_msg)

        if not self.token_id_map:
            self.build_token_id_map(verbose=verbose)

        if verbose:
            print(f"Cloning pruned model with strategy: {strategy.value}")

        config = self.org_model.config.__class__(**self.org_model.config.to_dict())

        org_hidden_size = config.hidden_size
        org_num_layers = config.num_hidden_layers
        org_intermediate_size = config.intermediate_size
        org_num_heads = config.num_attention_heads
        org_num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        org_head_dim = getattr(config, "head_dim", org_hidden_size // org_num_heads)

        new_hidden_size = pruning_config.hidden_size or org_hidden_size
        new_num_layers = pruning_config.num_hidden_layers or org_num_layers
        new_intermediate_size = pruning_config.intermediate_size or org_intermediate_size
        new_num_heads = pruning_config.num_attention_heads or org_num_heads
        new_num_kv_heads = pruning_config.num_key_value_heads or org_num_kv_heads
        new_head_dim = pruning_config.head_dim or org_head_dim

        config.hidden_size = new_hidden_size
        config.num_hidden_layers = new_num_layers
        config.intermediate_size = new_intermediate_size
        config.num_attention_heads = new_num_heads
        if hasattr(config, "num_key_value_heads"):
            config.num_key_value_heads = new_num_kv_heads
        if hasattr(config, "head_dim"):
            config.head_dim = new_head_dim

        target_vocab_size = len(self.target_tokenizer)
        config.vocab_size = target_vocab_size
        config.eos_token_id = self.target_tokenizer.eos_token_id
        config.bos_token_id = self.target_tokenizer.bos_token_id
        config.pad_token_id = self.target_tokenizer.pad_token_id

        if verbose:
            print(
                f"Original: hidden={org_hidden_size}, layers={org_num_layers}, "
                f"intermediate={org_intermediate_size}, heads={org_num_heads}, kv_heads={org_num_kv_heads}"
            )
            print(
                f"Pruned:   hidden={new_hidden_size}, layers={new_num_layers}, "
                f"intermediate={new_intermediate_size}, heads={new_num_heads}, kv_heads={new_num_kv_heads}"
            )

        target_model = AutoModelForCausalLM.from_config(config)
        target_model.resize_token_embeddings(target_vocab_size)

        if verbose:
            print(f"Model vocab size: {target_model.config.vocab_size}")
            print("Copying and pruning weights from original model...")

        with torch.no_grad():
            org_state_dict = self.org_model.state_dict()
            target_state_dict = target_model.state_dict()

            for name, target_param in target_state_dict.items():
                if "embed_tokens" in name or "lm_head" in name:
                    continue

                if self._should_skip_layer(name, new_num_layers, org_num_layers):
                    continue

                source_name = self._map_layer_name(name, new_num_layers, org_num_layers)
                if source_name not in org_state_dict:
                    continue

                source_param = org_state_dict[source_name]

                pruned_param = self._prune_weight(
                    source_param,
                    target_param.shape,
                    name,
                    org_hidden_size,
                    new_hidden_size,
                    org_intermediate_size,
                    new_intermediate_size,
                    org_num_heads,
                    new_num_heads,
                    org_num_kv_heads,
                    new_num_kv_heads,
                    org_head_dim,
                    new_head_dim,
                )

                if pruned_param is not None:
                    target_state_dict[name].copy_(pruned_param)

            target_model.load_state_dict(target_state_dict)

        if verbose:
            print("Mapping embeddings with pruning...")

        errors = []
        for i in range(target_vocab_size):
            if i not in self.token_id_map:
                errors.append(i)
                continue

            source_ids = self.token_id_map[i]
            if not source_ids:
                errors.append(i)
                continue

            try:
                self._map_single_embedding_pruned(
                    target_model, i, source_ids, strategy, org_hidden_size, new_hidden_size
                )
            except Exception as e:
                errors.append(i)
                if verbose:
                    print(f"Error mapping token {i}: {e}")

            if verbose and (i + 1) % 5000 == 0:
                print(f"Mapped {i + 1}/{target_vocab_size} embeddings")

        if verbose:
            if errors:
                print(f"Warning: {len(errors)} tokens could not be mapped")
            print("Pruned model cloning complete!")

        if self.org_model.config.tie_word_embeddings:
            target_model.tie_weights()

        return target_model

    def _should_skip_layer(
        self, name: str, new_num_layers: int, org_num_layers: int
    ) -> bool:
        """Check if a layer should be skipped based on layer pruning."""
        match = re.search(r"layers\.(\d+)\.", name)
        if match:
            layer_idx = int(match.group(1))
            return layer_idx >= new_num_layers
        return False

    def _map_layer_name(
        self, name: str, new_num_layers: int, org_num_layers: int
    ) -> str:
        """Map layer name (currently 1:1, takes first N layers)."""
        return name

    def _prune_weight(
        self,
        source: torch.Tensor,
        target_shape: tuple,
        name: str,
        org_hidden: int,
        new_hidden: int,
        org_intermediate: int,
        new_intermediate: int,
        org_heads: int,
        new_heads: int,
        org_kv_heads: int,
        new_kv_heads: int,
        org_head_dim: int,
        new_head_dim: int,
    ) -> Optional[torch.Tensor]:
        """Prune a weight tensor to match target shape by slicing."""
        if source.shape == target_shape:
            return source.clone()

        if len(source.shape) == 1:
            return source[: target_shape[0]].clone()

        elif len(source.shape) == 2:
            out_features, in_features = target_shape
            src_out, src_in = source.shape

            pruned = source[: min(out_features, src_out), : min(in_features, src_in)]

            if pruned.shape != target_shape:
                result = torch.zeros(target_shape, dtype=source.dtype)
                result[: pruned.shape[0], : pruned.shape[1]] = pruned
                return result

            return pruned.clone()

        else:
            slices = tuple(
                slice(0, min(s, t)) for s, t in zip(source.shape, target_shape)
            )
            pruned = source[slices].clone()

            if pruned.shape != target_shape:
                result = torch.zeros(target_shape, dtype=source.dtype)
                result[slices] = pruned
                return result

            return pruned

    def _map_single_embedding_pruned(
        self,
        target_model: AutoModelForCausalLM,
        target_id: int,
        source_ids: list[int],
        strategy: EmbeddingStrategy,
        org_hidden_size: int,
        new_hidden_size: int,
    ) -> None:
        """Map embeddings for a single token with dimension pruning."""
        with torch.no_grad():
            source_embeddings = self.org_model.model.embed_tokens.weight[source_ids]
            source_embeddings = source_embeddings[:, :new_hidden_size]
            combined_embedding = self._combine_embeddings(source_embeddings, strategy)
            target_model.model.embed_tokens.weight[target_id] = combined_embedding

    def get_token_info(self, token: str) -> dict:
        """
        Get information about how a token is mapped.

        Args:
            token: The token string to look up

        Returns:
            Dictionary with token information
        """
        target_id = self.target_tokenizer.vocab.get(token)
        if target_id is None:
            return {"error": f"Token '{token}' not found in target tokenizer"}

        source_ids = self.token_id_map.get(target_id, [])
        source_tokens = [self.org_tokenizer.decode([sid]) for sid in source_ids]

        return {
            "token": token,
            "target_id": target_id,
            "source_ids": source_ids,
            "source_tokens": source_tokens,
        }

    def print_vocab_samples(self, n: int = 10) -> None:
        """Print sample vocabulary entries from both tokenizers."""
        print("Original tokenizer samples:")
        org_vocab_keys = list(self.org_tokenizer.vocab.keys())
        for i in range(min(n, len(org_vocab_keys))):
            vocab = org_vocab_keys[i]
            vocab_id = self.org_tokenizer.vocab[vocab]
            print(f"  {i}: {repr(vocab)}\t{vocab_id}")
        print(f"Total: {len(org_vocab_keys)} tokens\n")

        print("Target tokenizer samples:")
        target_vocab_keys = list(self.target_tokenizer.vocab.keys())
        for i in range(min(n, len(target_vocab_keys))):
            vocab = target_vocab_keys[i]
            vocab_id = self.target_tokenizer.vocab[vocab]
            print(f"  {i}: {repr(vocab)}\t{vocab_id}")
        print(f"Total: {len(target_vocab_keys)} tokens")
