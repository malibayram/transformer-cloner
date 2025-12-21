"""Wrapper tokenizer that remaps token IDs for vocab-pruned models."""

from typing import Optional, Union
import torch


class VocabPrunedTokenizer:
    """
    A wrapper tokenizer that remaps token IDs from the original tokenizer
    to the pruned vocabulary space.
    
    This allows using a vocab-pruned model with natural text input.
    Tokens not in the pruned vocabulary are mapped to the UNK token.
    
    Example:
        >>> pruned_tokenizer = VocabPrunedTokenizer(
        ...     original_tokenizer=tokenizer,
        ...     id_mapping={0: 0, 1: 1, 2: 2, 100: 3, 200: 4},
        ...     unk_token_id=2,  # map unknown to pad or unk
        ... )
        >>> input_ids = pruned_tokenizer("Hello world", return_tensors="pt")
    """
    
    def __init__(
        self,
        original_tokenizer,
        id_mapping: dict[int, int],
        unk_token_id: int = 0,
    ):
        """
        Initialize the wrapper tokenizer.
        
        Args:
            original_tokenizer: The original HuggingFace tokenizer
            id_mapping: Mapping from original token IDs to new token IDs
            unk_token_id: New token ID to use for unmapped tokens
        """
        self.original_tokenizer = original_tokenizer
        self.id_mapping = id_mapping
        self.reverse_mapping = {v: k for k, v in id_mapping.items()}
        self.unk_token_id = unk_token_id
        self._vocab_size = len(id_mapping)
        
        # Copy important attributes from original tokenizer
        self.model_max_length = original_tokenizer.model_max_length
        self.padding_side = original_tokenizer.padding_side
        self.truncation_side = getattr(original_tokenizer, 'truncation_side', 'right')
        
        # Map special token IDs
        self.bos_token_id = id_mapping.get(original_tokenizer.bos_token_id)
        self.eos_token_id = id_mapping.get(original_tokenizer.eos_token_id)
        self.pad_token_id = id_mapping.get(original_tokenizer.pad_token_id, self.unk_token_id)
        self.unk_token_id_mapped = self.unk_token_id
        
        # Copy special tokens
        self.bos_token = original_tokenizer.bos_token
        self.eos_token = original_tokenizer.eos_token
        self.pad_token = original_tokenizer.pad_token
        self.unk_token = original_tokenizer.unk_token
    
    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size
    
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size
    
    def _remap_ids(self, input_ids: list[int]) -> list[int]:
        """Remap token IDs from original to pruned vocabulary."""
        return [self.id_mapping.get(tid, self.unk_token_id) for tid in input_ids]
    
    def _remap_ids_reverse(self, input_ids: list[int]) -> list[int]:
        """Remap token IDs from pruned back to original vocabulary."""
        return [self.reverse_mapping.get(tid, 0) for tid in input_ids]
    
    def __call__(
        self,
        text: Union[str, list[str]],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        """
        Tokenize text and remap to pruned vocabulary.
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Whether to add special tokens
            padding: Padding strategy
            truncation: Truncation strategy
            max_length: Maximum length
            return_tensors: Return type ('pt' for PyTorch)
            **kwargs: Additional arguments passed to original tokenizer
            
        Returns:
            TokenizerOutput with remapped input_ids
        """
        # Use original tokenizer
        encoded = self.original_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=None,  # Get lists first
            **kwargs,
        )
        
        # Remap input_ids
        if isinstance(text, str):
            encoded['input_ids'] = self._remap_ids(encoded['input_ids'])
        else:
            encoded['input_ids'] = [self._remap_ids(ids) for ids in encoded['input_ids']]
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            import torch
            if isinstance(text, str):
                encoded['input_ids'] = torch.tensor([encoded['input_ids']])
                if 'attention_mask' in encoded:
                    encoded['attention_mask'] = torch.tensor([encoded['attention_mask']])
            else:
                encoded['input_ids'] = torch.tensor(encoded['input_ids'])
                if 'attention_mask' in encoded:
                    encoded['attention_mask'] = torch.tensor(encoded['attention_mask'])
        
        return encoded
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        **kwargs,
    ) -> list[int]:
        """Encode text to remapped token IDs."""
        original_ids = self.original_tokenizer.encode(
            text, add_special_tokens=add_special_tokens, **kwargs
        )
        return self._remap_ids(original_ids)
    
    def decode(
        self,
        token_ids: Union[list[int], torch.Tensor],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        """Decode remapped token IDs back to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Handle nested lists (batch)
        if token_ids and isinstance(token_ids[0], list):
            return [self.decode(ids, skip_special_tokens, **kwargs) for ids in token_ids]
        
        # Remap back to original IDs
        original_ids = self._remap_ids_reverse(token_ids)
        return self.original_tokenizer.decode(
            original_ids, skip_special_tokens=skip_special_tokens, **kwargs
        )
    
    def batch_decode(
        self,
        sequences: Union[list[list[int]], torch.Tensor],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> list[str]:
        """Decode a batch of remapped token IDs."""
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        
        return [
            self.decode(seq, skip_special_tokens=skip_special_tokens, **kwargs)
            for seq in sequences
        ]
    
    def get_vocab(self) -> dict[str, int]:
        """Get the pruned vocabulary."""
        original_vocab = self.original_tokenizer.get_vocab()
        pruned_vocab = {}
        
        for token, original_id in original_vocab.items():
            if original_id in self.id_mapping:
                pruned_vocab[token] = self.id_mapping[original_id]
        
        return pruned_vocab
    
    @property
    def vocab(self) -> dict[str, int]:
        """Get the pruned vocabulary."""
        return self.get_vocab()
    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the wrapper tokenizer.
        
        Saves the original tokenizer plus a mapping file.
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save original tokenizer
        self.original_tokenizer.save_pretrained(save_directory, **kwargs)
        
        # Save mapping
        mapping_path = os.path.join(save_directory, "vocab_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump({
                "id_mapping": {str(k): v for k, v in self.id_mapping.items()},
                "unk_token_id": self.unk_token_id,
                "vocab_size": self._vocab_size,
            }, f, indent=2)
        
        print(f"Saved pruned tokenizer to {save_directory}")
        print(f"  - vocab_mapping.json contains the ID mapping")
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """
        Load a saved wrapper tokenizer.
        
        Args:
            path: Path to the saved tokenizer directory
            **kwargs: Additional arguments for AutoTokenizer
            
        Returns:
            VocabPrunedTokenizer instance
        """
        import os
        import json
        from transformers import AutoTokenizer
        
        # Load mapping
        mapping_path = os.path.join(path, "vocab_mapping.json")
        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)
        
        id_mapping = {int(k): v for k, v in mapping_data["id_mapping"].items()}
        unk_token_id = mapping_data["unk_token_id"]
        
        # Load original tokenizer
        original_tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        
        return cls(
            original_tokenizer=original_tokenizer,
            id_mapping=id_mapping,
            unk_token_id=unk_token_id,
        )
    
    def __repr__(self) -> str:
        return (
            f"VocabPrunedTokenizer(vocab_size={self._vocab_size}, "
            f"original_vocab_size={len(self.original_tokenizer)})"
        )
