"""Configuration for model architecture pruning."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PruningConfig:
    """
    Configuration for model pruning.

    Allows reducing the model architecture by specifying smaller dimensions.
    Set any value to None to keep the original model's value.

    Attributes:
        hidden_size: Embedding dimension (e.g., 768 -> 512)
        num_hidden_layers: Number of transformer layers (e.g., 12 -> 6)
        intermediate_size: FFN intermediate dimension (e.g., 3072 -> 1536)
        num_attention_heads: Number of attention heads (e.g., 12 -> 8)
        num_key_value_heads: Number of KV heads for GQA (e.g., 4 -> 2)
        head_dim: Dimension per attention head (e.g., 64 -> 32)

    Example:
        >>> config = PruningConfig(
        ...     hidden_size=512,
        ...     num_hidden_layers=6,
        ...     intermediate_size=1536,
        ... )
    """

    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None

    def validate(self, original_config) -> list[str]:
        """
        Validate the pruning config against the original model config.

        Args:
            original_config: The original model's config object

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []

        # Get original values
        org_hidden = original_config.hidden_size
        org_layers = original_config.num_hidden_layers
        org_intermediate = original_config.intermediate_size
        org_heads = original_config.num_attention_heads
        org_kv_heads = getattr(original_config, "num_key_value_heads", org_heads)
        org_head_dim = getattr(original_config, "head_dim", org_hidden // org_heads)

        # Get new values (use original if None)
        new_hidden = self.hidden_size or org_hidden
        new_layers = self.num_hidden_layers or org_layers
        new_intermediate = self.intermediate_size or org_intermediate
        new_heads = self.num_attention_heads or org_heads
        new_kv_heads = self.num_key_value_heads or org_kv_heads
        new_head_dim = self.head_dim or org_head_dim

        # Check dimensions don't exceed original
        if new_hidden > org_hidden:
            errors.append(
                f"hidden_size ({new_hidden}) cannot exceed original ({org_hidden})"
            )
        if new_layers > org_layers:
            errors.append(
                f"num_hidden_layers ({new_layers}) cannot exceed original ({org_layers})"
            )
        if new_intermediate > org_intermediate:
            errors.append(
                f"intermediate_size ({new_intermediate}) cannot exceed original ({org_intermediate})"
            )
        if new_heads > org_heads:
            errors.append(
                f"num_attention_heads ({new_heads}) cannot exceed original ({org_heads})"
            )
        if new_kv_heads > org_kv_heads:
            errors.append(
                f"num_key_value_heads ({new_kv_heads}) cannot exceed original ({org_kv_heads})"
            )
        if new_head_dim > org_head_dim:
            errors.append(
                f"head_dim ({new_head_dim}) cannot exceed original ({org_head_dim})"
            )

        # Check positive values
        if new_hidden <= 0:
            errors.append(f"hidden_size must be positive, got {new_hidden}")
        if new_layers <= 0:
            errors.append(f"num_hidden_layers must be positive, got {new_layers}")
        if new_intermediate <= 0:
            errors.append(f"intermediate_size must be positive, got {new_intermediate}")
        if new_heads <= 0:
            errors.append(f"num_attention_heads must be positive, got {new_heads}")
        if new_kv_heads <= 0:
            errors.append(f"num_key_value_heads must be positive, got {new_kv_heads}")

        # Check attention head constraints
        if new_heads % new_kv_heads != 0:
            errors.append(
                f"num_attention_heads ({new_heads}) must be divisible by "
                f"num_key_value_heads ({new_kv_heads})"
            )

        # Check head_dim compatibility with hidden_size
        # Note: Some models (like Gemma) use explicit head_dim that doesn't follow
        # the hidden_size = heads * head_dim pattern. Only check this constraint
        # if the ORIGINAL model follows this pattern.
        if hasattr(original_config, "head_dim"):
            # Model uses explicit head_dim - only validate if original follows the rule
            org_follows_rule = org_hidden >= org_heads * org_head_dim
            if org_follows_rule and new_hidden < new_heads * new_head_dim:
                errors.append(
                    f"hidden_size ({new_hidden}) should be >= "
                    f"num_attention_heads * head_dim ({new_heads * new_head_dim})"
                )
        else:
            # Model derives head_dim from hidden_size / num_heads
            if new_hidden % new_heads != 0:
                errors.append(
                    f"hidden_size ({new_hidden}) must be divisible by "
                    f"num_attention_heads ({new_heads})"
                )

        return errors
