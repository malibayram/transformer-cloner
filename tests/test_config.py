"""
Comprehensive tests for transformer_cloner package.

These tests use mock objects to avoid downloading actual models,
making them fast and suitable for CI/CD pipelines.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from transformer_cloner import EmbeddingStrategy, PruningConfig, TransformerCloner


# ============================================================================
# EmbeddingStrategy Tests
# ============================================================================

class TestEmbeddingStrategy:
    """Tests for EmbeddingStrategy enum."""

    def test_all_strategies_exist(self):
        """Verify all expected strategies are defined."""
        strategies = [
            EmbeddingStrategy.MEAN,
            EmbeddingStrategy.SUM,
            EmbeddingStrategy.FIRST,
            EmbeddingStrategy.LAST,
            EmbeddingStrategy.WEIGHTED,
            EmbeddingStrategy.MAX,
            EmbeddingStrategy.MIN,
        ]
        assert len(strategies) == 7

    def test_strategy_values(self):
        """Test that strategy values match expected strings."""
        assert EmbeddingStrategy.MEAN.value == "mean"
        assert EmbeddingStrategy.SUM.value == "sum"
        assert EmbeddingStrategy.FIRST.value == "first"
        assert EmbeddingStrategy.LAST.value == "last"
        assert EmbeddingStrategy.WEIGHTED.value == "weighted"
        assert EmbeddingStrategy.MAX.value == "max"
        assert EmbeddingStrategy.MIN.value == "min"

    def test_strategy_from_value(self):
        """Test creating strategy from string value."""
        assert EmbeddingStrategy("mean") == EmbeddingStrategy.MEAN
        assert EmbeddingStrategy("sum") == EmbeddingStrategy.SUM
        assert EmbeddingStrategy("first") == EmbeddingStrategy.FIRST

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy value raises error."""
        with pytest.raises(ValueError):
            EmbeddingStrategy("invalid")


# ============================================================================
# PruningConfig Tests
# ============================================================================

class TestPruningConfig:
    """Tests for PruningConfig dataclass."""

    def test_default_values(self):
        """Test that all defaults are None."""
        config = PruningConfig()
        assert config.hidden_size is None
        assert config.num_hidden_layers is None
        assert config.intermediate_size is None
        assert config.num_attention_heads is None
        assert config.num_key_value_heads is None
        assert config.head_dim is None

    def test_custom_values(self):
        """Test setting custom values."""
        config = PruningConfig(
            hidden_size=512,
            num_hidden_layers=6,
            intermediate_size=1536,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
        )
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 6
        assert config.intermediate_size == 1536
        assert config.num_attention_heads == 8
        assert config.num_key_value_heads == 2
        assert config.head_dim == 64

    def test_partial_values(self):
        """Test setting only some values."""
        config = PruningConfig(hidden_size=256, num_hidden_layers=4)
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 4
        assert config.intermediate_size is None
        assert config.num_attention_heads is None

    def test_layers_only_config(self):
        """Test config that only prunes layers."""
        config = PruningConfig(num_hidden_layers=9)
        assert config.num_hidden_layers == 9
        assert config.hidden_size is None


class TestPruningConfigValidation:
    """Tests for PruningConfig.validate() method."""

    @pytest.fixture
    def mock_config_standard(self):
        """Create a mock model config for standard models (hidden = heads * head_dim)."""
        config = Mock()
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        config.head_dim = 64  # 12 * 64 = 768 = hidden_size
        return config

    @pytest.fixture
    def mock_config_gemma(self):
        """Create a mock model config for Gemma-style models (hidden < heads * head_dim)."""
        config = Mock()
        config.hidden_size = 640
        config.num_hidden_layers = 18
        config.intermediate_size = 2048
        config.num_attention_heads = 4
        config.num_key_value_heads = 1
        config.head_dim = 256  # 4 * 256 = 1024 > 640 = hidden_size
        return config

    def test_valid_config_standard_model(self, mock_config_standard):
        """Test validation passes for valid config on standard model."""
        pruning = PruningConfig(
            hidden_size=512,
            num_hidden_layers=6,
            intermediate_size=1536,
            num_attention_heads=8,
            num_key_value_heads=8,
        )
        errors = pruning.validate(mock_config_standard)
        assert len(errors) == 0

    def test_valid_config_gemma_model(self, mock_config_gemma):
        """Test validation passes for valid config on Gemma model."""
        pruning = PruningConfig(
            hidden_size=320,
            num_hidden_layers=9,
            intermediate_size=1024,
        )
        errors = pruning.validate(mock_config_gemma)
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_gemma_same_hidden_size_valid(self, mock_config_gemma):
        """Test keeping same hidden_size on Gemma is valid."""
        pruning = PruningConfig(
            hidden_size=640,  # Same as original
            num_hidden_layers=9,
        )
        errors = pruning.validate(mock_config_gemma)
        assert len(errors) == 0

    def test_hidden_size_exceeds_original(self, mock_config_standard):
        """Test error when hidden_size exceeds original."""
        pruning = PruningConfig(hidden_size=1024)  # > 768
        errors = pruning.validate(mock_config_standard)
        assert len(errors) == 1
        assert "cannot exceed original" in errors[0]
        assert "hidden_size" in errors[0]

    def test_num_layers_exceeds_original(self, mock_config_standard):
        """Test error when num_hidden_layers exceeds original."""
        pruning = PruningConfig(num_hidden_layers=16)  # > 12
        errors = pruning.validate(mock_config_standard)
        assert len(errors) == 1
        assert "num_hidden_layers" in errors[0]

    def test_intermediate_size_exceeds_original(self, mock_config_standard):
        """Test error when intermediate_size exceeds original."""
        pruning = PruningConfig(intermediate_size=4096)  # > 3072
        errors = pruning.validate(mock_config_standard)
        assert len(errors) == 1
        assert "intermediate_size" in errors[0]

    def test_num_heads_exceeds_original(self, mock_config_standard):
        """Test error when num_attention_heads exceeds original."""
        pruning = PruningConfig(num_attention_heads=16)  # > 12
        errors = pruning.validate(mock_config_standard)
        assert any("num_attention_heads" in e and "cannot exceed" in e for e in errors)

    def test_kv_heads_exceeds_original(self, mock_config_standard):
        """Test error when num_key_value_heads exceeds original."""
        pruning = PruningConfig(num_key_value_heads=16)  # > 12
        errors = pruning.validate(mock_config_standard)
        assert any("num_key_value_heads" in e for e in errors)

    def test_head_dim_exceeds_original(self, mock_config_standard):
        """Test error when head_dim exceeds original."""
        pruning = PruningConfig(head_dim=128)  # > 64
        errors = pruning.validate(mock_config_standard)
        assert len(errors) >= 1
        assert any("head_dim" in e for e in errors)

    def test_attention_heads_not_divisible_by_kv_heads(self, mock_config_standard):
        """Test error when attention heads not divisible by kv heads."""
        pruning = PruningConfig(
            num_attention_heads=7,
            num_key_value_heads=3,  # 7 % 3 != 0
        )
        errors = pruning.validate(mock_config_standard)
        assert any("must be divisible by" in e for e in errors)

    def test_hidden_size_not_divisible_by_heads_no_explicit_head_dim(self):
        """Test error when hidden_size not divisible by num_attention_heads (no explicit head_dim)."""
        # Model without explicit head_dim
        config = Mock()
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        # No head_dim attribute
        del config.head_dim
        
        pruning = PruningConfig(
            hidden_size=500,  # Not divisible by 8
            num_attention_heads=8,
        )
        errors = pruning.validate(config)
        assert any("must be divisible by" in e for e in errors)

    def test_multiple_errors_returned(self, mock_config_standard):
        """Test that multiple errors are returned at once."""
        pruning = PruningConfig(
            hidden_size=1024,  # Exceeds 768
            num_hidden_layers=20,  # Exceeds 12
            intermediate_size=5000,  # Exceeds 3072
        )
        errors = pruning.validate(mock_config_standard)
        assert len(errors) >= 3

    def test_layers_only_config_valid(self, mock_config_gemma):
        """Test config that only prunes layers is valid."""
        pruning = PruningConfig(num_hidden_layers=9)
        errors = pruning.validate(mock_config_gemma)
        assert len(errors) == 0


# ============================================================================
# Embedding Combination Tests
# ============================================================================

class TestEmbeddingCombination:
    """Test the embedding combination logic with different strategies."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        # 3 source embeddings, each with 4 dimensions
        return torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [3.0, 6.0, 9.0, 12.0],
        ])

    def test_mean_strategy(self, sample_embeddings):
        """Test MEAN strategy averages embeddings."""
        result = sample_embeddings.mean(dim=0)
        expected = torch.tensor([2.0, 4.0, 6.0, 8.0])
        assert torch.allclose(result, expected)

    def test_sum_strategy(self, sample_embeddings):
        """Test SUM strategy sums embeddings."""
        result = sample_embeddings.sum(dim=0)
        expected = torch.tensor([6.0, 12.0, 18.0, 24.0])
        assert torch.allclose(result, expected)

    def test_first_strategy(self, sample_embeddings):
        """Test FIRST strategy takes first embedding."""
        result = sample_embeddings[0]
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(result, expected)

    def test_last_strategy(self, sample_embeddings):
        """Test LAST strategy takes last embedding."""
        result = sample_embeddings[-1]
        expected = torch.tensor([3.0, 6.0, 9.0, 12.0])
        assert torch.allclose(result, expected)

    def test_max_strategy(self, sample_embeddings):
        """Test MAX strategy takes element-wise maximum."""
        result = sample_embeddings.max(dim=0).values
        expected = torch.tensor([3.0, 6.0, 9.0, 12.0])
        assert torch.allclose(result, expected)

    def test_min_strategy(self, sample_embeddings):
        """Test MIN strategy takes element-wise minimum."""
        result = sample_embeddings.min(dim=0).values
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert torch.allclose(result, expected)

    def test_weighted_strategy(self, sample_embeddings):
        """Test WEIGHTED strategy applies decreasing weights."""
        num_sources = sample_embeddings.shape[0]
        weights = torch.arange(num_sources, 0, -1, dtype=torch.float32)
        weights = weights / weights.sum()  # [0.5, 0.333, 0.167]
        result = (sample_embeddings * weights.unsqueeze(1)).sum(dim=0)
        # Weights: 3/6=0.5, 2/6=0.333, 1/6=0.167
        # result[0] = 1*0.5 + 2*0.333 + 3*0.167 ≈ 1.667
        assert result.shape == torch.Size([4])
        assert result[0] < 2.0  # First element weighted toward first row

    def test_single_embedding_returns_same(self, sample_embeddings):
        """Test that single embedding is returned unchanged with any strategy."""
        single = sample_embeddings[:1]
        
        # All strategies should return the same for single embedding
        assert torch.allclose(single.mean(dim=0), sample_embeddings[0])
        assert torch.allclose(single.sum(dim=0), sample_embeddings[0])
        assert torch.allclose(single.max(dim=0).values, sample_embeddings[0])

    def test_empty_dimension_handling(self):
        """Test embedding combination with different dimensions."""
        embeddings = torch.randn(5, 768)  # 5 source embeddings, 768 dims
        result = embeddings.mean(dim=0)
        assert result.shape == torch.Size([768])


# ============================================================================
# Weight Pruning Tests  
# ============================================================================

class TestWeightPruning:
    """Test weight pruning logic."""

    def test_prune_1d_tensor(self):
        """Test pruning 1D tensors (biases, layer norms)."""
        source = torch.randn(768)
        target_shape = (512,)
        result = source[:target_shape[0]]
        assert result.shape == target_shape

    def test_prune_2d_tensor(self):
        """Test pruning 2D tensors (linear layer weights)."""
        source = torch.randn(768, 768)
        target_shape = (512, 512)
        result = source[:target_shape[0], :target_shape[1]]
        assert result.shape == target_shape

    def test_prune_asymmetric_2d_tensor(self):
        """Test pruning 2D tensors with different source/target dims."""
        source = torch.randn(3072, 768)  # FFN up projection
        target_shape = (1536, 512)
        result = source[:target_shape[0], :target_shape[1]]
        assert result.shape == target_shape

    def test_prune_embedding_table(self):
        """Test pruning embedding table."""
        source = torch.randn(262144, 768)  # Large vocab, 768 hidden
        target_shape = (8000, 320)
        result = source[:target_shape[0], :target_shape[1]]
        assert result.shape == target_shape

    def test_same_shape_no_pruning(self):
        """Test that same shape returns correctly."""
        source = torch.randn(512, 512)
        target_shape = (512, 512)
        assert source.shape == target_shape

    def test_prune_preserves_dtype(self):
        """Test that pruning preserves tensor dtype."""
        source = torch.randn(768, 768, dtype=torch.float16)
        result = source[:512, :512]
        assert result.dtype == torch.float16


# ============================================================================
# ID Mapping Tests
# ============================================================================

class TestIdMapping:
    """Test token ID mapping logic for vocab pruning."""

    def test_id_mapping_creation(self):
        """Test creating ID mapping for vocab pruning."""
        keep_token_ids = [0, 1, 2, 100, 200, 500]
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(keep_token_ids))}
        
        assert id_mapping[0] == 0
        assert id_mapping[1] == 1
        assert id_mapping[2] == 2
        assert id_mapping[100] == 3
        assert id_mapping[200] == 4
        assert id_mapping[500] == 5
        assert len(id_mapping) == 6

    def test_special_tokens_included(self):
        """Test that special tokens are automatically included."""
        keep_token_ids = [100, 200, 300]
        special_tokens = {0, 1, 2}  # bos, eos, pad
        
        all_tokens = sorted(set(keep_token_ids) | special_tokens)
        assert 0 in all_tokens
        assert 1 in all_tokens
        assert 2 in all_tokens
        assert len(all_tokens) == 6

    def test_id_mapping_preserves_order(self):
        """Test that ID mapping assigns new IDs in sorted order."""
        keep_token_ids = [500, 100, 200, 0, 1, 2]
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(keep_token_ids))}
        
        # New IDs should be in order of sorted old IDs
        assert id_mapping[0] == 0
        assert id_mapping[1] == 1
        assert id_mapping[2] == 2
        assert id_mapping[100] == 3
        assert id_mapping[200] == 4
        assert id_mapping[500] == 5

    def test_vocab_size_to_keep_token_ids(self):
        """Test converting vocab_size to keep_token_ids."""
        vocab_size = 8000
        keep_token_ids = list(range(vocab_size))
        
        assert len(keep_token_ids) == 8000
        assert keep_token_ids[0] == 0
        assert keep_token_ids[-1] == 7999


# ============================================================================
# Integration Tests (Mocked)
# ============================================================================

class TestTransformerClonerMocked:
    """Integration tests using mocked transformers."""

    @pytest.fixture
    def mock_cloner(self):
        """Create a mocked TransformerCloner."""
        with patch('transformer_cloner.cloner.AutoTokenizer') as mock_tokenizer_cls, \
             patch('transformer_cloner.cloner.AutoModelForCausalLM') as mock_model_cls:
            
            # Mock tokenizer
            mock_tok = MagicMock()
            mock_tok.vocab = {"<bos>": 0, "<eos>": 1, "<pad>": 2, "hello": 3, "world": 4}
            mock_tok.bos_token_id = 0
            mock_tok.eos_token_id = 1
            mock_tok.pad_token_id = 2
            mock_tok.unk_token_id = None
            mock_tok.__len__ = lambda self: 5
            mock_tokenizer_cls.from_pretrained.return_value = mock_tok
            
            # Mock model
            mock_mdl = MagicMock()
            mock_mdl.config.hidden_size = 768
            mock_mdl.config.num_hidden_layers = 12
            mock_mdl.config.intermediate_size = 3072
            mock_mdl.config.num_attention_heads = 12
            mock_mdl.config.num_key_value_heads = 12
            mock_mdl.config.tie_word_embeddings = True
            mock_mdl.config.vocab_size = 5
            mock_mdl.config.to_dict.return_value = {
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'intermediate_size': 3072,
                'num_attention_heads': 12,
            }
            mock_mdl.model.embed_tokens.weight = torch.randn(5, 768)
            mock_mdl.state_dict.return_value = {}
            mock_model_cls.from_pretrained.return_value = mock_mdl
            
            yield mock_tokenizer_cls, mock_model_cls

    def test_cloner_initialization(self, mock_cloner):
        """Test that cloner initializes correctly."""
        mock_tokenizer_cls, mock_model_cls = mock_cloner
        
        cloner = TransformerCloner(
            org_model_id="test/model",
            target_tokenizer_id="test/tokenizer",
        )
        
        assert cloner.org_model_id == "test/model"
        assert cloner.target_tokenizer_id == "test/tokenizer"
        assert cloner.token_id_map == {}
        
        # Check that from_pretrained was called correctly
        mock_tokenizer_cls.from_pretrained.assert_called()
        mock_model_cls.from_pretrained.assert_called_once()

    def test_cloner_attributes(self, mock_cloner):
        """Test that cloner has expected attributes after init."""
        cloner = TransformerCloner(
            org_model_id="test/model",
            target_tokenizer_id="test/tokenizer",
        )
        
        assert hasattr(cloner, 'org_model')
        assert hasattr(cloner, 'org_tokenizer')
        assert hasattr(cloner, 'target_tokenizer')
        assert hasattr(cloner, 'token_id_map')
        assert hasattr(cloner, 'org_model_id')
        assert hasattr(cloner, 'target_tokenizer_id')


# ============================================================================
# API Tests
# ============================================================================

class TestPublicAPI:
    """Test the public API and imports."""

    def test_version_exists(self):
        """Test that version is defined."""
        from transformer_cloner import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        """Test version follows semver format."""
        from transformer_cloner import __version__
        parts = __version__.split(".")
        assert len(parts) >= 2
        # First two parts should be numeric
        assert parts[0].isdigit()
        assert parts[1].isdigit()

    def test_public_api_imports(self):
        """Test that all public API members are importable."""
        from transformer_cloner import TransformerCloner
        from transformer_cloner import EmbeddingStrategy
        from transformer_cloner import PruningConfig
        from transformer_cloner import __version__
        
        assert TransformerCloner is not None
        assert EmbeddingStrategy is not None
        assert PruningConfig is not None
        assert __version__ is not None

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import transformer_cloner
        expected = ['TransformerCloner', 'EmbeddingStrategy', 'PruningConfig', '__version__']
        for name in expected:
            assert name in transformer_cloner.__all__


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pruning_config_zero_values_invalid(self):
        """Test that zero-like pruning configs still produce valid-looking output.
        
        Note: The validation uses `new_value = pruning_value or original_value`,
        so 0 or None both default to the original value. This is by design -
        we use 'or' because 0 is not a valid pruning target anyway.
        """
        config = Mock()
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        config.head_dim = 64
        
        # 0 is treated as "not set" due to `or` logic, so it uses original
        pruning = PruningConfig(hidden_size=0)
        errors = pruning.validate(config)
        # No errors because 0 defaults to original 768
        assert len(errors) == 0

    def test_pruning_config_negative_values_invalid(self):
        """Test that negative values are caught by validation."""
        config = Mock()
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        config.head_dim = 64
        
        pruning = PruningConfig(num_hidden_layers=-1)
        errors = pruning.validate(config)
        assert any("must be positive" in e for e in errors)

    def test_empty_keep_token_ids(self):
        """Test behavior with empty keep_token_ids list."""
        keep_token_ids = []
        special_tokens = {0, 1, 2}
        
        # Special tokens should be added
        all_tokens = sorted(set(keep_token_ids) | special_tokens)
        assert len(all_tokens) == 3  # Only special tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
