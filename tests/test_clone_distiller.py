"""Unit tests for CloneDistiller classes."""

import pytest
from unittest.mock import MagicMock, patch

from transformer_cloner import (
    PruningConfig,
    SentenceTransformerCloneDistiller,
    CloneDistillerConfig,
    TransformerCloneDistiller,
    TransformerCloneDistillerConfig,
)


class TestCloneDistillerConfig:
    """Test CloneDistillerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = CloneDistillerConfig(teacher_model="test-model")
        assert config.teacher_model == "test-model"
        assert config.target_tokenizer is None
        assert config.pruning_config is None
        assert config.train_epochs == 1
        assert config.loss_type == "mse"

    def test_config_with_pruning(self):
        """Test configuration with pruning config."""
        pruning = PruningConfig(num_hidden_layers=6)
        config = CloneDistillerConfig(
            teacher_model="test-model",
            pruning_config=pruning,
        )
        assert config.pruning_config is not None
        assert config.pruning_config.num_hidden_layers == 6


class TestTransformerCloneDistillerConfig:
    """Test TransformerCloneDistillerConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = TransformerCloneDistillerConfig(teacher_model="test-lm")
        assert config.teacher_model == "test-lm"
        assert config.distillation_type == "logit"
        assert config.temperature == 2.0


class TestSentenceTransformerCloneDistiller:
    """Test SentenceTransformerCloneDistiller class."""

    def test_init(self):
        """Test initialization."""
        distiller = SentenceTransformerCloneDistiller(
            teacher_model="sentence-transformers/all-MiniLM-L6-v2",
            output_dir="./test_output",
        )
        assert distiller.teacher_model_path == "sentence-transformers/all-MiniLM-L6-v2"
        assert distiller.config.output_dir == "./test_output"

    def test_init_with_pruning(self):
        """Test initialization with pruning config."""
        pruning = PruningConfig(num_hidden_layers=3)
        distiller = SentenceTransformerCloneDistiller(
            teacher_model="sentence-transformers/all-MiniLM-L6-v2",
            pruning_config=pruning,
        )
        assert distiller.config.pruning_config is not None
        assert distiller.config.pruning_config.num_hidden_layers == 3


class TestTransformerCloneDistiller:
    """Test TransformerCloneDistiller class."""

    def test_init(self):
        """Test initialization."""
        distiller = TransformerCloneDistiller(
            teacher_model="gpt2",
            output_dir="./test_output",
        )
        assert distiller.teacher_model_id == "gpt2"
        assert distiller.config.output_dir == "./test_output"

    def test_init_with_config(self):
        """Test initialization with full config."""
        config = TransformerCloneDistillerConfig(
            teacher_model="gpt2",
            temperature=3.0,
            alpha=0.7,
        )
        distiller = TransformerCloneDistiller(
            teacher_model="gpt2",
            config=config,
        )
        assert distiller.config.temperature == 3.0
        assert distiller.config.alpha == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
