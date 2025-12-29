"""
Transformer Cloner - Clone and prune transformer models with new tokenizers.

A library for cloning transformer models with new tokenizers, including
support for vocabulary mapping, embedding strategies, and model pruning.
"""

from transformer_cloner.embedding_strategy import EmbeddingStrategy
from transformer_cloner.pruning_config import PruningConfig
from transformer_cloner.cloner import TransformerCloner
from transformer_cloner.vocab_pruned_tokenizer import VocabPrunedTokenizer
from transformer_cloner.sentence_transformer_cloner import SentenceTransformerCloner

__version__ = "0.2.6"
__all__ = [
    "TransformerCloner",
    "SentenceTransformerCloner",
    "EmbeddingStrategy",
    "PruningConfig",
    "VocabPrunedTokenizer",
    "__version__",
]

