"""Embedding combination strategies for vocabulary mapping."""

from enum import Enum


class EmbeddingStrategy(Enum):
    """
    Strategies for combining multiple embeddings into one.

    When mapping from a new tokenizer to an original tokenizer,
    a single new token may correspond to multiple original tokens.
    This enum defines how to combine those multiple embeddings.

    Attributes:
        MEAN: Average of all source embeddings (default, recommended)
        SUM: Sum of all source embeddings
        FIRST: Use only the first token's embedding
        LAST: Use only the last token's embedding
        WEIGHTED: Weighted average with more weight to first tokens
        MAX: Element-wise maximum across embeddings
        MIN: Element-wise minimum across embeddings
    """

    MEAN = "mean"
    SUM = "sum"
    FIRST = "first"
    LAST = "last"
    WEIGHTED = "weighted"
    MAX = "max"
    MIN = "min"
