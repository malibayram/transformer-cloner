# 🔄 Transformer Cloner

[![PyPI version](https://badge.fury.io/py/transformer-cloner.svg)](https://pypi.org/project/transformer-cloner/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Clone and prune transformer models with new tokenizers.** Create smaller, more efficient models by mapping vocabularies, reducing dimensions, and pruning layers.

## Use Cases

- 🌍 **Language Adaptation**: Use a custom tokenizer optimized for your language
- 📉 **Model Compression**: Create smaller models for edge deployment
- 🎓 **Knowledge Distillation**: Generate student models from teacher models
- 🔬 **Research**: Experiment with different model architectures
- 🔤 **SentenceTransformer Cloning**: Clone embedding models with new tokenizers and pruning

---

## 📦 Installation

```bash
pip install transformer-cloner
```

**Requirements:**

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+

---

## 📖 Complete API Reference

### TransformerCloner

The main class for cloning and pruning transformer models.

```python
from transformer_cloner import TransformerCloner

cloner = TransformerCloner(
    org_model_id: str,         # HuggingFace model ID or local path to original model
    target_tokenizer_id: str,  # HuggingFace tokenizer ID or local path to target tokenizer
    token: str = None,         # Optional HuggingFace API token for gated models
)
```

**Attributes after initialization:**

- `cloner.org_model` - The loaded original model
- `cloner.org_tokenizer` - The original model's tokenizer
- `cloner.target_tokenizer` - The target tokenizer
- `cloner.token` - The HuggingFace API token (if provided)

---

### Method: `build_token_id_map()`

Build a mapping from target tokenizer IDs to original tokenizer IDs.

```python
token_map = cloner.build_token_id_map(
    batch_size: int = 5000,   # Number of tokens to process per batch (higher = faster but more memory)
    verbose: bool = True,     # Whether to print progress
) -> dict[int, list[int]]     # Returns {target_token_id: [source_token_ids]}
```

**Example:**

```python
cloner = TransformerCloner(
    org_model_id="google/gemma-3-270m-it",
    target_tokenizer_id="alibayram/turkish-tokenizer",
)

# Build the token mapping
token_map = cloner.build_token_id_map(batch_size=10000, verbose=True)
# Output: Building token ID map for 65536 tokens...
#         Processed 10000/65536 tokens
#         ...
#         Token ID map built with 65536 entries

print(token_map[100])  # [234, 567] - target token 100 maps to source tokens 234, 567
```

---

### Method: `clone()`

Clone the model with a new tokenizer, mapping embeddings from the original model.

```python
model = cloner.clone(
    strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN,  # How to combine multiple source embeddings
    verbose: bool = True,                                   # Whether to print progress
) -> AutoModelForCausalLM                                   # Returns the cloned model
```

**Example:**

```python
from transformer_cloner import TransformerCloner, EmbeddingStrategy

cloner = TransformerCloner(
    org_model_id="google/gemma-3-270m-it",
    target_tokenizer_id="alibayram/turkish-tokenizer",
)

# Clone with mean embedding strategy
model = cloner.clone(strategy=EmbeddingStrategy.MEAN, verbose=True)
# Output: Building token ID map for 65536 tokens...
#         Cloning model with strategy: mean
#         Model vocab size: 65536, Tokenizer vocab size: 65536
#         Copying weights from original model...
#         Mapping embeddings...
#         Mapped 1000/65536 embeddings
#         ...
#         Model cloning complete!

# Save the cloned model
model.save_pretrained("./cloned-model")
```

---

### Method: `clone_with_lm_head()`

Clone the model including the language modeling head (for models with untied weights).

```python
model = cloner.clone_with_lm_head(
    strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN,  # How to combine embeddings
    verbose: bool = True,                                   # Whether to print progress
) -> AutoModelForCausalLM                                   # Returns the cloned model
```

**Example:**

```python
# For models where lm_head is NOT tied to embeddings
model = cloner.clone_with_lm_head(strategy=EmbeddingStrategy.WEIGHTED)
# Output: ... (same as clone)
#         Mapping lm_head weights...
#         lm_head mapping complete!

model.save_pretrained("./cloned-with-lm-head")
```

---

### Method: `clone_pruned()`

Clone the model with architecture pruning (smaller hidden size, fewer layers, etc.).

```python
model = cloner.clone_pruned(
    pruning_config: PruningConfig,                          # Configuration for pruned dimensions
    strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN,   # How to combine embeddings
    verbose: bool = True,                                    # Whether to print progress
) -> AutoModelForCausalLM                                    # Returns the pruned model
```

**Example:**

```python
from transformer_cloner import TransformerCloner, PruningConfig, EmbeddingStrategy

cloner = TransformerCloner(
    org_model_id="google/gemma-3-270m-it",
    target_tokenizer_id="alibayram/turkish-tokenizer",
)

# Create a smaller model
pruning_config = PruningConfig(
    hidden_size=320,           # Reduce from 640 to 320
    num_hidden_layers=9,       # Reduce from 18 to 9
    intermediate_size=1024,    # Reduce from 2048 to 1024
    num_attention_heads=2,     # Reduce from 4 to 2
    num_key_value_heads=1,     # Keep at 1
)

model = cloner.clone_pruned(
    pruning_config=pruning_config,
    strategy=EmbeddingStrategy.MEAN,
    verbose=True,
)
# Output: Original: hidden=640, layers=18, intermediate=2048, heads=4, kv_heads=1
#         Pruned:   hidden=320, layers=9, intermediate=1024, heads=2, kv_heads=1
#         Model vocab size: 65536
#         Copying and pruning weights from original model...
#         Mapping embeddings with pruning...
#         Pruned model cloning complete!

model.save_pretrained("./pruned-model")
```

---

### Method: `clone_with_vocab_pruning()`

Clone model with a reduced embedding table (fewer tokens).

```python
model, tokenizer, id_mapping = cloner.clone_with_vocab_pruning(
    keep_token_ids: Optional[list[int]] = None,   # Specific token IDs to keep
    vocab_size: Optional[int] = None,             # Keep first N tokens (ignored if keep_token_ids provided)
    pruning_config: Optional[PruningConfig] = None,  # Optional architecture pruning
    verbose: bool = True,                          # Whether to print progress
) -> tuple[AutoModelForCausalLM, AutoTokenizer, dict[int, int]]
# Returns: (model, original_tokenizer, id_mapping)
# id_mapping: {old_token_id: new_embedding_index}
```

> **Note:** The original tokenizer is returned unchanged because modifying SentencePiece/BPE vocabularies breaks them. Use `id_mapping` to convert token IDs to embedding indices.

**Example 1: Keep first N tokens**

```python
from transformer_cloner import TransformerCloner

cloner = TransformerCloner(
    org_model_id="google/gemma-3-270m-it",
    target_tokenizer_id="google/gemma-3-270m-it",  # Same tokenizer
)

# Keep only first 8000 tokens
model, tokenizer, id_mapping = cloner.clone_with_vocab_pruning(
    vocab_size=8000,
    verbose=True,
)
# Output: Cloning with vocab pruning: 8000 tokens
#         New vocab size: 8000
#         Creating model with vocab_size=8000
#         Copying weights from original model...
#         Mapping embeddings (direct 1:1)...
#         Mapped 8000 embeddings directly
#         Vocab-pruned model cloning complete!

model.save_pretrained("./vocab-pruned-model")

# Use id_mapping to convert token IDs
print(id_mapping)  # {0: 0, 1: 1, 2: 2, ..., 7999: 7999}
```

**Example 2: Keep specific tokens**

```python
# Keep only specific token IDs
important_tokens = [0, 1, 2, 100, 200, 500, 1000, 2000, 5000]

model, tokenizer, id_mapping = cloner.clone_with_vocab_pruning(
    keep_token_ids=important_tokens,
    verbose=True,
)

print(id_mapping)
# {0: 0, 1: 1, 2: 2, 100: 3, 200: 4, 500: 5, 1000: 6, 2000: 7, 5000: 8}
```

**Example 3: Combined vocab + architecture pruning**

```python
from transformer_cloner import TransformerCloner, PruningConfig

cloner = TransformerCloner(
    org_model_id="google/gemma-3-270m-it",
    target_tokenizer_id="google/gemma-3-270m-it",
)

# Combine vocab pruning with architecture pruning
model, tokenizer, id_mapping = cloner.clone_with_vocab_pruning(
    vocab_size=8000,
    pruning_config=PruningConfig(
        hidden_size=320,
        num_hidden_layers=6,
        intermediate_size=1024,
    ),
    verbose=True,
)

# Result: Tiny model with 8000 tokens and smaller architecture
model.save_pretrained("./tiny-model")
```

---

### Method: `get_token_info()`

Get information about how a specific token is mapped.

```python
info = cloner.get_token_info(
    token: str,  # The token string to look up
) -> dict      # Returns token mapping information
```

**Example:**

```python
cloner.build_token_id_map()

info = cloner.get_token_info("hello")
print(info)
# {
#     'token': 'hello',
#     'target_id': 1234,
#     'source_ids': [567, 890],
#     'source_tokens': ['hel', 'lo']
# }

# Token not found
info = cloner.get_token_info("xyz123")
# {'error': "Token 'xyz123' not found in target tokenizer"}
```

---

### Method: `print_vocab_samples()`

Print sample vocabulary entries from both tokenizers.

```python
cloner.print_vocab_samples(
    n: int = 10,  # Number of samples to print
) -> None
```

**Example:**

```python
cloner.print_vocab_samples(n=5)
# Output:
# Original tokenizer samples:
#   0: '<bos>'   0
#   1: '<eos>'   1
#   2: '<pad>'   2
#   3: '▁'       3
#   4: '▁the'    4
# Total: 262144 tokens
#
# Target tokenizer samples:
#   0: '<bos>'   0
#   1: '<eos>'   1
#   2: '<pad>'   2
#   3: '▁'       3
#   4: '▁ve'     4
# Total: 65536 tokens
```

---

## 🎯 EmbeddingStrategy

When a target token maps to multiple source tokens, choose how to combine their embeddings:

```python
from transformer_cloner import EmbeddingStrategy
```

| Strategy   | Value        | Description                                   | Best For                             |
| ---------- | ------------ | --------------------------------------------- | ------------------------------------ |
| `MEAN`     | `"mean"`     | Average of all embeddings                     | **Default**, balanced representation |
| `SUM`      | `"sum"`      | Sum of all embeddings                         | Preserving total magnitude           |
| `FIRST`    | `"first"`    | First token's embedding only                  | Prefix-focused tokens                |
| `LAST`     | `"last"`     | Last token's embedding only                   | Suffix-focused tokens                |
| `WEIGHTED` | `"weighted"` | Weighted average (first tokens weighted more) | Morphological priority               |
| `MAX`      | `"max"`      | Element-wise maximum                          | Preserving dominant features         |
| `MIN`      | `"min"`      | Element-wise minimum                          | Preserving minimal features          |

**Example:**

```python
# Use different strategies
model = cloner.clone(strategy=EmbeddingStrategy.MEAN)     # Average
model = cloner.clone(strategy=EmbeddingStrategy.WEIGHTED) # First tokens matter more
model = cloner.clone(strategy=EmbeddingStrategy.FIRST)    # Only first token
```

---

## ⚙️ PruningConfig

Configuration dataclass for model architecture pruning.

```python
from transformer_cloner import PruningConfig

config = PruningConfig(
    hidden_size: Optional[int] = None,           # Embedding dimension
    num_hidden_layers: Optional[int] = None,     # Number of transformer layers
    intermediate_size: Optional[int] = None,     # FFN intermediate dimension
    num_attention_heads: Optional[int] = None,   # Number of attention heads
    num_key_value_heads: Optional[int] = None,   # Number of KV heads (for GQA)
    head_dim: Optional[int] = None,              # Dimension per attention head
)
```

Set any value to `None` to keep the original model's value.

**Example configurations:**

```python
# Half the layers only
config = PruningConfig(num_hidden_layers=9)  # 18 -> 9

# Half all dimensions
config = PruningConfig(
    hidden_size=320,          # 640 -> 320
    num_hidden_layers=9,      # 18 -> 9
    intermediate_size=1024,   # 2048 -> 1024
)

# Tiny model
config = PruningConfig(
    hidden_size=128,
    num_hidden_layers=3,
    intermediate_size=512,
    num_attention_heads=2,
    num_key_value_heads=1,
)
```

### Validation

Use `validate()` to check if your config is valid before cloning:

```python
errors = config.validate(cloner.org_model.config)
if errors:
    print("Validation errors:", errors)
else:
    print("Config is valid!")
```

**Validation checks:**

- ✅ Dimensions don't exceed original model
- ✅ All values are positive
- ✅ `num_attention_heads` is divisible by `num_key_value_heads`
- ✅ `hidden_size` is compatible with attention configuration

---

## 🔄 SentenceTransformerCloner

Clone SentenceTransformer models with new tokenizers and/or architecture pruning.

```python
from transformer_cloner import SentenceTransformerCloner, PruningConfig

cloner = SentenceTransformerCloner(
    model_path: str,                          # Path to SentenceTransformer model
    target_tokenizer_id: Optional[str] = None, # New tokenizer (None = keep original)
    pruning_config: Optional[PruningConfig] = None,  # Optional architecture pruning
    token: str = None,                        # Optional HuggingFace API token
)
```

### Clone with New Tokenizer

```python
from transformer_cloner import SentenceTransformerCloner

cloner = SentenceTransformerCloner(
    model_path="./embeddinggemma",
    target_tokenizer_id="alibayram/turkish-tokenizer"
)
cloner.clone(verbose=True)
cloner.save("./cloned_sentence_transformer")
```

### Clone with Architecture Pruning

```python
from transformer_cloner import SentenceTransformerCloner, PruningConfig

config = PruningConfig(
    hidden_size=512,
    num_hidden_layers=12,
)

cloner = SentenceTransformerCloner(
    model_path="./embeddinggemma",
    target_tokenizer_id="alibayram/turkish-tokenizer",
    pruning_config=config
)
cloner.clone(verbose=True)
cloner.save("./cloned_pruned_model")
```

**What gets handled:**

- **Transformer**: Cloned using `TransformerCloner` with embedding mapping
- **Pooling**: Config's `word_embedding_dimension` updated to match new `hidden_size`
- **Dense layers**: Weights sliced when dimensions match `hidden_size`
- **Normalize**: Copied as-is (no weights)

## 📊 Gemma-3-270m Architecture Reference

For `google/gemma-3-270m-it`:

| Parameter             | Original Value |
| --------------------- | -------------- |
| `hidden_size`         | 640            |
| `num_hidden_layers`   | 18             |
| `intermediate_size`   | 2048           |
| `num_attention_heads` | 4              |
| `num_key_value_heads` | 1              |
| `head_dim`            | 256            |
| `vocab_size`          | 262144         |

**Example pruned configs:**

```python
# ~50% size (9 layers, same hidden)
PruningConfig(num_hidden_layers=9)

# ~25% size (half dimensions)
PruningConfig(
    hidden_size=320,
    num_hidden_layers=9,
    intermediate_size=1024,
)

# Tiny (~12% size)
PruningConfig(
    hidden_size=160,
    num_hidden_layers=6,
    intermediate_size=512,
    num_attention_heads=2,
)
```

---

## 🔧 Complete Workflow Example

```python
from transformer_cloner import TransformerCloner, PruningConfig, EmbeddingStrategy

# 1. Initialize
cloner = TransformerCloner(
    org_model_id="google/gemma-3-270m-it",
    target_tokenizer_id="my-org/turkish-gemma-tokenizer",
)

# 2. Explore the vocabularies
cloner.print_vocab_samples(n=5)

# 3. Build token mapping
token_map = cloner.build_token_id_map()

# 4. Check a specific token
info = cloner.get_token_info("merhaba")
print(f"'merhaba' maps to source tokens: {info['source_tokens']}")

# 5. Create pruned model
pruning_config = PruningConfig(
    hidden_size=320,
    num_hidden_layers=9,
    intermediate_size=1024,
)

# Validate first
errors = pruning_config.validate(cloner.org_model.config)
if errors:
    raise ValueError(f"Invalid config: {errors}")

# 6. Clone with pruning
model = cloner.clone_pruned(
    pruning_config=pruning_config,
    strategy=EmbeddingStrategy.MEAN,
)

# 7. Save
model.save_pretrained("./turkish-gemma-small")
cloner.target_tokenizer.save_pretrained("./turkish-gemma-small")

print("Done! Model saved to ./turkish-gemma-small")
```

---

## 🧪 Test Scripts

The repository includes test scripts to validate the package with various models:

| Script                                                                                                                                        | Description                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| [`test_multi_model_cloning.py`](https://github.com/malibayram/transformer-cloner/blob/main/scripts/test_multi_model_cloning.py)               | Test vocab pruning across multiple model families |
| [`test_generation.py`](https://github.com/malibayram/transformer-cloner/blob/main/scripts/test_generation.py)                                 | Test text generation with cloned models           |
| [`test_cross_tokenizer_cloning.py`](https://github.com/malibayram/transformer-cloner/blob/main/scripts/test_cross_tokenizer_cloning.py)       | Test cloning with a custom tokenizer              |
| [`test_sentence_transformer_cloner.py`](https://github.com/malibayram/transformer-cloner/blob/main/tests/test_sentence_transformer_cloner.py) | Test SentenceTransformer cloning with pruning     |

**Run locally:**

```bash
# Clone with vocab pruning and save locally
python scripts/test_multi_model_cloning.py

# Test generation on saved models
python scripts/test_generation.py

# Clone with a custom Turkish tokenizer
python scripts/test_cross_tokenizer_cloning.py

# Test SentenceTransformer cloning
python tests/test_sentence_transformer_cloner.py
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Built on top of [🤗 Transformers](https://github.com/huggingface/transformers)
- Inspired by vocabulary adaptation research in multilingual NLP
