"""Combined clone and distillation training for general transformer models.

This module provides TransformerCloneDistiller, which combines:
1. Model cloning with new tokenizer and/or architecture pruning
2. Distillation training using the original model as teacher

For causal LLM models (like Qwen, Llama, Gemma, etc.)
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)

from transformer_cloner.cloner import TransformerCloner
from transformer_cloner.embedding_strategy import EmbeddingStrategy
from transformer_cloner.pruning_config import PruningConfig

logger = logging.getLogger(__name__)


@dataclass
class TransformerCloneDistillerConfig:
    """Configuration for transformer clone + distillation training."""

    # Teacher model
    teacher_model: str = ""

    # Cloning options
    target_tokenizer: str | None = None
    pruning_config: PruningConfig | None = None
    embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.MEAN

    # Training options
    train_epochs: int = 1
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Distillation options
    distillation_type: str = "logit"  # logit, embedding, combined
    temperature: float = 2.0
    alpha: float = 0.5  # Weight for distillation vs task loss

    # Data options
    train_data: str | None = None
    eval_data: str | None = None
    text_column: str = "text"
    max_samples: int | None = None
    max_seq_length: int = 512

    # Evaluation
    eval_steps: int = 500
    logging_steps: int = 100

    # Output
    output_dir: str = "./clone_distilled_lm"
    save_steps: int = 500

    # Hardware
    device: str = "auto"
    token: str | None = None  # HuggingFace token


class TransformerCloneDistiller:
    """
    Clone a transformer model with pruning/tokenizer change and train via distillation.

    For causal language models (LLMs) like Llama, Qwen, Gemma, etc.

    Example:
        >>> distiller = TransformerCloneDistiller(
        ...     teacher_model="Qwen/Qwen2-1.5B",
        ...     pruning_config=PruningConfig(num_hidden_layers=12, hidden_size=1024),
        ... )
        >>> distiller.load_data("wikitext", "wikitext-2-raw-v1")
        >>> distiller.run()
        >>> distiller.save("./distilled_qwen")
    """

    def __init__(
        self,
        teacher_model: str,
        target_tokenizer: str | None = None,
        pruning_config: PruningConfig | None = None,
        config: TransformerCloneDistillerConfig | None = None,
        output_dir: str = "./clone_distilled_lm",
        device: str = "auto",
        token: str | None = None,
    ):
        """
        Initialize the TransformerCloneDistiller.

        Args:
            teacher_model: Path or HuggingFace ID for the teacher model.
            target_tokenizer: Optional path or ID for a new tokenizer.
            pruning_config: Optional pruning configuration for model compression.
            config: Optional full configuration object.
            output_dir: Directory to save output models.
            device: Device for training ("auto", "cuda", "cpu", "mps").
            token: HuggingFace token for gated models.
        """
        # Set up configuration
        if config is not None:
            self.config = config
        else:
            self.config = TransformerCloneDistillerConfig(
                teacher_model=teacher_model,
                target_tokenizer=target_tokenizer,
                pruning_config=pruning_config,
                output_dir=output_dir,
                device=device,
                token=token,
            )

        # Override with explicit parameters
        if target_tokenizer is not None:
            self.config.target_tokenizer = target_tokenizer
        if pruning_config is not None:
            self.config.pruning_config = pruning_config
        if output_dir != "./clone_distilled_lm":
            self.config.output_dir = output_dir
        if token is not None:
            self.config.token = token

        # Determine device
        self.device = self._get_device()

        # Models and tokenizers
        self.teacher_model_id = self.config.teacher_model
        self.teacher_model: PreTrainedModel | None = None
        self.teacher_tokenizer: PreTrainedTokenizer | None = None
        self.student_model: PreTrainedModel | None = None
        self.student_tokenizer: PreTrainedTokenizer | None = None

        # Training data
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0

        logger.info(f"Initialized TransformerCloneDistiller")
        logger.info(f"  Teacher: {self.teacher_model_id}")
        logger.info(f"  Target tokenizer: {self.config.target_tokenizer or 'same as teacher'}")
        logger.info(f"  Pruning config: {self.config.pruning_config}")

    def _get_device(self) -> torch.device:
        """Determine the device to use."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.config.device)

    def _load_teacher(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the teacher model and tokenizer."""
        if self.teacher_model is None:
            logger.info(f"Loading teacher model: {self.teacher_model_id}")
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_id,
                token=self.config.token,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(
                self.teacher_model_id,
                token=self.config.token,
            )

        self.teacher_model.to(self.device)
        self.teacher_model.eval()

        return self.teacher_model, self.teacher_tokenizer

    def clone(self, verbose: bool = True) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Clone the teacher model with the configured tokenizer/pruning.

        Args:
            verbose: Whether to print progress.

        Returns:
            Tuple of (cloned_model, tokenizer).
        """
        teacher, teacher_tokenizer = self._load_teacher()

        # Determine target tokenizer
        target_tokenizer_id = self.config.target_tokenizer or self.teacher_model_id

        # Create cloner
        cloner = TransformerCloner(
            org_model_id=self.teacher_model_id,
            target_tokenizer_id=target_tokenizer_id,
            token=self.config.token,
        )

        # Clone based on configuration
        if self.config.pruning_config is not None:
            logger.info("Cloning with architecture pruning...")
            self.student_model, self.student_tokenizer = cloner.clone_pruned(
                pruning_config=self.config.pruning_config,
                strategy=self.config.embedding_strategy,
                verbose=verbose,
            )
        elif self.config.target_tokenizer is not None:
            logger.info("Cloning with new tokenizer...")
            self.student_model = cloner.clone(
                strategy=self.config.embedding_strategy,
                verbose=verbose,
            )
            self.student_tokenizer = cloner.target_tokenizer
        else:
            logger.info("No cloning needed - copying teacher model...")
            # Just use the teacher model as starting point
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.teacher_model_id,
                token=self.config.token,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            )
            self.student_tokenizer = AutoTokenizer.from_pretrained(
                self.teacher_model_id,
                token=self.config.token,
            )

        self.student_model.to(self.device)

        # Log model info
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        compression = 1 - (student_params / teacher_params)

        logger.info(f"Clone complete:")
        logger.info(f"  Teacher params: {teacher_params:,}")
        logger.info(f"  Student params: {student_params:,}")
        logger.info(f"  Compression: {compression:.1%}")

        return self.student_model, self.student_tokenizer

    def load_data(
        self,
        train_data: str | Dataset | None = None,
        config_name: str | None = None,
        text_column: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """
        Load training data.

        Args:
            train_data: Training dataset path/name or Dataset object.
            config_name: Dataset configuration name (e.g., "wikitext-2-raw-v1").
            text_column: Name of the text column in the dataset.
            max_samples: Maximum samples to use.
        """
        if text_column is not None:
            self.config.text_column = text_column
        if max_samples is not None:
            self.config.max_samples = max_samples

        if train_data is not None:
            if isinstance(train_data, Dataset):
                self.train_dataset = train_data
            else:
                logger.info(f"Loading dataset: {train_data}")
                if config_name:
                    self.train_dataset = load_dataset(train_data, config_name, split="train")
                else:
                    self.train_dataset = load_dataset(train_data, split="train")

            # Limit samples
            if self.config.max_samples and len(self.train_dataset) > self.config.max_samples:
                self.train_dataset = self.train_dataset.select(range(self.config.max_samples))

            logger.info(f"Loaded training data: {len(self.train_dataset)} samples")

    def _get_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute distillation loss."""
        temperature = self.config.temperature

        # Soft targets loss (KL divergence)
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

        if labels is not None and self.config.alpha < 1.0:
            # Hard targets loss (cross-entropy)
            hard_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return self.config.alpha * soft_loss + (1 - self.config.alpha) * hard_loss

        return soft_loss

    def train(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """
        Train the student model via distillation.

        Args:
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            verbose: Whether to show progress.

        Returns:
            Dictionary of training metrics.
        """
        if self.student_model is None:
            raise ValueError("Student model not created. Call clone() first.")
        if self.train_dataset is None:
            raise ValueError("Training data not loaded. Call load_data() first.")

        # Override config if specified
        epochs = epochs or self.config.train_epochs
        batch_size = batch_size or self.config.train_batch_size
        learning_rate = learning_rate or self.config.learning_rate

        logger.info(f"Starting distillation training:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Temperature: {self.config.temperature}")

        # Ensure teacher and student are loaded
        teacher, teacher_tokenizer = self._load_teacher()

        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Calculate total steps
        num_batches = len(self.train_dataset) // batch_size
        total_steps = num_batches * epochs

        # Set up scheduler
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Training loop
        self.student_model.train()
        self.global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_steps = 0

            # Get text data
            texts = self.train_dataset[self.config.text_column]
            indices = list(range(0, len(texts), batch_size))

            if verbose:
                progress = tqdm(indices, desc=f"Epoch {epoch + 1}/{epochs}")
            else:
                progress = indices

            for i in progress:
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.student_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get teacher logits (no grad)
                with torch.no_grad():
                    teacher_outputs = teacher(**inputs)
                    teacher_logits = teacher_outputs.logits

                # Get student logits
                student_outputs = self.student_model(**inputs)
                student_logits = student_outputs.logits

                # Handle vocab size mismatch
                min_vocab = min(student_logits.size(-1), teacher_logits.size(-1))
                student_logits = student_logits[..., :min_vocab]
                teacher_logits = teacher_logits[..., :min_vocab]

                # Compute loss
                loss = self._get_distillation_loss(student_logits, teacher_logits)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student_model.parameters(),
                        self.config.max_grad_norm,
                    )

                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                num_steps += 1
                self.global_step += 1

                if verbose:
                    progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / max(num_steps, 1)
            logger.info(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        return {"train_loss": avg_loss}

    def run(
        self,
        epochs: int | None = None,
        verbose: bool = True,
    ) -> dict[str, float]:
        """
        Run the full clone + train pipeline.

        Args:
            epochs: Number of training epochs (overrides config).
            verbose: Whether to show progress.

        Returns:
            Dictionary of training metrics.
        """
        logger.info("Running Clone + Distillation pipeline...")

        # Step 1: Clone
        self.clone(verbose=verbose)

        # Step 2: Train (if data is loaded)
        if self.train_dataset is not None:
            results = self.train(epochs=epochs, verbose=verbose)
        else:
            logger.warning("No training data loaded. Call load_data() to enable training.")
            results = {}

        # Step 3: Save
        self.save(verbose=verbose)

        return results

    def save(
        self,
        output_path: str | None = None,
        verbose: bool = True,
    ) -> str:
        """
        Save the student model and tokenizer.

        Args:
            output_path: Path to save the model. Defaults to config.output_dir.
            verbose: Whether to print progress.

        Returns:
            Path where the model was saved.
        """
        if self.student_model is None:
            raise ValueError("No student model to save. Call clone() first.")

        output_path = output_path or self.config.output_dir
        Path(output_path).mkdir(parents=True, exist_ok=True)

        self.student_model.save_pretrained(output_path)
        self.student_tokenizer.save_pretrained(output_path)

        if verbose:
            logger.info(f"Model saved to: {output_path}")

        return output_path
