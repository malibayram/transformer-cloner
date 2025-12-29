"""Combined clone and distillation training for SentenceTransformer models.

This module provides SentenceTransformerCloneDistiller, which combines:
1. Model cloning with new tokenizer and/or architecture pruning
2. Distillation training using the original model as teacher
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
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from transformer_cloner.pruning_config import PruningConfig
from transformer_cloner.sentence_transformer_cloner import SentenceTransformerCloner

logger = logging.getLogger(__name__)


@dataclass
class CloneDistillerConfig:
    """Configuration for clone + distillation training."""

    # Teacher model
    teacher_model: str = ""

    # Cloning options
    target_tokenizer: str | None = None
    pruning_config: PruningConfig | None = None

    # Training options
    train_epochs: int = 1
    train_batch_size: int = 64
    eval_batch_size: int = 64
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    loss_type: str = "mse"  # mse, cosine, combined

    # Data options
    train_data: str | None = None
    eval_data: str | None = None
    text_column: str = "sentence"
    max_samples: int | None = None
    max_seq_length: int = 512

    # Evaluation
    eval_steps: int = 500
    logging_steps: int = 100

    # Output
    output_dir: str = "./clone_distilled"
    save_steps: int = 500

    # Hardware
    device: str = "auto"


class SentenceTransformerCloneDistiller:
    """
    Clone a SentenceTransformer model with pruning/tokenizer change and train via distillation.

    Combines the functionality of SentenceTransformerCloner and distillation training
    into a single workflow.

    Example:
        >>> distiller = SentenceTransformerCloneDistiller(
        ...     teacher_model="mixedbread-ai/mxbai-embed-large-v1",
        ...     pruning_config=PruningConfig(hidden_size=512, num_hidden_layers=6),
        ... )
        >>> distiller.load_data("sentence-transformers/all-nli")
        >>> distiller.run()  # Clone + Train
        >>> distiller.save("./final_model")
    """

    def __init__(
        self,
        teacher_model: str | SentenceTransformer,
        target_tokenizer: str | None = None,
        pruning_config: PruningConfig | None = None,
        config: CloneDistillerConfig | None = None,
        output_dir: str = "./clone_distilled",
        device: str = "auto",
    ):
        """
        Initialize the SentenceTransformerCloneDistiller.

        Args:
            teacher_model: Path or HuggingFace ID for the teacher model.
            target_tokenizer: Optional path or ID for a new tokenizer.
            pruning_config: Optional pruning configuration for model compression.
            config: Optional full configuration object.
            output_dir: Directory to save output models.
            device: Device for training ("auto", "cuda", "cpu", "mps").
        """
        # Set up configuration
        if config is not None:
            self.config = config
        else:
            self.config = CloneDistillerConfig(
                teacher_model=teacher_model if isinstance(teacher_model, str) else "",
                target_tokenizer=target_tokenizer,
                pruning_config=pruning_config,
                output_dir=output_dir,
                device=device,
            )

        # Override with explicit parameters
        if target_tokenizer is not None:
            self.config.target_tokenizer = target_tokenizer
        if pruning_config is not None:
            self.config.pruning_config = pruning_config
        if output_dir != "./clone_distilled":
            self.config.output_dir = output_dir

        # Determine device
        self.device = self._get_device()

        # Load or store teacher model
        if isinstance(teacher_model, str):
            self.teacher_model_path = teacher_model
            self.teacher_model: SentenceTransformer | None = None
        else:
            self.teacher_model_path = ""
            self.teacher_model = teacher_model

        # Student model (created during clone)
        self.student_model: SentenceTransformer | None = None
        self._student_temp_path: str | None = None

        # Training data
        self.train_dataset: Dataset | None = None
        self.eval_dataset: Dataset | None = None

        # Training state
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0

        logger.info(f"Initialized SentenceTransformerCloneDistiller")
        logger.info(f"  Teacher: {self.teacher_model_path or 'provided model'}")
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

    def _load_teacher(self) -> SentenceTransformer:
        """Load the teacher model if not already loaded."""
        if self.teacher_model is None:
            logger.info(f"Loading teacher model: {self.teacher_model_path}")
            self.teacher_model = SentenceTransformer(self.teacher_model_path)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        return self.teacher_model

    def clone(self, verbose: bool = True) -> SentenceTransformer:
        """
        Clone the teacher model with the configured tokenizer/pruning.

        Args:
            verbose: Whether to print progress.

        Returns:
            The cloned student model.
        """
        teacher = self._load_teacher()

        # Determine if we need to do any cloning
        needs_cloning = (
            self.config.target_tokenizer is not None or
            self.config.pruning_config is not None
        )

        if not needs_cloning:
            logger.info("No tokenizer change or pruning - using teacher as student base")
            # Create a copy of the teacher
            self._student_temp_path = tempfile.mkdtemp()
            teacher.save(self._student_temp_path)
            self.student_model = SentenceTransformer(self._student_temp_path)
        else:
            logger.info("Cloning teacher model...")

            # Get teacher path for cloner
            teacher_path = self.teacher_model_path
            if not teacher_path:
                # Save teacher to temp location
                teacher_path = tempfile.mkdtemp()
                teacher.save(teacher_path)

            # Create cloner
            cloner = SentenceTransformerCloner(
                model_path=teacher_path,
                target_tokenizer_id=self.config.target_tokenizer,
                pruning_config=self.config.pruning_config,
            )

            # Clone
            cloner.clone(verbose=verbose)

            # Save to temp directory
            self._student_temp_path = tempfile.mkdtemp()
            cloner.save(self._student_temp_path, verbose=verbose)

            # Load as SentenceTransformer
            self.student_model = SentenceTransformer(self._student_temp_path)

        self.student_model.to(self.device)

        # Log model info
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        compression = 1 - (student_params / teacher_params)

        logger.info(f"Clone complete:")
        logger.info(f"  Teacher params: {teacher_params:,}")
        logger.info(f"  Student params: {student_params:,}")
        logger.info(f"  Compression: {compression:.1%}")

        return self.student_model

    def load_data(
        self,
        train_data: str | Dataset | None = None,
        eval_data: str | Dataset | None = None,
        text_column: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        """
        Load training and evaluation data.

        Args:
            train_data: Training dataset path/name or Dataset object.
            eval_data: Evaluation dataset path/name or Dataset object.
            text_column: Name of the text column in the dataset.
            max_samples: Maximum samples to use.
        """
        if text_column is not None:
            self.config.text_column = text_column
        if max_samples is not None:
            self.config.max_samples = max_samples

        # Load training data
        if train_data is not None:
            self.train_dataset = self._load_dataset(train_data, "train")
            logger.info(f"Loaded training data: {len(self.train_dataset)} samples")

        # Load eval data
        if eval_data is not None:
            self.eval_dataset = self._load_dataset(eval_data, "validation")
            logger.info(f"Loaded eval data: {len(self.eval_dataset)} samples")

    def _load_dataset(self, data: str | Dataset, split: str = "train") -> Dataset:
        """Load a dataset."""
        if isinstance(data, Dataset):
            return data

        try:
            dataset = load_dataset(data, split=split)
        except Exception:
            # Try loading as DatasetDict
            loaded = load_dataset(data)
            if split in loaded:
                dataset = loaded[split]
            else:
                dataset = list(loaded.values())[0]

        # Limit samples
        if self.config.max_samples and len(dataset) > self.config.max_samples:
            dataset = dataset.select(range(self.config.max_samples))

        return dataset

    def _get_loss_function(self) -> Callable:
        """Get the loss function based on config."""
        if self.config.loss_type == "mse":
            return F.mse_loss
        elif self.config.loss_type == "cosine":
            def cosine_loss(student, teacher):
                return (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()
            return cosine_loss
        elif self.config.loss_type == "combined":
            def combined_loss(student, teacher):
                mse = F.mse_loss(student, teacher)
                cosine = (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()
                return 0.5 * mse + 0.5 * cosine
            return combined_loss
        else:
            return F.mse_loss

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
        logger.info(f"  Loss type: {self.config.loss_type}")

        # Ensure teacher is loaded
        teacher = self._load_teacher()

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

        # Get loss function
        loss_fn = self._get_loss_function()

        # Training loop
        self.student_model.train()
        self.global_step = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_steps = 0

            # Create batches
            sentences = self.train_dataset[self.config.text_column]
            indices = list(range(0, len(sentences), batch_size))

            if verbose:
                progress = tqdm(indices, desc=f"Epoch {epoch + 1}/{epochs}")
            else:
                progress = indices

            for i in progress:
                batch_sentences = sentences[i:i + batch_size]

                # Get teacher embeddings (no grad)
                with torch.no_grad():
                    teacher_embeddings = teacher.encode(
                        batch_sentences,
                        convert_to_tensor=True,
                        show_progress_bar=False,
                    )

                # Get student embeddings
                student_embeddings = self.student_model.encode(
                    batch_sentences,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )

                # Compute loss
                loss = loss_fn(student_embeddings, teacher_embeddings)

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
        Save the student model.

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

        self.student_model.save(output_path)

        if verbose:
            logger.info(f"Model saved to: {output_path}")

        return output_path

    def evaluate(
        self,
        sentences: list[str] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        """
        Evaluate the student model against the teacher.

        Args:
            sentences: Sentences to evaluate on. Uses eval_dataset if not provided.
            batch_size: Batch size for encoding.

        Returns:
            Dictionary with MSE and cosine similarity scores.
        """
        if self.student_model is None:
            raise ValueError("No student model. Call clone() first.")

        teacher = self._load_teacher()
        batch_size = batch_size or self.config.eval_batch_size

        # Get sentences
        if sentences is None:
            if self.eval_dataset is not None:
                sentences = self.eval_dataset[self.config.text_column][:1000]
            else:
                raise ValueError("No sentences provided and no eval_dataset loaded.")

        # Encode with both models
        with torch.no_grad():
            teacher_embeddings = teacher.encode(
                sentences,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
            )
            student_embeddings = self.student_model.encode(
                sentences,
                convert_to_tensor=True,
                show_progress_bar=True,
                batch_size=batch_size,
            )

        # Compute metrics
        mse = F.mse_loss(student_embeddings, teacher_embeddings).item()
        cosine_sim = F.cosine_similarity(
            student_embeddings, teacher_embeddings, dim=-1
        ).mean().item()

        results = {
            "mse": mse,
            "cosine_similarity": cosine_sim,
        }

        logger.info(f"Evaluation: MSE={mse:.6f}, Cosine={cosine_sim:.4f}")

        return results
