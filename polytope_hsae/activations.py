"""
Batched activation capture from language models.

This module handles efficient capture of residual stream activations from
transformer models for the polytope discovery experiments.
"""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ActivationConfig:
    """Configuration for activation capture."""

    model_name: str
    layer_name: str  # e.g., "final_residual_pre_unembed"
    batch_size: int = 32
    max_length: int = 128
    device: str = "cuda:0"
    dtype: torch.dtype = torch.bfloat16
    hook_layer_path: Optional[str] = None  # explicit dotted path for hook layer


class ActivationCapture:
    """Captures activations from transformer models."""

    def __init__(
        self, config: ActivationConfig, negative_corpus: Optional[List[str]] = None
    ):
        """Initialize activation capture.

        Args:
            config: ActivationConfig object.
            negative_corpus: Optional external corpus of prompts to use when
                sampling negatives. When provided, random samples from this
                corpus are mixed into the automatically generated negatives
                to improve contrast quality.
        """
        self.config = config
        self.device = torch.device(config.device)
        self.negative_corpus = negative_corpus

        # Use AutoModelForCausalLM when available to ensure consistent heads;
        # fall back to AutoModel if necessary.
        logger.info(f"Loading model {config.model_name}")
        try:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=config.dtype,
                device_map="auto" if "cuda" in config.device else None,
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                config.model_name,
                torch_dtype=config.dtype,
                device_map="auto" if "cuda" in config.device else None,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

        # Find target layer (or plan B: capture last_hidden_state directly)
        self.target_layer = None
        try:
            if self.config.hook_layer_path:
                self.target_layer = self._get_module_by_path(
                    self.config.hook_layer_path
                )
            else:
                self.target_layer = self._find_target_layer()
            logger.info(f"Target layer: {self.target_layer}")
        except Exception:
            logger.warning(
                "Falling back to capturing last_hidden_state (no explicit hook layer found)."
            )

        # Storage for captured activations
        self.captured_activations = []
        self.hook_handle = None

    def _find_target_layer(self) -> torch.nn.Module:
        """Find the target layer for activation capture.

        This method falls back to simple string heuristics if no explicit
        ``hook_layer_path`` is provided in :class:`ActivationConfig`. Users
        working with bespoke model architectures are encouraged to supply the
        exact module path via ``hook_layer_path`` for reliability.
        """
        layer_name = self.config.layer_name.lower()

        # Common layer patterns
        if "final" in layer_name and "residual" in layer_name:
            # Look for final layer normalization or similar
            for name, module in self.model.named_modules():
                if ("final" in name.lower() or "last" in name.lower()) and (
                    "norm" in name.lower() or "layer_norm" in name.lower()
                ):
                    return module

            # Fallback: use the last transformer layer
            if hasattr(self.model, "layers"):
                return self.model.layers[-1]
            if hasattr(self.model, "h"):  # GPT-style
                return self.model.h[-1]
            if hasattr(self.model, "transformer") and hasattr(
                self.model.transformer, "h"
            ):
                return self.model.transformer.h[-1]

        # For specific layer names, search directly
        for name, module in self.model.named_modules():
            if layer_name in name.lower():
                return module

        raise ValueError(f"Could not find target layer: {self.config.layer_name}")

    def _get_module_by_path(self, path: str) -> torch.nn.Module:
        """Retrieve a module from the model via a dotted path."""
        module: torch.nn.Module = self.model
        for attr in path.split("."):
            module = getattr(module, attr)
        return module

    def _activation_hook(self, module, input, output):
        """Hook function to capture activations."""
        if isinstance(output, tuple):
            activation = output[0]  # Usually the first element is the main output
        else:
            activation = output

        # Store activation (detached to avoid gradients)
        self.captured_activations.append(activation.detach().cpu())

    def capture_batch_activations(self, prompts: List[str]) -> torch.Tensor:
        """
        Capture activations for a batch of prompts.

        Args:
            prompts: List of text prompts

        Returns:
            Tensor of activations [batch_size, seq_len, hidden_dim]
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        # Clear previous captures
        self.captured_activations = []

        # Register hook if we have a target layer; else run once and grab last_hidden_state
        if self.target_layer is not None:
            self.hook_handle = self.target_layer.register_forward_hook(
                self._activation_hook
            )

        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            if self.captured_activations:
                activations = self.captured_activations[0]
                return activations
            # Fallback: use last_hidden_state if hook approach failed
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state.detach().cpu()
            raise RuntimeError("No activations were captured")

        finally:
            # Clean up hook
            if self.hook_handle:
                self.hook_handle.remove()
                self.hook_handle = None

    def capture_last_token_activations(self, prompts: List[str]) -> torch.Tensor:
        """
        Capture activations for the last token of each prompt.

        Args:
            prompts: List of text prompts

        Returns:
            Tensor of last-token activations [batch_size, hidden_dim]
        """
        # Tokenize to get attention masks
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )

        # Get sequence lengths (last non-padding token)
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing

        # Capture full activations
        full_activations = self.capture_batch_activations(
            prompts
        )  # [batch, seq, hidden]

        # Extract last token activations
        batch_size = full_activations.shape[0]
        hidden_dim = full_activations.shape[2]
        last_token_activations = torch.zeros(batch_size, hidden_dim)

        for i in range(batch_size):
            last_pos = seq_lengths[i].item()
            last_token_activations[i] = full_activations[i, last_pos]

        return last_token_activations

    def capture_concept_activations(
        self, concept_prompts: Dict[str, List[str]], save_path: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Capture activations for multiple concepts.

        Args:
            concept_prompts: Dict mapping concept_id -> list of prompts
            save_path: Optional path to save activations

        Returns:
            Dict mapping concept_id -> activation tensor
        """
        logger.info(f"Capturing activations for {len(concept_prompts)} concepts")

        concept_activations = {}

        for concept_id, prompts in tqdm(
            concept_prompts.items(), desc="Capturing activations"
        ):
            # Process in batches
            all_activations = []

            for i in range(0, len(prompts), self.config.batch_size):
                batch_prompts = prompts[i : i + self.config.batch_size]
                batch_activations = self.capture_last_token_activations(batch_prompts)
                all_activations.append(batch_activations)

            # Concatenate all batches
            concept_activations[concept_id] = torch.cat(all_activations, dim=0)

            logger.debug(
                f"Captured {concept_activations[concept_id].shape[0]} activations for {concept_id}"
            )

        # Save if requested
        if save_path:
            self.save_activations(concept_activations, save_path)

        return concept_activations

    def capture_hierarchical_activations(
        self,
        hierarchies,
        negative_sampling_ratio: float = 1.0,
        save_path: Optional[str] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Capture activations for hierarchical concepts with positive/negative examples.

        Args:
            hierarchies: List of ConceptHierarchy objects
            negative_sampling_ratio: Ratio of negative to positive examples
            save_path: Optional path to save activations

        Returns:
            Nested dict: concept_id -> {'pos': tensor, 'neg': tensor}
        """
        logger.info(
            f"Capturing hierarchical activations for {len(hierarchies)} hierarchies"
        )

        all_activations = {}

        for hierarchy in tqdm(hierarchies, desc="Processing hierarchies"):
            parent_id = hierarchy.parent.synset_id

            # Capture parent activations
            parent_pos_prompts = hierarchy.parent_prompts
            parent_neg_prompts = self._generate_negative_prompts(
                parent_pos_prompts, negative_sampling_ratio
            )

            all_activations[parent_id] = {
                "pos": self._capture_prompts_in_batches(parent_pos_prompts),
                "neg": self._capture_prompts_in_batches(parent_neg_prompts),
            }

            # Capture child activations
            for child in hierarchy.children:
                child_id = child.synset_id
                child_pos_prompts = hierarchy.child_prompts[child_id]
                child_neg_prompts = self._generate_negative_prompts(
                    child_pos_prompts, negative_sampling_ratio
                )

                all_activations[child_id] = {
                    "pos": self._capture_prompts_in_batches(child_pos_prompts),
                    "neg": self._capture_prompts_in_batches(child_neg_prompts),
                }

        # Save if requested
        if save_path:
            self.save_hierarchical_activations(all_activations, save_path)

        return all_activations

    def _capture_prompts_in_batches(self, prompts: List[str]) -> torch.Tensor:
        """Capture activations for prompts in batches."""
        all_activations = []

        for i in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[i : i + self.config.batch_size]
            batch_activations = self.capture_last_token_activations(batch_prompts)
            all_activations.append(batch_activations)

        return torch.cat(all_activations, dim=0)

    def _generate_negative_prompts(
        self, positive_prompts: List[str], ratio: float
    ) -> List[str]:
        """Generate negative prompts using antonyms or external corpora."""

        negative_prompts: List[str] = []
        negation_prefixes = [
            "This is not about",
            "This has nothing to do with",
            "This is unrelated to",
            "This is the opposite o",
            "This contradicts",
        ]

        max_negatives = int(len(positive_prompts) * ratio)

        for prompt in positive_prompts:
            if len(negative_prompts) >= max_negatives:
                break

            tokens = prompt.split()
            replaced = False
            for token in tokens:
                synsets = wn.synsets(token)
                for syn in synsets:
                    for lemma in syn.lemmas():
                        if lemma.antonyms():
                            antonym = lemma.antonyms()[0].name().replace("_", " ")
                            negative_prompts.append(prompt.replace(token, antonym))
                            replaced = True
                            break
                    if replaced:
                        break
                if replaced:
                    break

            if replaced:
                continue

            if self.negative_corpus:
                negative_prompts.append(np.random.choice(self.negative_corpus))
                continue

            prefix = np.random.choice(negation_prefixes)
            negative_prompts.append(f"{prefix} {prompt.lower()}")

        return negative_prompts

    def save_activations(
        self,
        activations: Dict[str, torch.Tensor],
        filepath: str,
        use_async: bool = False,
    ) -> Optional[mp.Process]:
        """Save activations to HDF5 file.

        When ``use_async`` is True, the save operation is executed in a
        separate process and the handle is returned.
        """

        def _save() -> None:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(filepath, "w") as f:
                for concept_id, activation_tensor in activations.items():
                    f.create_dataset(
                        concept_id, data=activation_tensor.to(torch.float32).numpy()
                    )
            logger.info(
                f"Saved activations for {len(activations)} concepts to {filepath}"
            )

        if use_async:
            proc = mp.Process(target=_save)
            proc.start()
            return proc
        _save()
        return None

    def save_hierarchical_activations(
        self,
        activations: Dict[str, Dict[str, torch.Tensor]],
        filepath: str,
        use_async: bool = False,
    ) -> Optional[mp.Process]:
        """Save hierarchical activations to HDF5 file."""

        def _save() -> None:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(filepath, "w") as f:
                for concept_id, pos_neg_dict in activations.items():
                    concept_group = f.create_group(concept_id)
                    for pos_neg, activation_tensor in pos_neg_dict.items():
                        concept_group.create_dataset(
                            pos_neg, data=activation_tensor.to(torch.float32).numpy()
                        )
            logger.info(
                f"Saved hierarchical activations for {len(activations)} concepts to {filepath}"
            )

        if use_async:
            proc = mp.Process(target=_save)
            proc.start()
            return proc
        _save()
        return None

    def load_activations(self, filepath: str) -> Dict[str, torch.Tensor]:
        """Load activations from HDF5 file."""
        activations: Dict[str, torch.Tensor] = {}

        with h5py.File(filepath, "r") as f:
            for concept_id in f.keys():
                activations[concept_id] = torch.from_numpy(f[concept_id][:])

        logger.info(
            f"Loaded activations for {len(activations)} concepts from {filepath}"
        )
        return activations

    def load_hierarchical_activations(
        self, filepath: str
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load hierarchical activations from HDF5 file."""
        activations: Dict[str, Dict[str, torch.Tensor]] = {}

        with h5py.File(filepath, "r") as f:
            for concept_id in f.keys():
                concept_group = f[concept_id]
                activations[concept_id] = {}
                for pos_neg in concept_group.keys():
                    activations[concept_id][pos_neg] = torch.from_numpy(
                        concept_group[pos_neg][:]
                    )

        logger.info(
            f"Loaded hierarchical activations for {len(activations)} concepts from {filepath}"
        )
        return activations


def create_activation_shards(
    activations: Dict[str, torch.Tensor],
    shard_size: int = 1000,
    output_dir: str = "activation_shards",
) -> List[str]:
    """
    Create sharded activation files for efficient loading.

    Args:
        activations: Dictionary of concept activations
        shard_size: Number of samples per shard
        output_dir: Directory to save shards

    Returns:
        List of shard file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_files = []

    # Combine all activations
    all_concept_ids = []
    all_activations = []

    for concept_id, activation_tensor in activations.items():
        n_samples = activation_tensor.shape[0]
        all_concept_ids.extend([concept_id] * n_samples)
        all_activations.append(activation_tensor)

    all_activations = torch.cat(all_activations, dim=0)

    # Create shards
    n_samples = len(all_activations)
    n_shards = (n_samples + shard_size - 1) // shard_size

    for shard_idx in range(n_shards):
        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, n_samples)

        shard_activations = all_activations[start_idx:end_idx]
        shard_concept_ids = all_concept_ids[start_idx:end_idx]

        shard_path = output_path / f"shard_{shard_idx:04d}.h5"

        with h5py.File(shard_path, "w") as f:
            f.create_dataset("activations", data=shard_activations.numpy())
            f.create_dataset(
                "concept_ids", data=[s.encode("utf-8") for s in shard_concept_ids]
            )

        shard_files.append(str(shard_path))

    logger.info(f"Created {n_shards} activation shards in {output_dir}")
    return shard_files
