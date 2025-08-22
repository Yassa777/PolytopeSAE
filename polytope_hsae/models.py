"""
SAE and H-SAE model architectures.

This module implements standard SAEs and Hierarchical SAEs with Top-K routing,
projectors, and teacher initialization capabilities.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


@dataclass
class SAEConfig:
    """Configuration for standard SAE."""

    input_dim: int
    hidden_dim: int
    l1_penalty: float = 1e-3
    normalize_decoder: bool = True
    tied_weights: bool = False


@dataclass
class HSAEConfig:
    """Configuration for Hierarchical SAE."""

    input_dim: int
    n_parents: int
    topk_parent: int
    subspace_dim: int
    n_children_per_parent: int
    topk_child: int = 1
    l1_parent: float = 1e-3
    l1_child: float = 1e-3
    biorth_lambda: float = 1e-3
    causal_ortho_lambda: float = 3e-4
    router_temp_start: float = 1.5
    router_temp_end: float = 0.7
    normalize_decoder: bool = True
    # "euclidean" or "causal" normalization of decoder columns
    normalize_decoder_mode: str = "euclidean"
    top_level_beta: float = 0.1

    # JAX-compatibility toggles
    use_tied_decoders_parent: bool = False
    use_tied_decoders_child: bool = False
    tie_projectors: bool = False
    use_decoder_bias: bool = True
    use_offdiag_biorth: bool = False  # Use off-diagonal-only cross-orthogonality
    use_gradient_checkpointing: bool = False
    # Routing behaviour: "hard" (Top-K with ST) or "soft" mixture
    routing_forward_mode: str = "hard"


class StandardSAE(nn.Module):
    """Standard Sparse Autoencoder."""

    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config

        # Encoder and decoder
        self.encoder = nn.Linear(config.input_dim, config.hidden_dim, bias=True)

        if config.tied_weights:
            # Tied weights: decoder is transpose of encoder
            self.decoder = None
        else:
            self.decoder = nn.Linear(config.hidden_dim, config.input_dim, bias=False)

        # Decoder bias (separate from encoder bias)
        self.decoder_bias = nn.Parameter(torch.zeros(config.input_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        # Xavier initialization for encoder
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if self.decoder is not None:
            # Initialize decoder columns to unit norm
            with torch.no_grad():
                decoder_norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
                self.decoder.weight.div_(decoder_norms + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse representation."""
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode sparse representation to reconstruction."""
        if self.decoder is not None:
            return F.linear(z, self.decoder.weight.t()) + self.decoder_bias
        else:
            # Tied weights
            return F.linear(z, self.encoder.weight) + self.decoder_bias

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of (reconstruction, latent_codes, metrics)
        """
        # Encode
        z = self.encode(x)

        # Decode
        x_hat = self.decode(z)

        # Compute metrics
        metrics = {
            "l1_penalty": self.config.l1_penalty * torch.mean(torch.abs(z)),
            "reconstruction_loss": F.mse_loss(x_hat, x),
            "sparsity": torch.mean((z > 0).float()),
            "mean_activation": torch.mean(z),
        }

        return x_hat, z, metrics

    def normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm."""
        if self.decoder is not None and self.config.normalize_decoder:
            with torch.no_grad():
                decoder_norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
                self.decoder.weight.div_(decoder_norms + 1e-8)


class Router(nn.Module):
    """Top-level router projecting inputs to parent codes."""

    def __init__(self, input_dim: int, n_parents: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, n_parents, bias=True)
        self.decoder = nn.Linear(n_parents, input_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, codes: torch.Tensor, tied: bool) -> torch.Tensor:
        if tied:
            return F.linear(codes, self.encoder.weight)
        return F.linear(codes, self.decoder.weight)


class SubspaceModule(nn.Module):
    """Handles per-parent subspace projections and child SAE."""

    def __init__(self, input_dim: int, subspace_dim: int, n_children: int):
        super().__init__()
        self.down_projector = nn.Linear(input_dim, subspace_dim, bias=False)
        self.up_projector = nn.Linear(subspace_dim, input_dim, bias=False)
        self.encoder = nn.Linear(subspace_dim, n_children, bias=True)
        self.decoder = nn.Linear(n_children, subspace_dim, bias=False)

    def project_down(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_projector(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, tied: bool) -> torch.Tensor:
        if tied:
            return F.linear(z, self.encoder.weight)
        return F.linear(z, self.decoder.weight)

    def project_up(self, x: torch.Tensor, tied: bool) -> torch.Tensor:
        if tied:
            return F.linear(x, self.down_projector.weight)
        return self.up_projector(x)


class HierarchicalSAE(nn.Module):
    """Hierarchical Sparse Autoencoder with Top-K routing."""

    def __init__(self, config: HSAEConfig):
        super().__init__()
        self.config = config

        # Router module
        self.router = Router(config.input_dim, config.n_parents)

        # Subspace modules, one per parent
        self.subspaces = nn.ModuleList(
            [
                SubspaceModule(
                    config.input_dim, config.subspace_dim, config.n_children_per_parent
                )
                for _ in range(config.n_parents)
            ]
        )

        # Decoder bias
        self.decoder_bias = nn.Parameter(torch.zeros(config.input_dim))

        # Temperature for gumbel softmax (annealed during training)
        self.register_buffer(
            "router_temperature", torch.tensor(config.router_temp_start)
        )

        self._initialize_weights()

    def _sample_codes(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Sample codes according to routing mode."""
        if self.config.routing_forward_mode == "soft":
            return F.softmax(logits / (self.router_temperature + 1e-8), dim=-1)

        if k == 1:
            if self.training:
                return F.gumbel_softmax(logits, tau=self.router_temperature, hard=True)
            codes = torch.zeros_like(logits)
            idx = logits.argmax(dim=-1, keepdim=True)
            codes.scatter_(-1, idx, 1.0)
            return codes

        if self.training:
            noise = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(noise.clamp_min(1e-8)).clamp_min(1e-8))
            logits_noisy = logits + gumbel * self.router_temperature
            topk_vals, topk_idx = torch.topk(logits_noisy, k, dim=-1)
            hard_mask = torch.zeros_like(logits)
            hard_mask.scatter_(-1, topk_idx, 1.0)
            soft = F.softmax(logits / (self.router_temperature + 1e-8), dim=-1)
            soft_masked = soft * hard_mask
            soft_masked = soft_masked / (soft_masked.sum(dim=-1, keepdim=True) + 1e-8)
            return (hard_mask - soft_masked).detach() + soft_masked

        topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
        codes = torch.zeros_like(logits)
        codes.scatter_(-1, topk_idx, 1.0)
        return codes

    def _initialize_weights(self):
        """Initialize weights."""
        # Router weights
        nn.init.xavier_uniform_(self.router.encoder.weight)
        nn.init.zeros_(self.router.encoder.bias)
        with torch.no_grad():
            parent_norms = torch.norm(self.router.decoder.weight, dim=0, keepdim=True)
            self.router.decoder.weight.div_(parent_norms + 1e-8)

        # Subspace modules
        for sub in self.subspaces:
            nn.init.orthogonal_(sub.down_projector.weight)
            nn.init.orthogonal_(sub.up_projector.weight)
            nn.init.xavier_uniform_(sub.encoder.weight)
            nn.init.zeros_(sub.encoder.bias)
            with torch.no_grad():
                child_norms = torch.norm(sub.decoder.weight, dim=0, keepdim=True)
                sub.decoder.weight.div_(child_norms + 1e-8)

    def initialize_from_teacher(
        self,
        parent_vectors: torch.Tensor,
        child_projectors: List[Tuple[torch.Tensor, torch.Tensor]],
        geometry=None,
        child_deltas: Optional[torch.Tensor] = None,
        seed_parent_decoder: bool = True,
        seed_child_decoder: bool = True,
        seed_projectors: bool = True,
        child_mode: str = "delta",
        init_jitter_deg: float = 0.0,
    ):
        """
        Initialize decoder weights from teacher vectors.

        Args:
            parent_vectors: Parent concept vectors [n_parents, input_dim]
            child_projectors: List of (down_proj, up_proj) tuples for each parent
            geometry: CausalGeometry instance for normalization
        """
        logger.info("Initializing H-SAE from teacher vectors")

        with torch.no_grad():
            if init_jitter_deg > 0 and geometry is not None:
                for i in range(parent_vectors.shape[0]):
                    parent_vectors[i] = geometry.random_rotate(parent_vectors[i], init_jitter_deg)
                if child_deltas is not None:
                    for i in range(child_deltas.shape[0]):
                        for j in range(child_deltas.shape[1]):
                            child_deltas[i, j] = geometry.random_rotate(child_deltas[i, j], init_jitter_deg)

            if seed_parent_decoder:
                # Seed as many parents as we have teacher vectors
                n_avail = parent_vectors.shape[0]
                n_model = self.config.n_parents
                n = min(n_avail, n_model)
                self.router.decoder.weight[:, :n].copy_(parent_vectors[:n].T)
                if n < n_model:
                    logger.warning(f"Only seeding {n}/{n_model} parents from teacher; leaving the rest random.")
                if geometry is not None:
                    for i in range(n):
                        parent_vec = self.router.decoder.weight[:, i]
                        self.router.decoder.weight[:, i] = geometry.normalize_causal(parent_vec)

            if seed_projectors:
                for i, (down_proj, up_proj) in enumerate(child_projectors):
                    if i >= self.config.n_parents:
                        break
                    sub = self.subspaces[i]
                    sub.down_projector.weight.copy_(down_proj)
                    sub.up_projector.weight.copy_(up_proj)

            for i in range(self.config.n_parents):
                sub = self.subspaces[i]
                if seed_child_decoder and child_deltas is not None and i < child_deltas.shape[0]:
                    for j in range(min(self.config.n_children_per_parent, child_deltas.shape[1])):
                        delta = child_deltas[i, j]
                        if child_mode == "absolute" and seed_parent_decoder and i < parent_vectors.shape[0]:
                            vec = delta
                        else:
                            vec = delta
                        sub.decoder.weight[:, j] = sub.down_projector.weight @ vec
                    child_norms = torch.norm(sub.decoder.weight, dim=0, keepdim=True)
                    sub.decoder.weight.div_(child_norms + 1e-8)
                else:
                    nn.init.orthogonal_(sub.decoder.weight)
                    child_norms = torch.norm(sub.decoder.weight, dim=0, keepdim=True)
                    sub.decoder.weight.div_(child_norms + 1e-8)

    def encode_parent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input through parent (router) SAE.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of (parent_logits, parent_codes)
        """
        parent_logits = self.router.encode(x)
        parent_codes = self._sample_codes(parent_logits, self.config.topk_parent)
        return parent_logits, parent_codes

    def encode_children(
        self, x: torch.Tensor, parent_codes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input through child SAEs for active parents.

        Args:
            x: Input tensor [batch_size, input_dim]
            parent_codes: Parent activation codes [batch_size, n_parents]

        Returns:
            Tuple of (child_logits, child_codes) [batch_size, n_parents, n_children_per_parent]
        """
        batch_size = x.shape[0]
        child_logits = torch.zeros(
            batch_size,
            self.config.n_parents,
            self.config.n_children_per_parent,
            device=x.device,
            dtype=x.dtype,
        )
        child_codes = torch.zeros_like(child_logits)

        for parent_idx, sub in enumerate(self.subspaces):
            active_mask = parent_codes[:, parent_idx] > 0
            if not active_mask.any():
                continue

            x_subspace = sub.project_down(x[active_mask])
            if self.config.use_gradient_checkpointing and x_subspace.requires_grad:
                logits_p = checkpoint(sub.encode, x_subspace)
            else:
                logits_p = sub.encode(x_subspace)
            # ensure dtype matches destination (bf16 under AMP)
            if logits_p.dtype != child_logits.dtype:
                logits_p = logits_p.to(child_logits.dtype)
            child_logits[active_mask, parent_idx] = logits_p

            child_codes_p = self._sample_codes(logits_p, self.config.topk_child)
            if child_codes_p.dtype != child_codes.dtype:
                child_codes_p = child_codes_p.to(child_codes.dtype)
            child_codes[active_mask, parent_idx] = child_codes_p

        return child_logits, child_codes

    def decode(
        self, parent_codes: torch.Tensor, child_codes: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode parent and child codes to reconstruction.

        Args:
            parent_codes: Parent activation codes [batch_size, n_parents]
            child_codes: Child activation codes [batch_size, n_parents, n_children_per_parent]

        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        batch_size = parent_codes.shape[0]
        reconstruction = torch.zeros(
            batch_size, self.config.input_dim,
            device=parent_codes.device, dtype=parent_codes.dtype
        )

        parent_recon = self.router.decode(
            parent_codes, self.config.use_tied_decoders_parent
        )
        reconstruction += parent_recon

        for parent_idx, sub in enumerate(self.subspaces):
            active_mask = parent_codes[:, parent_idx] > 0
            if not active_mask.any():
                continue

            child_codes_p = child_codes[active_mask, parent_idx]
            if self.config.use_gradient_checkpointing and child_codes_p.requires_grad:

                def _decode(cc):
                    return sub.decode(cc, self.config.use_tied_decoders_child)

                child_recon_subspace = checkpoint(_decode, child_codes_p)
            else:
                child_recon_subspace = sub.decode(
                    child_codes_p, self.config.use_tied_decoders_child
                )

            if (
                self.config.use_gradient_checkpointing
                and child_recon_subspace.requires_grad
            ):

                def _project(u):
                    return sub.project_up(u, self.config.tie_projectors)

                child_recon_full = checkpoint(_project, child_recon_subspace)
            else:
                child_recon_full = sub.project_up(
                    child_recon_subspace, self.config.tie_projectors
                )

            reconstruction[active_mask] += child_recon_full

        if self.config.use_decoder_bias:
            reconstruction += self.decoder_bias.to(reconstruction.dtype)

        return reconstruction

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        """
        Forward pass through hierarchical SAE.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            Tuple of (reconstruction, (parent_codes, child_codes), metrics)
        """
        # Encode through hierarchy
        parent_logits, parent_codes = self.encode_parent(x)
        child_logits, child_codes = self.encode_children(x, parent_codes)

        # Decode
        x_hat = self.decode(parent_codes, child_codes)

        # Compute top-level reconstruction for baseline parity
        parent_recon = self.router.decode(
            parent_codes, self.config.use_tied_decoders_parent
        )
        if self.config.use_decoder_bias:
            parent_recon += self.decoder_bias.to(parent_recon.dtype)

        # Compute metrics
        metrics = {
            "reconstruction_loss": F.mse_loss(x_hat, x),
            "top_level_recon_loss": F.mse_loss(parent_recon, x),
            "l1_parent": self.config.l1_parent * torch.mean(torch.abs(parent_codes)),
            "l1_child": self.config.l1_child * torch.mean(torch.abs(child_codes)),
            "parent_sparsity": torch.mean((parent_codes > 0).float()),
            "child_sparsity": torch.mean((child_codes > 0).float()),
            "parent_usage": torch.mean(torch.sum(parent_codes > 0, dim=0).float()),
            "child_usage": torch.mean(torch.sum(child_codes > 0, dim=(0, 1)).float()),
            "active_parents_per_sample": torch.mean(
                torch.sum(parent_codes > 0, dim=1).float()
            ),
            "active_children_per_sample": torch.mean(
                torch.sum(child_codes > 0, dim=(1, 2)).float()
            ),
            "router_temperature": self.router_temperature.item(),
        }

        # Bi-orthogonality penalty
        if self.config.biorth_lambda > 0:
            if self.config.use_offdiag_biorth:
                # JAX-style: off-diagonal penalty only
                biorth_penalty = self._biorth_penalty_parent_offdiag()
                for i in range(self.config.n_parents):
                    biorth_penalty += self._biorth_penalty_child_offdiag(i)
            else:
                # Our style: E^T D ≈ I penalty
                biorth_penalty = 0

                if not self.config.use_tied_decoders_parent:
                    parent_enc_dec = (
                        self.router.encoder.weight @ self.router.decoder.weight
                    )
                    parent_identity = torch.eye(self.config.n_parents, device=x.device)
                    biorth_penalty += (
                        torch.norm(parent_enc_dec - parent_identity, "fro") ** 2
                    )

                if not self.config.use_tied_decoders_child:
                    for sub in self.subspaces:
                        child_enc_dec = sub.encoder.weight @ sub.decoder.weight
                        child_identity = torch.eye(
                            self.config.n_children_per_parent, device=x.device
                        )
                        biorth_penalty += (
                            torch.norm(child_enc_dec - child_identity, "fro") ** 2
                        )

            metrics["biorth_penalty"] = self.config.biorth_lambda * biorth_penalty

        return x_hat, (parent_codes, child_codes), metrics

    def compute_causal_orthogonality_penalty(self, geometry) -> torch.Tensor:
        """
        Compute causal orthogonality penalty: ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0.

        Args:
            geometry: CausalGeometry instance

        Returns:
            Orthogonality penalty scalar
        """
        device = self.router.decoder.weight.device
        W = geometry.W.to(device=device, dtype=self.router.decoder.weight.dtype)  # [d,d]

        P = self.router.decoder.weight  # [d, P]
        P_w = W @ P                     # [d, P]

        penalty = P.new_zeros(())
        # Loop parents (different modules per parent), but not children
        for i, sub in enumerate(self.subspaces):
            if self.config.tie_projectors:
                child_full = sub.down_projector.weight.t() @ sub.decoder.weight
            else:
                child_full = sub.up_projector.weight @ sub.decoder.weight
            delta = child_full - P[:, i:i+1]

            Pw_i = P_w[:, i:i+1]
            Dw_i = W @ delta
            inner = (Pw_i * Dw_i).sum(dim=0)
            penalty = penalty + (inner ** 2).sum()

        return self.config.causal_ortho_lambda * penalty

    def normalize_decoder_weights(self, geometry=None):
        """Normalize decoder columns to unit norm."""
        if not self.config.normalize_decoder:
            return
        with torch.no_grad():
            if self.config.normalize_decoder_mode == "causal" and geometry is not None:
                for i in range(self.config.n_parents):
                    col = self.router.decoder.weight[:, i]
                    self.router.decoder.weight[:, i] = geometry.normalize_causal(col)

                for sub in self.subspaces:
                    for j in range(self.config.n_children_per_parent):
                        col = sub.decoder.weight[:, j]
                        if self.config.tie_projectors:
                            full = F.linear(col.unsqueeze(0), sub.down_projector.weight).squeeze(0)
                        else:
                            full = sub.up_projector(col.unsqueeze(0)).squeeze(0)
                        full = geometry.normalize_causal(full)
                        if self.config.tie_projectors:
                            col_new = F.linear(full.unsqueeze(0), sub.down_projector.weight.t()).squeeze(0)
                        else:
                            col_new = sub.down_projector.weight @ full
                        sub.decoder.weight[:, j] = col_new
            else:
                parent_norms = torch.norm(
                    self.router.decoder.weight, dim=0, keepdim=True
                )
                self.router.decoder.weight.div_(parent_norms + 1e-8)

                for sub in self.subspaces:
                    child_norms = torch.norm(sub.decoder.weight, dim=0, keepdim=True)
                    sub.decoder.weight.div_(child_norms + 1e-8)

    def update_router_temperature(self, step: int, total_steps: int):
        """Update router temperature according to schedule."""
        progress = step / total_steps
        temp = (
            self.config.router_temp_start * (1 - progress)
            + self.config.router_temp_end * progress
        )
        self.router_temperature.fill_(temp)

    def _biorth_penalty_parent_offdiag(self) -> torch.Tensor:
        """JAX-style off-diagonal cross-orthogonality penalty for parent."""
        E = F.normalize(self.router.encoder.weight, dim=1)  # [P, d]
        if self.config.use_tied_decoders_parent:
            D = F.normalize(self.router.encoder.weight, dim=1)  # [P, d]
        else:
            D = F.normalize(self.router.decoder.weight.t(), dim=0)  # [P, d]
        M = E @ D.t()  # [P, P]
        off = M - torch.diag(torch.diag(M))
        return off.pow(2).sum() / (M.numel() - M.shape[0])

    def _biorth_penalty_child_offdiag(self, parent_idx: int) -> torch.Tensor:
        """JAX-style off-diagonal cross-orthogonality penalty for child."""
        sub = self.subspaces[parent_idx]
        E = F.normalize(sub.encoder.weight, dim=1)  # [C, s]
        if self.config.use_tied_decoders_child:
            D = F.normalize(sub.encoder.weight, dim=1)  # [C, s]
        else:
            D = F.normalize(sub.decoder.weight.t(), dim=0)  # [C, s]
        M = E @ D.t()  # [C, C]
        off = M - torch.diag(torch.diag(M))
        return off.pow(2).sum() / (M.numel() - M.shape[0])


def create_baseline_hsae(config: HSAEConfig) -> HierarchicalSAE:
    """Create baseline H-SAE with random initialization."""
    return HierarchicalSAE(config)


def create_teacher_initialized_hsae(
    config: HSAEConfig,
    parent_vectors: torch.Tensor,
    child_projectors: List[Tuple[torch.Tensor, torch.Tensor]],
    geometry=None,
    child_deltas: Optional[torch.Tensor] = None,
    seed_parent_decoder: bool = True,
    seed_child_decoder: bool = True,
    seed_projectors: bool = True,
    child_mode: str = "delta",
    init_jitter_deg: float = 0.0,
) -> HierarchicalSAE:
    """Create teacher-initialized H-SAE."""
    hsae = HierarchicalSAE(config)
    hsae.initialize_from_teacher(
        parent_vectors,
        child_projectors,
        geometry,
        child_deltas=child_deltas,
        seed_parent_decoder=seed_parent_decoder,
        seed_child_decoder=seed_child_decoder,
        seed_projectors=seed_projectors,
        child_mode=child_mode,
        init_jitter_deg=init_jitter_deg,
    )
    return hsae


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_decoder_weights(model: HierarchicalSAE):
    """Freeze decoder weights for stabilization phase."""
    parts = getattr(model.config, "freeze_parts", ["parent_decoder", "child_decoders", "projectors"])
    if "parent_decoder" in parts and not model.config.use_tied_decoders_parent:
        model.router.decoder.weight.requires_grad = False
    for sub in model.subspaces:
        if "child_decoders" in parts and not model.config.use_tied_decoders_child:
            sub.decoder.weight.requires_grad = False
        if "projectors" in parts and not model.config.tie_projectors:
            sub.up_projector.weight.requires_grad = False
            sub.down_projector.weight.requires_grad = False


def unfreeze_decoder_weights(model: HierarchicalSAE):
    """Unfreeze decoder weights for adaptation phase."""
    parts = getattr(model.config, "freeze_parts", ["parent_decoder", "child_decoders", "projectors"])
    if "parent_decoder" in parts and not model.config.use_tied_decoders_parent:
        model.router.decoder.weight.requires_grad = True
    for sub in model.subspaces:
        if "child_decoders" in parts and not model.config.use_tied_decoders_child:
            sub.decoder.weight.requires_grad = True
        if "projectors" in parts and not model.config.tie_projectors:
            sub.up_projector.weight.requires_grad = True
            sub.down_projector.weight.requires_grad = True
