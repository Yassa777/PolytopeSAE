"""
SAE and H-SAE model architectures.

This module implements standard SAEs and Hierarchical SAEs with Top-K routing,
projectors, and teacher initialization capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging
from dataclasses import dataclass

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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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
            'l1_penalty': self.config.l1_penalty * torch.mean(torch.abs(z)),
            'reconstruction_loss': F.mse_loss(x_hat, x),
            'sparsity': torch.mean((z > 0).float()),
            'mean_activation': torch.mean(z),
        }
        
        return x_hat, z, metrics
    
    def normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm."""
        if self.decoder is not None and self.config.normalize_decoder:
            with torch.no_grad():
                decoder_norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
                self.decoder.weight.div_(decoder_norms + 1e-8)


class HierarchicalSAE(nn.Module):
    """Hierarchical Sparse Autoencoder with Top-K routing."""
    
    def __init__(self, config: HSAEConfig):
        super().__init__()
        self.config = config
        
        # Parent (router) SAE
        self.parent_encoder = nn.Linear(config.input_dim, config.n_parents, bias=True)
        self.parent_decoder = nn.Linear(config.n_parents, config.input_dim, bias=False)
        
        # Child SAEs (one per parent)
        self.child_encoders = nn.ModuleList()
        self.child_decoders = nn.ModuleList()
        self.down_projectors = nn.ModuleList()  # input_dim -> subspace_dim
        self.up_projectors = nn.ModuleList()   # subspace_dim -> input_dim
        
        for _ in range(config.n_parents):
            # Down/up projectors for this parent's subspace
            down_proj = nn.Linear(config.input_dim, config.subspace_dim, bias=False)
            up_proj = nn.Linear(config.subspace_dim, config.input_dim, bias=False)
            
            # Child encoder/decoder in subspace
            child_enc = nn.Linear(config.subspace_dim, config.n_children_per_parent, bias=True)
            child_dec = nn.Linear(config.n_children_per_parent, config.subspace_dim, bias=False)
            
            self.down_projectors.append(down_proj)
            self.up_projectors.append(up_proj)
            self.child_encoders.append(child_enc)
            self.child_decoders.append(child_dec)
        
        # Decoder bias
        self.decoder_bias = nn.Parameter(torch.zeros(config.input_dim))
        
        # Temperature for gumbel softmax (annealed during training)
        self.register_buffer('router_temperature', torch.tensor(config.router_temp_start))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        # Parent encoder/decoder
        nn.init.xavier_uniform_(self.parent_encoder.weight)
        nn.init.zeros_(self.parent_encoder.bias)
        
        # Initialize parent decoder columns to unit norm
        with torch.no_grad():
            parent_norms = torch.norm(self.parent_decoder.weight, dim=0, keepdim=True)
            self.parent_decoder.weight.div_(parent_norms + 1e-8)
        
        # Child components
        for i in range(self.config.n_parents):
            # Projectors: orthogonal initialization
            nn.init.orthogonal_(self.down_projectors[i].weight)
            nn.init.orthogonal_(self.up_projectors[i].weight)
            
            # Child encoder/decoder
            nn.init.xavier_uniform_(self.child_encoders[i].weight)
            nn.init.zeros_(self.child_encoders[i].bias)
            
            with torch.no_grad():
                child_norms = torch.norm(self.child_decoders[i].weight, dim=0, keepdim=True)
                self.child_decoders[i].weight.div_(child_norms + 1e-8)
    
    def initialize_from_teacher(self, 
                              parent_vectors: torch.Tensor,
                              child_projectors: List[Tuple[torch.Tensor, torch.Tensor]],
                              geometry=None):
        """
        Initialize decoder weights from teacher vectors.
        
        Args:
            parent_vectors: Parent concept vectors [n_parents, input_dim]
            child_projectors: List of (down_proj, up_proj) tuples for each parent
            geometry: CausalGeometry instance for normalization
        """
        logger.info("Initializing H-SAE from teacher vectors")
        
        with torch.no_grad():
            # Initialize parent decoder rows
            self.parent_decoder.weight.copy_(parent_vectors.T)  # Transpose for Linear layer
            
            if geometry is not None:
                # Normalize under causal norm
                for i in range(self.config.n_parents):
                    parent_vec = self.parent_decoder.weight[:, i]
                    normalized_vec = geometry.normalize_causal(parent_vec)
                    self.parent_decoder.weight[:, i] = normalized_vec
            
            # Initialize projectors and child decoders
            for i, (down_proj, up_proj) in enumerate(child_projectors):
                if i >= self.config.n_parents:
                    break
                    
                self.down_projectors[i].weight.copy_(down_proj)
                self.up_projectors[i].weight.copy_(up_proj)
                
                # Initialize child decoder with orthogonal basis in subspace
                # This will be learned during training
                nn.init.orthogonal_(self.child_decoders[i].weight)
                
                with torch.no_grad():
                    child_norms = torch.norm(self.child_decoders[i].weight, dim=0, keepdim=True)
                    self.child_decoders[i].weight.div_(child_norms + 1e-8)
    
    def encode_parent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input through parent (router) SAE.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Tuple of (parent_logits, parent_codes)
        """
        parent_logits = self.parent_encoder(x)
        
        # Top-K gating with temperature
        if self.training:
            # Gumbel softmax for differentiable top-k
            parent_probs = F.gumbel_softmax(parent_logits, tau=self.router_temperature, hard=True)
            # Keep only top-k
            topk_vals, topk_indices = torch.topk(parent_logits, self.config.topk_parent, dim=-1)
            parent_codes = torch.zeros_like(parent_logits)
            parent_codes.scatter_(-1, topk_indices, 1.0)
        else:
            # Hard top-k for inference
            topk_vals, topk_indices = torch.topk(parent_logits, self.config.topk_parent, dim=-1)
            parent_codes = torch.zeros_like(parent_logits)
            parent_codes.scatter_(-1, topk_indices, 1.0)
        
        return parent_logits, parent_codes
    
    def encode_children(self, x: torch.Tensor, parent_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input through child SAEs for active parents.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            parent_codes: Parent activation codes [batch_size, n_parents]
            
        Returns:
            Tuple of (child_logits, child_codes) [batch_size, n_parents, n_children_per_parent]
        """
        batch_size = x.shape[0]
        child_logits = torch.zeros(batch_size, self.config.n_parents, self.config.n_children_per_parent, device=x.device)
        child_codes = torch.zeros_like(child_logits)
        
        for parent_idx in range(self.config.n_parents):
            # Check which samples have this parent active
            active_mask = parent_codes[:, parent_idx] > 0
            if not active_mask.any():
                continue
                
            # Project to parent's subspace
            x_subspace = self.down_projectors[parent_idx](x[active_mask])
            
            # Encode in subspace
            child_logits_p = self.child_encoders[parent_idx](x_subspace)
            child_logits[active_mask, parent_idx] = child_logits_p
            
            # Top-K child selection (typically K=1)
            if self.config.topk_child == 1:
                child_codes_p = F.gumbel_softmax(child_logits_p, tau=self.router_temperature, hard=True)
            else:
                topk_vals, topk_indices = torch.topk(child_logits_p, self.config.topk_child, dim=-1)
                child_codes_p = torch.zeros_like(child_logits_p)
                child_codes_p.scatter_(-1, topk_indices, 1.0)
            
            child_codes[active_mask, parent_idx] = child_codes_p
        
        return child_logits, child_codes
    
    def decode(self, parent_codes: torch.Tensor, child_codes: torch.Tensor) -> torch.Tensor:
        """
        Decode parent and child codes to reconstruction.
        
        Args:
            parent_codes: Parent activation codes [batch_size, n_parents]
            child_codes: Child activation codes [batch_size, n_parents, n_children_per_parent]
            
        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        batch_size = parent_codes.shape[0]
        reconstruction = torch.zeros(batch_size, self.config.input_dim, device=parent_codes.device)
        
        # Parent reconstruction
        parent_recon = F.linear(parent_codes, self.parent_decoder.weight.t())
        reconstruction += parent_recon
        
        # Child reconstruction
        for parent_idx in range(self.config.n_parents):
            # Check which samples have this parent active
            active_mask = parent_codes[:, parent_idx] > 0
            if not active_mask.any():
                continue
            
            # Decode child codes in subspace
            child_codes_p = child_codes[active_mask, parent_idx]
            child_recon_subspace = F.linear(child_codes_p, self.child_decoders[parent_idx].weight.t())
            
            # Project back to full space
            child_recon_full = self.up_projectors[parent_idx](child_recon_subspace)
            reconstruction[active_mask] += child_recon_full
        
        # Add bias
        reconstruction += self.decoder_bias
        
        return reconstruction
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
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
        
        # Compute metrics
        metrics = {
            'reconstruction_loss': F.mse_loss(x_hat, x),
            'l1_parent': self.config.l1_parent * torch.mean(torch.abs(parent_codes)),
            'l1_child': self.config.l1_child * torch.mean(torch.abs(child_codes)),
            'parent_sparsity': torch.mean((parent_codes > 0).float()),
            'child_sparsity': torch.mean((child_codes > 0).float()),
            'parent_usage': torch.mean(torch.sum(parent_codes > 0, dim=0).float()),
            'child_usage': torch.mean(torch.sum(child_codes > 0, dim=(0, 1)).float()),
        }
        
        # Bi-orthogonality penalty
        if self.config.biorth_lambda > 0:
            biorth_penalty = 0
            
            # Parent bi-orthogonality: E^T D ≈ I
            parent_enc_dec = self.parent_encoder.weight @ self.parent_decoder.weight
            parent_identity = torch.eye(self.config.n_parents, device=x.device)
            biorth_penalty += torch.norm(parent_enc_dec - parent_identity, 'fro') ** 2
            
            # Child bi-orthogonality
            for i in range(self.config.n_parents):
                child_enc_dec = self.child_encoders[i].weight @ self.child_decoders[i].weight
                child_identity = torch.eye(self.config.n_children_per_parent, device=x.device)
                biorth_penalty += torch.norm(child_enc_dec - child_identity, 'fro') ** 2
            
            metrics['biorth_penalty'] = self.config.biorth_lambda * biorth_penalty
        
        return x_hat, (parent_codes, child_codes), metrics
    
    def compute_causal_orthogonality_penalty(self, geometry) -> torch.Tensor:
        """
        Compute causal orthogonality penalty: ⟨ℓ_p, δ_{c|p}⟩_c ≈ 0.
        
        Args:
            geometry: CausalGeometry instance
            
        Returns:
            Orthogonality penalty scalar
        """
        penalty = 0
        
        for parent_idx in range(self.config.n_parents):
            parent_vector = self.parent_decoder.weight[:, parent_idx]
            
            # For each child of this parent
            for child_idx in range(self.config.n_children_per_parent):
                # Get child vector in full space
                child_subspace = self.child_decoders[parent_idx].weight[:, child_idx]
                child_full = self.up_projectors[parent_idx](child_subspace)
                
                # Compute delta: δ_{c|p} = ℓ_c - ℓ_p
                delta = child_full - parent_vector
                
                # Causal inner product penalty
                inner_prod = geometry.causal_inner_product(parent_vector, delta)
                penalty += inner_prod ** 2
        
        return self.config.causal_ortho_lambda * penalty
    
    def normalize_decoder_weights(self):
        """Normalize decoder columns to unit norm."""
        if self.config.normalize_decoder:
            with torch.no_grad():
                # Parent decoder
                parent_norms = torch.norm(self.parent_decoder.weight, dim=0, keepdim=True)
                self.parent_decoder.weight.div_(parent_norms + 1e-8)
                
                # Child decoders
                for i in range(self.config.n_parents):
                    child_norms = torch.norm(self.child_decoders[i].weight, dim=0, keepdim=True)
                    self.child_decoders[i].weight.div_(child_norms + 1e-8)
    
    def update_router_temperature(self, step: int, total_steps: int):
        """Update router temperature according to schedule."""
        progress = step / total_steps
        temp = self.config.router_temp_start * (1 - progress) + self.config.router_temp_end * progress
        self.router_temperature.fill_(temp)


def create_baseline_hsae(config: HSAEConfig) -> HierarchicalSAE:
    """Create baseline H-SAE with random initialization."""
    return HierarchicalSAE(config)


def create_teacher_initialized_hsae(config: HSAEConfig,
                                  parent_vectors: torch.Tensor,
                                  child_projectors: List[Tuple[torch.Tensor, torch.Tensor]],
                                  geometry=None) -> HierarchicalSAE:
    """Create teacher-initialized H-SAE."""
    hsae = HierarchicalSAE(config)
    hsae.initialize_from_teacher(parent_vectors, child_projectors, geometry)
    return hsae


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_decoder_weights(model: HierarchicalSAE):
    """Freeze decoder weights for stabilization phase."""
    model.parent_decoder.weight.requires_grad = False
    for i in range(model.config.n_parents):
        model.child_decoders[i].weight.requires_grad = False
        model.up_projectors[i].weight.requires_grad = False


def unfreeze_decoder_weights(model: HierarchicalSAE):
    """Unfreeze decoder weights for adaptation phase."""
    model.parent_decoder.weight.requires_grad = True
    for i in range(model.config.n_parents):
        model.child_decoders[i].weight.requires_grad = True
        model.up_projectors[i].weight.requires_grad = True