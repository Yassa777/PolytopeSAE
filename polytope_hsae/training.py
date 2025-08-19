"""
Training loops and schedules for SAE and H-SAE models.

This module implements training routines with support for two-stage training,
temperature annealing, decoder normalization, and comprehensive logging.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import wandb
from pathlib import Path
import json
from datetime import datetime

from .models import HierarchicalSAE, HSAEConfig
from .metrics import compute_explained_variance, compute_cross_entropy_proxy

logger = logging.getLogger(__name__)


class HSAETrainer:
    """Trainer for Hierarchical SAE with support for teacher initialization."""
    
    def __init__(self, 
                 model: HierarchicalSAE,
                 config: Dict[str, Any],
                 geometry=None,
                 use_wandb: bool = True):
        """
        Initialize H-SAE trainer.
        
        Args:
            model: HierarchicalSAE model
            config: Training configuration
            geometry: CausalGeometry instance for causal orthogonality
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model
        self.config = config
        self.geometry = geometry
        self.use_wandb = use_wandb
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Setup scheduler if warmup is specified
        if 'warmup_steps' in config['training']:
            warmup_scheduler = LinearLR(
                self.optimizer, 
                start_factor=1e-8, 
                end_factor=1.0, 
                total_iters=config['training']['warmup_steps']
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=config['training'].get('total_steps', 10000) - config['training']['warmup_steps']
            )
            self.scheduler = SequentialLR(
                self.optimizer, 
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config['training']['warmup_steps']]
            )
        else:
            self.scheduler = None
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.log_dir = Path(config['logging']['save_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def train_step(self, batch: torch.Tensor, stage: str = "main") -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Input batch [batch_size, input_dim]
            stage: Training stage ("stabilize", "adapt", or "main")
            
        Returns:
            Dictionary with loss components and metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unit normalize inputs if specified
        if self.config.get('data', {}).get('unit_norm', False):
            batch = F.normalize(batch, dim=-1)
        
        # Forward pass
        x_hat, (parent_codes, child_codes), metrics = self.model(batch)
        
        # Compute losses
        recon_loss = metrics['reconstruction_loss']
        l1_loss = metrics['l1_parent'] + metrics['l1_child']
        
        # Top-level reconstruction loss (for baseline parity)
        top_level_loss = 0.0
        if hasattr(self.model.config, 'top_level_beta') and self.model.config.top_level_beta > 0:
            top_level_loss = self.model.config.top_level_beta * metrics['top_level_recon_loss']
        
        # Bi-orthogonality loss
        biorth_loss = metrics.get('biorth_penalty', 0.0)
        
        # Causal orthogonality loss (only in stabilize stage for teacher-init)
        causal_ortho_loss = 0.0
        if (stage == "stabilize" and 
            hasattr(self.model.config, 'causal_ortho_lambda') and 
            self.model.config.causal_ortho_lambda > 0 and 
            self.geometry is not None):
            # Ensure geometry is on the same device as model
            self.geometry.to(str(batch.device))
            causal_ortho_loss = self.model.compute_causal_orthogonality_penalty(self.geometry)
        
        # Total loss
        total_loss = recon_loss + l1_loss + top_level_loss + biorth_loss + causal_ortho_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if 'grad_clip' in self.config['training']:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training']['grad_clip']
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update router temperature
        total_steps = self.config['training'].get('total_steps', 10000)
        self.model.update_router_temperature(self.step, total_steps)
        
        # Normalize decoder weights
        self.model.normalize_decoder_weights()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.step += 1
        
        # Compute additional metrics
        with torch.no_grad():
            ev = compute_explained_variance(batch, x_hat)
            ce_proxy = compute_cross_entropy_proxy(batch, x_hat)
        
        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'l1_loss': l1_loss.item(),
            'top_level_loss': top_level_loss,
            'biorth_loss': biorth_loss if isinstance(biorth_loss, float) else biorth_loss.item(),
            'causal_ortho_loss': causal_ortho_loss if isinstance(causal_ortho_loss, float) else causal_ortho_loss.item(),
            '1-EV': 1.0 - ev,
            '1-CE': ce_proxy,
            'parent_sparsity': metrics['parent_sparsity'].item(),
            'child_sparsity': metrics['child_sparsity'].item(),
            'active_parents_per_sample': metrics['active_parents_per_sample'].item(),
            'active_children_per_sample': metrics['active_children_per_sample'].item(),
            'router_temperature': metrics['router_temperature'],
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    def train_baseline(self, 
                      train_loader, 
                      val_loader=None,
                      total_steps: int = 7000) -> Dict[str, List[float]]:
        """
        Train baseline H-SAE (single stage, random initialization).
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            total_steps: Total training steps
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"Training baseline H-SAE for {total_steps} steps")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'step': []
        }
        
        # Training loop
        train_iter = iter(train_loader)
        pbar = tqdm(range(total_steps), desc="Baseline Training")
        
        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Assume first element is the data
            
            batch = batch.to(next(self.model.parameters()).device)
            
            # Training step
            metrics = self.train_step(batch, stage="main")
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{metrics['total_loss']:.4f}",
                '1-EV': f"{metrics['1-EV']:.4f}",
                'Temp': f"{metrics['router_temperature']:.3f}"
            })
            
            # Logging
            if step % self.config['logging']['log_every'] == 0:
                history['train_loss'].append(metrics['total_loss'])
                history['step'].append(step)
                
                if self.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
                
                # Validation
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    history['val_loss'].append(val_metrics['total_loss'])
                    
                    if self.use_wandb:
                        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
            
            # Checkpointing
            if step % self.config['logging']['checkpoint_every'] == 0:
                self.save_checkpoint(step, stage="baseline")
        
        logger.info("Baseline training completed")
        return history
    
    def train_teacher_init(self,
                          train_loader,
                          val_loader=None,
                          stabilize_steps: int = 1500,
                          adapt_steps: int = 8500) -> Dict[str, List[float]]:
        """
        Train teacher-initialized H-SAE with two-stage schedule.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            stabilize_steps: Steps for stabilization phase (decoder frozen)
            adapt_steps: Steps for adaptation phase (decoder unfrozen)
            
        Returns:
            Dictionary with training history
        """
        total_steps = stabilize_steps + adapt_steps
        logger.info(f"Training teacher-initialized H-SAE: {stabilize_steps} stabilize + {adapt_steps} adapt = {total_steps} total steps")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'step': [],
            'stage': []
        }
        
        train_iter = iter(train_loader)
        
        # Stage A: Stabilize (freeze decoder)
        logger.info("Stage A: Stabilizing with frozen decoder")
        self._freeze_decoder_weights()
        
        pbar = tqdm(range(stabilize_steps), desc="Stage A: Stabilize")
        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(next(self.model.parameters()).device)
            
            # Training step with causal orthogonality
            metrics = self.train_step(batch, stage="stabilize")
            
            pbar.set_postfix({
                'Loss': f"{metrics['total_loss']:.4f}",
                '1-EV': f"{metrics['1-EV']:.4f}",
                'Causal': f"{metrics['causal_ortho_loss']:.6f}"
            })
            
            # Logging
            if step % self.config['logging']['log_every'] == 0:
                history['train_loss'].append(metrics['total_loss'])
                history['step'].append(step)
                history['stage'].append('stabilize')
                
                if self.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
                    wandb.log({"stage": "stabilize"}, step=step)
                
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    history['val_loss'].append(val_metrics['total_loss'])
                    
                    if self.use_wandb:
                        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
        
        # Stage B: Adapt (unfreeze decoder)
        logger.info("Stage B: Adapting with unfrozen decoder")
        self._unfreeze_decoder_weights()
        
        # Optionally reduce decoder learning rate
        decoder_lr_mult = self.config['training'].get('teacher_init', {}).get('decoder_lr_mult', 1.0)
        if decoder_lr_mult != 1.0:
            self._set_decoder_lr(self.config['training']['lr'] * decoder_lr_mult)
        
        pbar = tqdm(range(adapt_steps), desc="Stage B: Adapt")
        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            batch = batch.to(next(self.model.parameters()).device)
            
            # Training step without causal orthogonality
            metrics = self.train_step(batch, stage="adapt")
            
            pbar.set_postfix({
                'Loss': f"{metrics['total_loss']:.4f}",
                '1-EV': f"{metrics['1-EV']:.4f}",
                'Biorth': f"{metrics['biorth_loss']:.6f}"
            })
            
            # Logging
            global_step = stabilize_steps + step
            if step % self.config['logging']['log_every'] == 0:
                history['train_loss'].append(metrics['total_loss'])
                history['step'].append(global_step)
                history['stage'].append('adapt')
                
                if self.use_wandb:
                    wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=global_step)
                    wandb.log({"stage": "adapt"}, step=global_step)
                
                if val_loader is not None:
                    val_metrics = self.validate(val_loader)
                    history['val_loss'].append(val_metrics['total_loss'])
                    
                    if self.use_wandb:
                        wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)
            
            # Checkpointing
            if step % self.config['logging']['checkpoint_every'] == 0:
                self.save_checkpoint(global_step, stage="teacher_init")
        
        logger.info("Teacher-initialized training completed")
        return history
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        val_metrics = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                
                batch = batch.to(next(self.model.parameters()).device)
                
                if self.config.get('data', {}).get('unit_norm', False):
                    batch = F.normalize(batch, dim=-1)
                
                x_hat, (parent_codes, child_codes), metrics = self.model(batch)
                
                # Compute validation metrics
                recon_loss = metrics['reconstruction_loss']
                l1_loss = metrics['l1_parent'] + metrics['l1_child']
                biorth_loss = metrics.get('biorth_penalty', 0.0)
                total_loss = recon_loss + l1_loss + biorth_loss
                
                ev = compute_explained_variance(batch, x_hat)
                ce_proxy = compute_cross_entropy_proxy(batch, x_hat)
                
                val_metrics.append({
                    'total_loss': total_loss.item(),
                    'recon_loss': recon_loss.item(),
                    '1-EV': 1.0 - ev,
                    '1-CE': ce_proxy,
                    'parent_sparsity': metrics['parent_sparsity'].item(),
                    'child_sparsity': metrics['child_sparsity'].item(),
                })
        
        # Average metrics
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in val_metrics) / len(val_metrics)
        
        return avg_metrics
    
    def _freeze_decoder_weights(self):
        """Freeze decoder weights for stabilization phase."""
        self.model.parent_decoder.weight.requires_grad = False
        for i in range(self.model.config.n_parents):
            self.model.child_decoders[i].weight.requires_grad = False
            if not self.model.config.tie_projectors:
                self.model.up_projectors[i].weight.requires_grad = False
    
    def _unfreeze_decoder_weights(self):
        """Unfreeze decoder weights for adaptation phase."""
        if not self.model.config.use_tied_decoders_parent:
            self.model.parent_decoder.weight.requires_grad = True
        for i in range(self.model.config.n_parents):
            if not self.model.config.use_tied_decoders_child:
                self.model.child_decoders[i].weight.requires_grad = True
            if not self.model.config.tie_projectors:
                self.model.up_projectors[i].weight.requires_grad = True
    
    def _set_decoder_lr(self, lr: float):
        """Set different learning rate for decoder parameters."""
        decoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'decoder' in name or 'up_projector' in name:
                decoder_params.append(param)
            else:
                other_params.append(param)
        
        self.optimizer = AdamW([
            {'params': other_params, 'lr': self.config['training']['lr']},
            {'params': decoder_params, 'lr': lr}
        ], weight_decay=self.config['training']['weight_decay'])
    
    def save_checkpoint(self, step: int, stage: str = "main"):
        """Save model checkpoint."""
        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"hsae_step_{step}_{stage}.pt"
        
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'stage': stage,
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Clean up old checkpoints
        if hasattr(self.config['logging'], 'keep_last_k'):
            self._cleanup_checkpoints(checkpoint_dir, keep_last_k=self.config['logging']['keep_last_k'])
    
    def _cleanup_checkpoints(self, checkpoint_dir: Path, keep_last_k: int = 5):
        """Keep only the last k checkpoints."""
        checkpoints = list(checkpoint_dir.glob("hsae_step_*.pt"))
        if len(checkpoints) > keep_last_k:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.stem.split('_')[2]))
            # Remove old checkpoints
            for checkpoint in checkpoints[:-keep_last_k]:
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        logger.info(f"Loaded checkpoint from step {self.step}")
        return checkpoint


def create_data_loader(activations: torch.Tensor, 
                      batch_size: int = 8192,
                      shuffle: bool = True) -> torch.utils.data.DataLoader:
    """Create data loader from activation tensor."""
    dataset = torch.utils.data.TensorDataset(activations)
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False
    )