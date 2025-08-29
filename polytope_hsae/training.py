"""
Training loops and schedules for SAE and H-SAE models.

This module implements training routines with support for two-stage training,
temperature annealing, decoder normalization, and comprehensive logging.
"""

import logging
import contextlib
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None
import json
from datetime import datetime
from pathlib import Path

from .metrics import compute_cross_entropy_proxy, compute_explained_variance
from .models import HierarchicalSAE, HSAEConfig

logger = logging.getLogger(__name__)


class TrainingRestartException(Exception):
    """Exception raised when training needs to restart due to sanity check fixes."""
    pass


class HSAETrainer:
    """Trainer for Hierarchical SAE with support for teacher initialization."""

    def __init__(
        self,
        model: HierarchicalSAE,
        config: Dict[str, Any],
        geometry=None,
        use_wandb: bool = True,
        use_compile: bool = True,
    ):
        """
        Initialize H-SAE trainer.

        Args:
            model: HierarchicalSAE model
            config: Training configuration
            geometry: CausalGeometry instance for causal orthogonality
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.geometry = geometry
        self.use_wandb = use_wandb and wandb is not None
        device_str = config.get("run", {}).get("device", "cuda:0")
        if device_str.startswith("cuda") and not torch.cuda.is_available():
            device_str = "cpu"
        self.device = torch.device(device_str)
        self.model = model.to(self.device)

        # Fast math for Ampere/Blackwell
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        fused_ok = torch.cuda.is_available() and "fused" in AdamW.__init__.__code__.co_varnames
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
            betas=config["training"].get("betas", (0.9, 0.95)),
            fused=fused_ok,
        )

        # Setup scheduler if warmup is specified
        if "warmup_steps" in config["training"]:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=config["training"]["warmup_steps"],
            )
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config["training"].get("total_steps", 10000)
                - config["training"]["warmup_steps"],
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[config["training"]["warmup_steps"]],
            )
        else:
            self.scheduler = None

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.grad_clip = self.config["training"].get("grad_clip")

        if use_compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model, mode="max-autotune")

        # Logging
        self.log_dir = Path(config["logging"]["save_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: torch.Tensor, stage: str = "main") -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(self.device, non_blocking=True)
        x = self._normalize_batch(x)

        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            x_hat, (parent_codes, child_codes), metrics = self.model(x)
            # NaN/Inf guards on forward outputs
            for name, t in {
                "x_hat": x_hat,
                "parent_codes": parent_codes,
                "child_codes": child_codes,
            }.items():
                if not torch.isfinite(t).all():
                    raise FloatingPointError(f"Non-finite values in {name} during forward")
            loss_dict = self._compute_total_loss(x, x_hat, metrics, stage, parent_codes)

        self._optimizer_step(loss_dict["total_loss"])

        # Anneal router temperature with the correct schedule horizon
        horizon = None
        tcfg = self.config.get("training", {})
        if "teacher_init" in tcfg:
            a = int(tcfg["teacher_init"].get("stage_A", {}).get("steps", 0))
            b = int(tcfg["teacher_init"].get("stage_B", {}).get("steps", 0))
            horizon = max(1, a + b)
        if horizon is None or horizon <= 1:
            horizon = int(tcfg.get("baseline", {}).get("total_steps", tcfg.get("total_steps", 10000)))
        if hasattr(self.model, "update_router_temperature"):
            self.model.update_router_temperature(self.step, horizon)
        if hasattr(self.model, "normalize_decoder_weights"):
            # Support both causal and euclidean normalization based on config
            normalize_mode = self.config.get("hsae", {}).get("normalize_decoder", "euclidean")
            if normalize_mode == "causal" and self.geometry is not None:
                self.model.normalize_decoder_weights(self.geometry)
            else:
                self.model.normalize_decoder_weights()

        self.step += 1

        with torch.no_grad():
            ev = compute_explained_variance(x, x_hat)
            ce_proxy = compute_cross_entropy_proxy(x, x_hat)

            # Dictionary health diagnostics (quick stats)
            pd = self.model.router.decoder.weight if not self.model.config.use_tied_decoders_parent else self.model.router.encoder.weight.T
            pdn = pd / (pd.norm(dim=0, keepdim=True) + 1e-8)
            Pcos = (pdn.T @ pdn)
            Poff = Pcos - torch.diag(torch.diag(Pcos))
            parent_cos_median = torch.median(Poff.abs()).item() if Poff.numel() > 0 else 0.0
            parent_cos_p95 = torch.quantile(Poff.abs().reshape(-1), 0.95).item() if Poff.numel() > 0 else 0.0
            parent_norm_median = torch.median(pd.norm(dim=0)).item()
            # Child stats aggregated
            child_norms = []
            child_offabs = []
            for sub in self.model.subspaces:
                cd = sub.decoder.weight if not self.model.config.use_tied_decoders_child else sub.encoder.weight.T
                cdn = cd / (cd.norm(dim=0, keepdim=True) + 1e-8)
                Ccos = (cdn.T @ cdn)
                Coff = Ccos - torch.diag(torch.diag(Ccos))
                if Coff.numel() > 0:
                    child_offabs.append(Coff.abs().reshape(-1))
                child_norms.append(cd.norm(dim=0))
            child_norms_cat = torch.cat(child_norms) if child_norms else torch.zeros(1, device=x.device)
            child_offabs_cat = torch.cat(child_offabs) if child_offabs else torch.zeros(1, device=x.device)
            child_cos_median = torch.median(child_offabs_cat).item()
            child_cos_p95 = torch.quantile(child_offabs_cat, 0.95).item()
            child_norm_median = torch.median(child_norms_cat).item()

            # L0 and dead-feature counts within batch
            l0_parents = torch.sum(parent_codes > 0, dim=1).float().mean().item()
            l0_children = torch.sum(child_codes > 0, dim=(1, 2)).float().mean().item()
            dead_parents = (torch.sum(parent_codes > 0, dim=0) == 0).sum().item()
            dead_children = (torch.sum(child_codes > 0, dim=(0, 1)) == 0).sum().item()
            # Activation distribution summaries (stand-in for histograms)
            parent_code_p95 = torch.quantile(parent_codes[parent_codes>0].flatten(), 0.95).item() if (parent_codes>0).any() else 0.0
            child_code_p95 = torch.quantile(child_codes[child_codes>0].flatten(), 0.95).item() if (child_codes>0).any() else 0.0

        log_metrics = {
            **loss_dict,
            "EV": ev,
            "1-EV": 1.0 - ev,
            "1-CE": ce_proxy,
            "parent_sparsity": metrics["parent_sparsity"],
            "child_sparsity": metrics["child_sparsity"],
            "active_parents_per_sample": metrics["active_parents_per_sample"],
            "active_children_per_sample": metrics["active_children_per_sample"],
            "router_temperature": metrics["router_temperature"],
            "l1_multiplier": self._get_l1_multiplier(self.step),
            "lr": self.optimizer.param_groups[0]["lr"],
            # Usage visibility
            "l0_parents_per_token": l0_parents,
            "l0_children_per_token": l0_children,
            "dead_parent_features": float(dead_parents),
            "dead_child_features": float(dead_children),
            # Dictionary health
            "parent_decoder_norm_median": parent_norm_median,
            "parent_decoder_cos_median": parent_cos_median,
            "parent_decoder_cos_p95": parent_cos_p95,
            "child_decoder_norm_median": child_norm_median,
            "child_decoder_cos_median": child_cos_median,
            "child_decoder_cos_p95": child_cos_p95,
            # Activation summary
            "parent_code_p95": parent_code_p95,
            "child_code_p95": child_code_p95,
        }
        return {k: (v.item() if torch.is_tensor(v) else float(v)) for k, v in log_metrics.items()}

    def _normalize_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if self.config.get("data", {}).get("unit_norm", False):
            batch = F.normalize(batch, dim=-1)
        return batch

    def _get_l1_multiplier(self, step: int) -> float:
        """Get L1 warmup multiplier: ramp from 0 → 1 over l1_warmup_steps."""
        l1_warmup_steps = self.config["training"].get("l1_warmup_steps", 0)
        if l1_warmup_steps <= 0:
            return 1.0
        return min(1.0, step / l1_warmup_steps)

    def _compute_total_loss(
        self,
        batch: torch.Tensor,
        x_hat: torch.Tensor,
        metrics: Dict[str, torch.Tensor],
        stage: str,
        parent_codes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Support causal space reconstruction based on config
        recon_space = (
            self.config.get("losses", {}).get("reconstruction_space", "euclidean")
        )
        if recon_space == "causal" and self.geometry is not None:
            self.geometry.to(str(batch.device))
            recon_loss = F.mse_loss(
                self.geometry.whiten(x_hat), self.geometry.whiten(batch)
            )
            if parent_codes is not None:
                parent_recon = self.model.router.decode(
                    parent_codes, self.model.config.use_tied_decoders_parent
                )
                if self.model.config.use_decoder_bias:
                    parent_recon += self.model.decoder_bias
                top_level_recon = F.mse_loss(
                    self.geometry.whiten(parent_recon), self.geometry.whiten(batch)
                )
            else:
                top_level_recon = metrics.get("top_level_recon_loss", torch.tensor(0.0))
        else:
            recon_loss = metrics["reconstruction_loss"]
            top_level_recon = metrics.get("top_level_recon_loss", torch.tensor(0.0))

        # Apply L1 warmup
        l1_mult = self._get_l1_multiplier(self.step)
        l1_loss = l1_mult * (metrics["l1_parent"] + metrics["l1_child"])

        top_level_loss = torch.zeros((), device=batch.device)
        # Allow stage-specific top-level reconstruction weighting
        beta = getattr(self.model.config, "top_level_beta", 0.0)
        tcfg = self.config.get("training", {}).get("teacher_init", {})
        if stage == "stabilize":
            beta = float(tcfg.get("stage_A", {}).get("top_level_beta", beta))
        elif stage == "adapt":
            beta = float(tcfg.get("stage_B", {}).get("top_level_beta", beta))
        if beta > 0:
            top_level_loss = beta * top_level_recon

        biorth_loss = metrics.get("biorth_penalty", torch.zeros((), device=batch.device))

        causal_ortho_loss = torch.zeros((), device=batch.device)
        # Support stage-specific causal orthogonality control for ablations
        stage_conf = self.config.get("training", {}).get("teacher_init", {})
        enable_causal = False
        if stage == "stabilize":
            enable_causal = stage_conf.get("stage_A", {}).get(
                "enable_causal_ortho", True
            )
        elif stage == "adapt":
            enable_causal = stage_conf.get("stage_B", {}).get(
                "enable_causal_ortho", False
            )
        if (
            enable_causal
            and hasattr(self.model.config, "causal_ortho_lambda")
            and self.model.config.causal_ortho_lambda > 0
            and self.geometry is not None
        ):
            self.geometry.to(str(batch.device))
            causal_ortho_loss = self.model.compute_causal_orthogonality_penalty(
                self.geometry
            )

        total = recon_loss + l1_loss + top_level_loss + biorth_loss + causal_ortho_loss
        return {
            "total_loss": total,
            "recon_loss": recon_loss,
            "l1_loss": l1_loss,
            "top_level_loss": top_level_loss,
            "biorth_loss": biorth_loss,
            "causal_ortho_loss": causal_ortho_loss,
        }

    def _optimizer_step(self, total_loss: torch.Tensor):
        total_loss.backward()
        if self.grad_clip is not None:
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def _run_stage(
        self,
        train_iter,
        train_loader,
        val_loader,
        steps: int,
        stage: str,
        history: Dict[str, List[float]],
        start_step: int = 0,
        desc: Optional[str] = None,
        sanity_checker=None,
    ):
        pbar = tqdm(range(steps), desc=desc or stage)
        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(next(self.model.parameters()).device)

            metrics = self.train_step(batch, stage=stage)

            pbar.set_postfix(
                {
                    "Loss": f"{metrics['total_loss']:.4f}",
                    "1-EV": f"{metrics['1-EV']:.4f}",
                    "τ": f"{metrics.get('router_temperature', 0.0):.2f}",
                    "L1": f"{metrics.get('l1_multiplier', 1.0):.2f}",
                }
            )

            global_step = start_step + step
            
            # Run sanity checks if enabled
            if sanity_checker and val_loader:
                current_ev = 1.0 - metrics['1-EV']
                should_restart = sanity_checker.check_and_fix(global_step, val_loader, current_ev)
                if should_restart:
                    logger.info("🔄 Sanity checker requested training restart")
                    raise TrainingRestartException("Sanity checker applied fixes - restart needed")
            # Logging
            if step % self.config["logging"]["log_every"] == 0:
                history["train_loss"].append(metrics["total_loss"])
                history["step"].append(global_step)
                if "stage" in history:
                    history["stage"].append(stage)
                if self.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in metrics.items()}, step=global_step
                    )
                    if stage != "main":
                        wandb.log({"stage": stage}, step=global_step)
            
            # Validation (less frequent than logging)
            val_every = self.config["training"].get("val_every", self.config["logging"]["log_every"])
            if step % val_every == 0 and val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_loss"].append(val_metrics["total_loss"])
                if self.use_wandb:
                    wandb.log(
                        {f"val/{k}": v for k, v in val_metrics.items()},
                        step=global_step,
                    )

            if (
                self.config["logging"].get("checkpoint_every") is not None
                and global_step % self.config["logging"]["checkpoint_every"] == 0
            ):
                self.save_checkpoint(global_step, stage=stage)

        return train_iter

    def train_baseline(
        self, train_loader, val_loader=None, total_steps: int = 7000, sanity_checker=None
    ) -> Dict[str, List[float]]:
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

        history = {"train_loss": [], "val_loss": [], "step": []}
        train_iter = iter(train_loader)
        try:
            self._run_stage(
                train_iter,
                train_loader,
                val_loader,
                total_steps,
                stage="main",
                history=history,
                start_step=0,
                desc="Baseline Training",
                sanity_checker=sanity_checker,
            )
        except TrainingRestartException as e:
            logger.info(f"Training interrupted for restart: {e}")
            raise  # let the calling phase script handle the restart
        logger.info("Baseline training completed")
        return history

    def train_teacher_init(
        self,
        train_loader,
        val_loader=None,
        stabilize_steps: int = 1500,
        adapt_steps: int = 8500,
        sanity_checker=None,
    ) -> Dict[str, List[float]]:
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
        logger.info(
            f"Training teacher-initialized H-SAE: {stabilize_steps} stabilize + {adapt_steps} adapt = {total_steps} total steps"
        )

        history = {"train_loss": [], "val_loss": [], "step": [], "stage": []}
        train_iter = iter(train_loader)

        try:
            stage_A_cfg = (
                self.config.get("training", {})
                .get("teacher_init", {})
                .get("stage_A", {})
            )
            logger.info("Stage A: Stabilizing with frozen decoder")
            if stage_A_cfg.get("freeze_decoder", True):
                self._freeze_decoder_weights()
            else:
                self._unfreeze_decoder_weights()
            train_iter = self._run_stage(
                train_iter,
                train_loader,
                val_loader,
                stabilize_steps,
                stage="stabilize",
                history=history,
                start_step=0,
                desc="Stage A: Stabilize",
                sanity_checker=sanity_checker,
            )

            stage_B_cfg = (
                self.config.get("training", {})
                .get("teacher_init", {})
                .get("stage_B", {})
            )
            logger.info("Stage B: Adapting with unfrozen decoder")
            if stage_B_cfg.get("freeze_decoder", False):
                self._freeze_decoder_weights()
            else:
                self._unfreeze_decoder_weights()
            decoder_lr_mult = (
                self.config["training"].get("teacher_init", {}).get("decoder_lr_mult", 1.0)
            )
            if decoder_lr_mult != 1.0:
                self._set_decoder_lr(self.config["training"]["lr"] * decoder_lr_mult)

            self._run_stage(
                train_iter,
                train_loader,
                val_loader,
                adapt_steps,
                stage="adapt",
                history=history,
                start_step=stabilize_steps,
                desc="Stage B: Adapt",
                sanity_checker=sanity_checker,
            )
        except TrainingRestartException as e:
            logger.info(f"Training interrupted for restart: {e}")
            raise  # let the calling phase script handle the restart

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

                if self.config.get("data", {}).get("unit_norm", False):
                    batch = F.normalize(batch, dim=-1)

                x_hat, (parent_codes, child_codes), metrics = self.model(batch)

                # Compute validation metrics
                recon_loss = metrics["reconstruction_loss"]
                l1_loss = metrics["l1_parent"] + metrics["l1_child"]
                biorth_loss = metrics.get("biorth_penalty", 0.0)
                total_loss = recon_loss + l1_loss + biorth_loss

                # Always use standard EV as primary metric (comparable to flat SAEs)
                ev = compute_explained_variance(batch, x_hat)
                ce_proxy = compute_cross_entropy_proxy(batch, x_hat)

                val_metrics.append(
                    {
                        "total_loss": total_loss.item(),
                        "recon_loss": recon_loss.item(),
                        "EV": ev,
                        "1-EV": 1.0 - ev,
                        "1-CE": ce_proxy,
                        "parent_sparsity": metrics["parent_sparsity"].item(),
                        "child_sparsity": metrics["child_sparsity"].item(),
                    }
                )

        # Average metrics
        if not val_metrics:
            # Return default metrics if no validation batches were processed
            logger.warning("No validation batches processed - returning default metrics")
            return {
                "total_loss": float('inf'),
                "recon_loss": float('inf'),
                "EV": 0.0,
                "1-EV": 1.0,
                "1-CE": float('inf'),
                "parent_sparsity": 0.0,
                "child_sparsity": 0.0,
            }
        
        avg_metrics = {}
        for key in val_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in val_metrics) / len(val_metrics)

        return avg_metrics

    def _freeze_decoder_weights(self):
        """Freeze decoder weights for stabilization phase."""
        parts = self.config.get("hsae", {}).get(
            "freeze_parts", ["parent_decoder", "child_decoders", "projectors"]
        )
        if "parent_decoder" in parts and not self.model.config.use_tied_decoders_parent:
            self.model.router.decoder.weight.requires_grad = False
        for sub in self.model.subspaces:
            if "child_decoders" in parts and not self.model.config.use_tied_decoders_child:
                sub.decoder.weight.requires_grad = False
            if "projectors" in parts and not self.model.config.tie_projectors:
                sub.up_projector.weight.requires_grad = False
                sub.down_projector.weight.requires_grad = False

    def _unfreeze_decoder_weights(self):
        """Unfreeze decoder weights for adaptation phase."""
        parts = self.config.get("hsae", {}).get(
            "freeze_parts", ["parent_decoder", "child_decoders", "projectors"]
        )
        if "parent_decoder" in parts and not self.model.config.use_tied_decoders_parent:
            self.model.router.decoder.weight.requires_grad = True
        for sub in self.model.subspaces:
            if "child_decoders" in parts and not self.model.config.use_tied_decoders_child:
                sub.decoder.weight.requires_grad = True
            if "projectors" in parts and not self.model.config.tie_projectors:
                sub.up_projector.weight.requires_grad = True
                sub.down_projector.weight.requires_grad = True

    def _set_decoder_lr(self, lr: float):
        """Set different learning rate for decoder parameters."""
        decoder_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if "decoder" in name or "up_projector" in name:
                decoder_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = AdamW(
            [
                {"params": other_params, "lr": self.config["training"]["lr"]},
                {"params": decoder_params, "lr": lr},
            ],
            weight_decay=self.config["training"]["weight_decay"],
        )

    def save_checkpoint(self, step: int, stage: str = "main"):
        """Save model checkpoint."""
        checkpoint_dir = self.log_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"hsae_step_{step}_{stage}.pt"

        torch.save(
            {
                "step": step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "stage": stage,
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Clean up old checkpoints
        if hasattr(self.config["logging"], "keep_last_k"):
            self._cleanup_checkpoints(
                checkpoint_dir, keep_last_k=self.config["logging"]["keep_last_k"]
            )

    def _cleanup_checkpoints(self, checkpoint_dir: Path, keep_last_k: int = 5):
        """Keep only the last k checkpoints."""
        checkpoints = list(checkpoint_dir.glob("hsae_step_*.pt"))
        if len(checkpoints) > keep_last_k:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.stem.split("_")[2]))
            # Remove old checkpoints
            for checkpoint in checkpoints[:-keep_last_k]:
                checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            checkpoint_path, map_location=next(self.model.parameters()).device
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        logger.info(f"Loaded checkpoint from step {self.step}")
        return checkpoint


class H5ActivationDataset(Dataset):
    def __init__(self, h5_path: str, groups: Optional[List[str]] = None, last_token_only: bool = False):
        self.h5 = h5py.File(h5_path, "r")
        self.keys = groups or list(self.h5.keys())
        self.index: List[Tuple[str, str, int]] = []
        for k in self.keys:
            dset = self.h5[k]["pos"]
            n = dset.shape[0]
            self.index.extend([(k, "pos", i) for i in range(n)])
        self.last_token_only = last_token_only

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        k, split, i = self.index[idx]
        x = self.h5[k][split][i]
        return torch.from_numpy(x)


def create_data_loader(
    activation_tensors: Union[List[torch.Tensor], torch.Tensor, Dataset],
    batch_size: int,
    device: str,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """Create a ``DataLoader`` for activation tensors or datasets.

    The original implementation expected in-memory tensors.  This version also
    accepts any :class:`~torch.utils.data.Dataset` (for example one that
    streams activations from disk) and will construct a loader directly from it
    without concatenating tensors.
    """

    if isinstance(activation_tensors, Dataset):
        ds = activation_tensors
    else:
        X = (
            activation_tensors
            if isinstance(activation_tensors, torch.Tensor)
            else torch.cat(activation_tensors, dim=0)
        )
        ds = TensorDataset(X)

    is_cuda = str(device).startswith("cuda")
    if num_workers is None:
        num_workers = max(4, mp.cpu_count() // 2)
    if pin_memory is None:
        pin_memory = is_cuda
    if drop_last is None:
        drop_last = len(ds) >= batch_size  # only drop when we have >=1 full batch

    # Auto-clamp batch size to dataset length to avoid zero-batch loaders
    effective_bs = min(batch_size, len(ds)) if len(ds) > 0 else batch_size

    loader = DataLoader(
        ds,
        batch_size=effective_bs,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=drop_last,
    )
    return loader
