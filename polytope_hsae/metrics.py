"""
Evaluation metrics for SAE and H-SAE models.

This module implements standard metrics including explained variance, cross-entropy proxy,
purity, leakage, and steering-specific metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def compute_explained_variance(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """
    Compute explained variance: EV = 1 - ||x - x_hat||²_F / ||x - x_mean||²_F.

    Args:
        x: Original activations [batch_size, dim]
        x_hat: Reconstructed activations [batch_size, dim]

    Returns:
        Explained variance (higher is better)
    """
    x_mean = torch.mean(x, dim=0, keepdim=True)

    numerator = torch.sum((x - x_hat) ** 2)
    denominator = torch.sum((x - x_mean) ** 2)

    ev = 1.0 - (numerator / (denominator + 1e-8))
    return ev.item()


def compute_dual_explained_variance(
    x: torch.Tensor, 
    x_hat: torch.Tensor, 
    geometry=None,
    print_components: bool = False
) -> Dict[str, float]:
    """
    Compute both standard EV and geometry-aware EV (Mahalanobis).
    
    Args:
        x: Original activations [batch_size, dim] 
        x_hat: Reconstructed activations [batch_size, dim]
        geometry: CausalGeometry object (if None, only computes standard EV)
        print_components: Whether to print detailed breakdown
        
    Returns:
        Dictionary with 'standard_ev' and 'causal_ev' (if geometry provided)
    """
    results = {}
    
    # 1. STANDARD EV (PRIMARY) - apples-to-apples with flat SAEs
    x_mean = torch.mean(x, dim=0, keepdim=True)
    mse_standard = torch.sum((x - x_hat) ** 2)
    var_standard = torch.sum((x - x_mean) ** 2)
    ev_standard = 1.0 - (mse_standard / (var_standard + 1e-8))
    results['standard_ev'] = ev_standard.item()
    
    if print_components:
        logger.info(f"📊 STANDARD EV (Primary - comparable to flat SAEs):")
        logger.info(f"  MSE: {mse_standard.item():.6f}")
        logger.info(f"  Var: {var_standard.item():.6f}")
        logger.info(f"  EV:  {ev_standard.item():.6f}")
    
    # 2. GEOMETRY-AWARE EV (SECONDARY) - Mahalanobis metric
    if geometry is not None:
        # Whiten both x and x_hat, then compute EV in causal space
        x_whitened = geometry.whiten(x)
        x_hat_whitened = geometry.whiten(x_hat)
        x_mean_whitened = torch.mean(x_whitened, dim=0, keepdim=True)
        
        mse_causal = torch.sum((x_whitened - x_hat_whitened) ** 2)
        var_causal = torch.sum((x_whitened - x_mean_whitened) ** 2)
        ev_causal = 1.0 - (mse_causal / (var_causal + 1e-8))
        results['causal_ev'] = ev_causal.item()
        
        if print_components:
            logger.info(f"📊 CAUSAL EV (Secondary - Mahalanobis metric):")
            logger.info(f"  MSE (causal): {mse_causal.item():.6f}")
            logger.info(f"  Var (causal): {var_causal.item():.6f}")
            logger.info(f"  EV (causal):  {ev_causal.item():.6f}")
    
    return results


def compute_cross_entropy_proxy(
    x: torch.Tensor, x_hat: torch.Tensor, temperature: float = 1.0
) -> float:
    """
    Compute cross-entropy proxy as a measure of reconstruction quality.

    Args:
        x: Original activations [batch_size, dim]
        x_hat: Reconstructed activations [batch_size, dim]
        temperature: Temperature for softmax

    Returns:
        Cross-entropy proxy (lower is better)
    """
    # Normalize to probabilities
    p_true = F.softmax(x / temperature, dim=-1)
    log_p_pred = F.log_softmax(x_hat / temperature, dim=-1)

    # Cross-entropy
    ce = -torch.sum(p_true * log_p_pred, dim=-1).mean()
    return ce.item()


def compute_purity_metrics(
    parent_codes: torch.Tensor,
    child_codes: torch.Tensor,
    parent_labels: torch.Tensor,
    child_labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute purity metrics for hierarchical representations.

    Args:
        parent_codes: Parent activations [batch_size, n_parents]
        child_codes: Child activations [batch_size, n_parents, n_children]
        parent_labels: Parent concept labels [batch_size]
        child_labels: Child concept labels [batch_size]

    Returns:
        Dictionary with purity metrics
    """
    batch_size = parent_codes.shape[0]
    n_parents = parent_codes.shape[1]

    # Parent purity: do parent latents fire on correct parent contexts?
    parent_purity_scores = []
    parent_auc_scores = []

    for parent_idx in range(n_parents):
        # Get activations for this parent
        parent_activations = parent_codes[:, parent_idx].cpu().numpy()

        # Binary labels: 1 if this parent is active, 0 otherwise
        parent_binary_labels = (parent_labels == parent_idx).cpu().numpy()

        if np.sum(parent_binary_labels) > 0 and np.sum(1 - parent_binary_labels) > 0:
            # Compute AUC
            try:
                auc = roc_auc_score(parent_binary_labels, parent_activations)
                parent_auc_scores.append(auc)
            except ValueError:
                pass  # Skip if all labels are the same

            # Compute purity as mean activation on positive examples vs negative
            pos_mean = np.mean(parent_activations[parent_binary_labels == 1])
            neg_mean = np.mean(parent_activations[parent_binary_labels == 0])
            purity = pos_mean / (neg_mean + 1e-8)
            parent_purity_scores.append(purity)

    # Child purity: similar analysis for child latents
    child_purity_scores = []
    child_auc_scores = []

    for parent_idx in range(n_parents):
        for child_idx in range(child_codes.shape[2]):
            child_activations = child_codes[:, parent_idx, child_idx].cpu().numpy()

            # Binary labels: 1 if this child is active, 0 otherwise
            child_binary_labels = (
                (child_labels == (parent_idx * child_codes.shape[2] + child_idx))
                .cpu()
                .numpy()
            )

            if np.sum(child_binary_labels) > 0 and np.sum(1 - child_binary_labels) > 0:
                try:
                    auc = roc_auc_score(child_binary_labels, child_activations)
                    child_auc_scores.append(auc)
                except ValueError:
                    pass

                pos_mean = np.mean(child_activations[child_binary_labels == 1])
                neg_mean = np.mean(child_activations[child_binary_labels == 0])
                purity = pos_mean / (neg_mean + 1e-8)
                child_purity_scores.append(purity)

    return {
        "parent_purity_mean": np.mean(parent_purity_scores)
        if parent_purity_scores
        else 0.0,
        "parent_purity_std": np.std(parent_purity_scores)
        if parent_purity_scores
        else 0.0,
        "parent_auc_mean": np.mean(parent_auc_scores) if parent_auc_scores else 0.5,
        "parent_auc_std": np.std(parent_auc_scores) if parent_auc_scores else 0.0,
        "child_purity_mean": np.mean(child_purity_scores)
        if child_purity_scores
        else 0.0,
        "child_purity_std": np.std(child_purity_scores) if child_purity_scores else 0.0,
        "child_auc_mean": np.mean(child_auc_scores) if child_auc_scores else 0.5,
        "child_auc_std": np.std(child_auc_scores) if child_auc_scores else 0.0,
    }


def compute_leakage_metrics(
    parent_codes: torch.Tensor,
    child_codes: torch.Tensor,
    sparsity_threshold: float = 0.01,
) -> Dict[str, float]:
    """
    Compute leakage metrics for hierarchical representations.

    Args:
        parent_codes: Parent activations [batch_size, n_parents]
        child_codes: Child activations [batch_size, n_parents, n_children]
        sparsity_threshold: Threshold for considering a latent "active"

    Returns:
        Dictionary with leakage metrics
    """
    # Parent leakage: cross-parent activation
    parent_active = parent_codes > sparsity_threshold  # [batch_size, n_parents]
    parent_leakage_per_sample = torch.sum(
        parent_active, dim=1
    ).float()  # How many parents active per sample

    # Child leakage: cross-child activation within parent
    child_active = (
        child_codes > sparsity_threshold
    )  # [batch_size, n_parents, n_children]
    child_leakage_per_parent = torch.sum(
        child_active, dim=2
    ).float()  # [batch_size, n_parents]

    # Only consider child leakage when parent is active
    parent_active_expanded = parent_active.unsqueeze(-1)  # [batch_size, n_parents, 1]
    child_leakage_masked = child_leakage_per_parent * parent_active.float()

    return {
        "parent_leakage_mean": torch.mean(parent_leakage_per_sample).item(),
        "parent_leakage_std": torch.std(parent_leakage_per_sample).item(),
        "child_leakage_mean": torch.mean(
            child_leakage_masked[child_leakage_masked > 0]
        ).item()
        if torch.sum(child_leakage_masked > 0) > 0
        else 0.0,
        "child_leakage_std": torch.std(
            child_leakage_masked[child_leakage_masked > 0]
        ).item()
        if torch.sum(child_leakage_masked > 0) > 0
        else 0.0,
        "fraction_multi_parent": torch.mean(
            (parent_leakage_per_sample > 1).float()
        ).item(),
        "fraction_multi_child": torch.mean((child_leakage_masked > 1).float()).item(),
    }


def compute_split_absorb_rates(
    child_codes: torch.Tensor,
    parent_codes: torch.Tensor,
    usage_threshold: float = 0.001,
) -> Dict[str, float]:
    """
    Compute split and absorb rates for child latents.

    Args:
        child_codes: Child activations [batch_size, n_parents, n_children]
        parent_codes: Parent activations [batch_size, n_parents]
        usage_threshold: Minimum usage to consider a latent "used"

    Returns:
        Dictionary with split/absorb metrics
    """
    batch_size, n_parents, n_children = child_codes.shape

    # Usage statistics
    child_usage = torch.mean(child_codes > 0, dim=0)  # [n_parents, n_children]
    parent_usage = torch.mean(parent_codes > 0, dim=0)  # [n_parents]

    split_latents = 0  # Child latents that split (multiple children used)
    absorb_latents = 0  # Child latents that absorb (become parent-like)
    total_used_children = 0

    for parent_idx in range(n_parents):
        parent_used = parent_usage[parent_idx] > usage_threshold
        if not parent_used:
            continue

        children_usage = child_usage[parent_idx]  # [n_children]
        used_children = torch.sum(children_usage > usage_threshold).item()
        total_used_children += used_children

        if used_children > 1:
            split_latents += 1

        # Check for absorb: child usage similar to parent usage
        for child_idx in range(n_children):
            if children_usage[child_idx] > usage_threshold:
                # Compare child usage to parent usage
                usage_ratio = children_usage[child_idx] / (
                    parent_usage[parent_idx] + 1e-8
                )
                if usage_ratio > 0.8:  # Child used almost as much as parent
                    absorb_latents += 1

    n_used_parents = torch.sum(parent_usage > usage_threshold).item()

    return {
        "split_rate": split_latents / (n_used_parents + 1e-8),
        "absorb_rate": absorb_latents / (total_used_children + 1e-8),
        "avg_children_per_parent": total_used_children / (n_used_parents + 1e-8),
        "parent_usage_mean": torch.mean(parent_usage).item(),
        "child_usage_mean": torch.mean(child_usage).item(),
    }


def compute_steering_leakage(
    original_logits: torch.Tensor,
    steered_logits: torch.Tensor,
    target_tokens: List[int],
    non_target_tokens: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute steering leakage: off-target effects during concept steering.

    Args:
        original_logits: Original model logits [batch_size, vocab_size]
        steered_logits: Logits after steering intervention [batch_size, vocab_size]
        target_tokens: Token IDs that should be affected by steering
        non_target_tokens: Token IDs that should NOT be affected (optional)

    Returns:
        Dictionary with steering leakage metrics
    """
    logit_deltas = steered_logits - original_logits  # [batch_size, vocab_size]

    # Target effects (should be large)
    target_deltas = logit_deltas[:, target_tokens]  # [batch_size, n_targets]
    target_effect = torch.mean(torch.abs(target_deltas)).item()

    # Non-target effects (should be small)
    if non_target_tokens is not None:
        non_target_deltas = logit_deltas[:, non_target_tokens]
        non_target_effect = torch.mean(torch.abs(non_target_deltas)).item()
    else:
        # Use all tokens except targets
        mask = torch.ones(logit_deltas.shape[1], dtype=torch.bool)
        mask[target_tokens] = False
        non_target_deltas = logit_deltas[:, mask]
        non_target_effect = torch.mean(torch.abs(non_target_deltas)).item()

    # Steering precision: target effect / non-target effect
    steering_precision = target_effect / (non_target_effect + 1e-8)

    # Overall leakage: norm of non-target changes
    overall_leakage = torch.norm(non_target_deltas, dim=-1).mean().item()

    return {
        "target_effect": target_effect,
        "non_target_effect": non_target_effect,
        "steering_precision": steering_precision,
        "overall_leakage": overall_leakage,
        "leakage_ratio": non_target_effect / (target_effect + 1e-8),
    }


def compute_comprehensive_metrics(
    model,
    data_loader,
    concept_labels: Optional[torch.Tensor] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for H-SAE.

    Args:
        model: HierarchicalSAE model
        data_loader: Data loader with evaluation data
        concept_labels: Concept labels for purity computation (optional)
        device: Device to run evaluation on

    Returns:
        Dictionary with all evaluation metrics
    """
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if isinstance(batch, (list, tuple)):
                batch_data = batch[0].to(device)
                batch_labels = batch[1].to(device) if len(batch) > 1 else None
            else:
                batch_data = batch.to(device)
                batch_labels = None

            # Forward pass
            x_hat, (parent_codes, child_codes), model_metrics = model(batch_data)

            # Basic reconstruction metrics
            ev = compute_explained_variance(batch_data, x_hat)
            ce_proxy = compute_cross_entropy_proxy(batch_data, x_hat)

            # Leakage metrics
            leakage_metrics = compute_leakage_metrics(parent_codes, child_codes)

            # Split/absorb rates
            split_absorb_metrics = compute_split_absorb_rates(child_codes, parent_codes)

            # Purity metrics (if labels available)
            purity_metrics = {}
            if batch_labels is not None:
                # Assume labels are concept IDs
                parent_labels = batch_labels // child_codes.shape[2]  # Integer division
                child_labels = batch_labels
                purity_metrics = compute_purity_metrics(
                    parent_codes, child_codes, parent_labels, child_labels
                )

            # Combine all metrics
            batch_metrics = {
                "1-EV": 1.0 - ev,
                "1-CE": ce_proxy,
                **leakage_metrics,
                **split_absorb_metrics,
                **purity_metrics,
                **{
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in model_metrics.items()
                },
            }

            all_metrics.append(batch_metrics)

    # Average across batches
    final_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
            if values:
                final_metrics[key] = np.mean(values)
                final_metrics[f"{key}_std"] = np.std(values)

    return final_metrics


def log_metrics_summary(
    metrics: Dict[str, float], logger_instance: Optional[logging.Logger] = None
) -> None:
    """
    Log a summary of metrics in a readable format.

    Args:
        metrics: Dictionary of metrics to log
        logger_instance: Logger instance to use (defaults to module logger)
    """
    if logger_instance is None:
        logger_instance = logger

    logger_instance.info("=" * 50)
    logger_instance.info("METRICS SUMMARY")
    logger_instance.info("=" * 50)

    # Group metrics by category
    reconstruction_metrics = {
        k: v
        for k, v in metrics.items()
        if any(x in k.lower() for x in ["ev", "ce", "recon", "loss"])
    }
    sparsity_metrics = {
        k: v
        for k, v in metrics.items()
        if any(x in k.lower() for x in ["sparsity", "usage", "active"])
    }
    purity_metrics = {
        k: v
        for k, v in metrics.items()
        if any(x in k.lower() for x in ["purity", "auc", "leakage"])
    }
    hierarchy_metrics = {
        k: v
        for k, v in metrics.items()
        if any(x in k.lower() for x in ["split", "absorb", "parent", "child"])
    }

    # Log each category
    if reconstruction_metrics:
        logger_instance.info("Reconstruction Metrics:")
        for k, v in reconstruction_metrics.items():
            logger_instance.info(f"  {k}: {v:.6f}")

    if sparsity_metrics:
        logger_instance.info("Sparsity Metrics:")
        for k, v in sparsity_metrics.items():
            logger_instance.info(f"  {k}: {v:.6f}")

    if purity_metrics:
        logger_instance.info("Purity & Leakage Metrics:")
        for k, v in purity_metrics.items():
            logger_instance.info(f"  {k}: {v:.6f}")

    if hierarchy_metrics:
        logger_instance.info("Hierarchy Metrics:")
        for k, v in hierarchy_metrics.items():
            logger_instance.info(f"  {k}: {v:.6f}")

    logger_instance.info("=" * 50)
