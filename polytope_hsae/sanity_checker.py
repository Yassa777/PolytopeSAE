"""
Automated Training Sanity Checker for H-SAE Training

This module provides automated monitoring and fixing of common H-SAE training issues:
- Router too cold (routes not firing)
- Insufficient width (not enough parent/child choices)
- Insufficient capacity (subspace_dim too small)
- PCA upper bound validation

The system monitors training every 200 steps and can automatically restart training
with fixed configurations to save time and compute on RunPod.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class TrainingSanityChecker:
    """Automated sanity checker that monitors and fixes H-SAE training issues."""
    
    def __init__(self, config: Dict, model, trainer, exp_dir: Path):
        """
        Initialize sanity checker.
        
        Args:
            config: Training configuration
            model: H-SAE model
            trainer: HSAETrainer instance
            exp_dir: Experiment directory for saving checkpoints and logs
        """
        self.config = config
        self.model = model
        self.trainer = trainer
        self.exp_dir = exp_dir
        
        # Get sanity checker configuration with defaults
        scfg = config.get("sanity_checker", {})
        
        # Monitoring settings
        self.check_interval = scfg.get("check_interval", 600)  # Delayed from 200 to 600
        self.restart_count = 0
        self.max_restarts = scfg.get("max_restarts", 1)  # Reduced from 2 to 1
        
        # Gentler thresholds for Stage A with Top-K=1 and frozen decoders
        self.min_route_usage_pct = scfg.get("min_route_usage_pct", 15.0)  # Reduced from 50% to 15%
        self.min_active_dims_pct = scfg.get("min_active_dims_pct", 20.0)  # Reduced from 80% to 20%
        self.pca_ev_gap_threshold = scfg.get("pca_ev_gap_threshold", 0.15)  # If PCA EV - model EV > 0.15, increase capacity
        
        # Fix parameters
        self.router_temp_increase = 0.5  # Increase start temp by this much
        self.width_scale_factor = 1.5   # Multiply width by this factor
        self.capacity_increase = 32     # Add this many dims to subspace_dim
        
        # Tracking
        self.issue_history = []
        self.last_pca_ev = None
        
        logger.info(f"🔍 Sanity checker initialized with {self.max_restarts} max restarts")
        logger.info(f"    Check interval: {self.check_interval} steps")
        logger.info(f"    Min route usage: {self.min_route_usage_pct}% (Stage A friendly)")
        logger.info(f"    Min active dims: {self.min_active_dims_pct}% (Top-K=1 friendly)")
    
    def should_check(self, step: int) -> bool:
        """Check if we should run sanity checks at this step."""
        return step > 0 and step % self.check_interval == 0
    
    def check_and_fix(self, step: int, val_loader, current_ev: float) -> bool:
        """
        Run sanity checks and apply fixes if needed.
        
        Args:
            step: Current training step
            val_loader: Validation data loader
            current_ev: Current model EV performance
            
        Returns:
            True if training should restart, False to continue
        """
        if not self.should_check(step):
            return False
            
        logger.info(f"🔍 Running sanity checks at step {step}")
        
        # Diagnose issues
        issues = self.diagnose_issues(val_loader, current_ev, step)
        
        if not issues:
            logger.info("✅ All sanity checks passed - continuing training")
            return False
        
        # Log issues
        logger.warning(f"⚠️  Detected {len(issues)} training issues:")
        for issue in issues:
            logger.warning(f"  - {issue['type']}: {issue['description']}")
        
        # Apply fixes if we haven't exceeded restart limit
        if self.restart_count >= self.max_restarts:
            logger.warning(f"🛑 Max restarts ({self.max_restarts}) reached - continuing with current config")
            return False
        
        return self.apply_fixes(issues, step)
    
    def diagnose_issues(self, val_loader, current_ev: float, step: int) -> List[Dict]:
        """Diagnose training issues by analyzing model behavior."""
        issues = []
        
        # Collect validation statistics
        stats = self._collect_validation_stats(val_loader)
        
        # 1. Check route usage
        route_issues = self._check_route_usage(stats)
        issues.extend(route_issues)
        
        # 2. Check active dimensions
        active_dim_issues = self._check_active_dimensions(stats)
        issues.extend(active_dim_issues)
        
        # 3. Check PCA upper bound (every 1000 steps to save compute)
        if step % 1000 == 0:
            pca_issues = self._check_pca_upper_bound(val_loader, current_ev)
            issues.extend(pca_issues)
        
        # 4. Check for dead/saturated routes
        dead_route_issues = self._check_dead_routes(stats)
        issues.extend(dead_route_issues)
        
        # Store issue history
        self.issue_history.append({
            'step': step,
            'issues': issues,
            'stats': stats
        })
        
        return issues
    
    def _collect_validation_stats(self, val_loader) -> Dict:
        """Collect comprehensive validation statistics."""
        self.model.eval()
        
        parent_usage = torch.zeros(self.model.config.n_parents)
        child_usage = torch.zeros(self.model.config.n_parents, self.model.config.n_children_per_parent)
        total_samples = 0
        total_active_dims = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= 10:  # Limit to 10 batches for speed
                    break
                    
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(next(self.model.parameters()).device)
                
                # Forward pass
                _, (parent_codes, child_codes), _ = self.model(batch)
                
                # Track usage
                parent_active = (parent_codes > 0).float()  # [batch, n_parents]
                child_active = (child_codes > 0).float()    # [batch, n_parents, n_children]
                
                # Accumulate parent usage
                parent_usage += torch.sum(parent_active, dim=0).cpu()
                
                # Accumulate child usage
                child_usage += torch.sum(child_active, dim=0).cpu()
                
                # Count active dimensions per sample
                batch_active_dims = torch.sum(parent_active, dim=1) + torch.sum(child_active, dim=(1, 2))
                total_active_dims += torch.sum(batch_active_dims).item()
                total_samples += batch.shape[0]
        
        # Calculate statistics
        parent_usage_pct = (parent_usage > 0).float().mean().item() * 100
        child_usage_pct = (child_usage > 0).float().mean().item() * 100
        avg_active_dims = total_active_dims / total_samples if total_samples > 0 else 0
        
        # Theoretical max active dims
        theoretical_max = self.model.config.topk_parent + (
            self.model.config.topk_parent * self.model.config.topk_child * self.model.config.subspace_dim
        )
        
        stats = {
            'parent_usage_pct': parent_usage_pct,
            'child_usage_pct': child_usage_pct,
            'avg_active_dims': avg_active_dims,
            'theoretical_max_dims': theoretical_max,
            'active_dims_pct': (avg_active_dims / theoretical_max * 100) if theoretical_max > 0 else 0,
            'parent_usage_counts': parent_usage,
            'child_usage_counts': child_usage,
            'total_samples': total_samples
        }
        
        # Log current stats
        logger.info(f"📊 Current Stats:")
        logger.info(f"  Parent usage: {parent_usage_pct:.1f}% ({torch.sum(parent_usage > 0).item()}/{self.model.config.n_parents})")
        logger.info(f"  Child usage: {child_usage_pct:.1f}%")
        logger.info(f"  Active dims: {avg_active_dims:.1f}/{theoretical_max} ({stats['active_dims_pct']:.1f}%)")
        
        return stats
    
    def _check_route_usage(self, stats: Dict) -> List[Dict]:
        """Check if routes are being used sufficiently."""
        issues = []
        
        if stats['parent_usage_pct'] < self.min_route_usage_pct:
            issues.append({
                'type': 'low_parent_usage',
                'description': f"Only {stats['parent_usage_pct']:.1f}% of parents firing (< {self.min_route_usage_pct}%)",
                'severity': 'high',
                'suggested_fix': 'increase_router_temp'
            })
        
        if stats['child_usage_pct'] < self.min_route_usage_pct:
            issues.append({
                'type': 'low_child_usage', 
                'description': f"Only {stats['child_usage_pct']:.1f}% of children firing (< {self.min_route_usage_pct}%)",
                'severity': 'high',
                'suggested_fix': 'increase_router_temp'
            })
        
        return issues
    
    def _check_active_dimensions(self, stats: Dict) -> List[Dict]:
        """Check if we're using enough active dimensions per token."""
        issues = []
        
        if stats['active_dims_pct'] < self.min_active_dims_pct:
            issues.append({
                'type': 'low_active_dims',
                'description': f"Only {stats['avg_active_dims']:.1f}/{stats['theoretical_max_dims']} active dims ({stats['active_dims_pct']:.1f}% < {self.min_active_dims_pct}%)",
                'severity': 'medium',
                'suggested_fix': 'increase_width'
            })
        
        return issues
    
    def _check_pca_upper_bound(self, val_loader, current_ev: float) -> List[Dict]:
        """Check PCA upper bound to validate capacity."""
        issues = []
        
        try:
            # Collect validation data for PCA
            val_data = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    if batch_idx >= 20:  # Limit for memory
                        break
                    if isinstance(batch, (list, tuple)):
                        batch = batch[0]
                    val_data.append(batch.cpu().numpy())
            
            if not val_data:
                return issues
                
            X = np.concatenate(val_data, axis=0)
            
            # Fit PCA with current capacity
            n_components = min(
                self.model.config.topk_parent * self.model.config.subspace_dim + self.model.config.topk_parent,
                X.shape[1] - 1,
                X.shape[0] - 1
            )
            
            if n_components < 10:
                logger.warning("⚠️  Too few components for meaningful PCA check")
                return issues
            
            pca = PCA(n_components=n_components)
            X_transformed = pca.fit_transform(X)
            X_reconstructed = pca.inverse_transform(X_transformed)
            
            # Compute PCA EV
            mse = np.mean((X - X_reconstructed) ** 2)
            var = np.var(X)
            pca_ev = 1.0 - (mse / (var + 1e-8))
            
            self.last_pca_ev = pca_ev
            
            logger.info(f"📈 PCA Upper Bound: {pca_ev:.3f} (current model: {current_ev:.3f})")
            
            # Check if there's a significant gap
            if pca_ev - current_ev > self.pca_ev_gap_threshold:
                issues.append({
                    'type': 'capacity_bottleneck',
                    'description': f"PCA@{n_components} achieves {pca_ev:.3f} EV vs model {current_ev:.3f} (gap: {pca_ev - current_ev:.3f})",
                    'severity': 'medium',
                    'suggested_fix': 'increase_capacity'
                })
                
        except Exception as e:
            logger.warning(f"⚠️  PCA check failed: {e}")
        
        return issues
    
    def _check_dead_routes(self, stats: Dict) -> List[Dict]:
        """Check for completely dead routes."""
        issues = []
        
        # Count completely unused parents
        dead_parents = torch.sum(stats['parent_usage_counts'] == 0).item()
        if dead_parents > self.model.config.n_parents * 0.3:  # > 30% dead
            issues.append({
                'type': 'dead_parents',
                'description': f"{dead_parents}/{self.model.config.n_parents} parents never fired",
                'severity': 'high',
                'suggested_fix': 'increase_router_temp'
            })
        
        return issues
    
    def apply_fixes(self, issues: List[Dict], step: int) -> bool:
        """Apply fixes for detected issues and restart training."""
        logger.info(f"🔧 Applying fixes for {len(issues)} issues...")
        
        # Determine primary fix needed
        fix_type = self._determine_primary_fix(issues)
        
        # Save checkpoint before restarting
        checkpoint_path = self._save_restart_checkpoint(step)
        
        # Apply the fix to config
        old_config = self.config.copy()
        success = self._apply_config_fix(fix_type)
        
        if not success:
            logger.error("❌ Failed to apply config fix")
            return False
        
        # Log the fix
        self._log_fix_applied(fix_type, old_config, step)
        
        self.restart_count += 1
        logger.info(f"🔄 Restarting training (attempt {self.restart_count}/{self.max_restarts})")
        
        return True  # Signal to restart training
    
    def _determine_primary_fix(self, issues: List[Dict]) -> str:
        """Determine the primary fix to apply based on issue severity and type."""
        # Priority order: router_temp > width > capacity
        for issue in issues:
            if issue['suggested_fix'] == 'increase_router_temp':
                return 'increase_router_temp'
        
        for issue in issues:
            if issue['suggested_fix'] == 'increase_width':
                return 'increase_width'
        
        for issue in issues:
            if issue['suggested_fix'] == 'increase_capacity':
                return 'increase_capacity'
        
        return 'increase_router_temp'  # Default
    
    def _apply_config_fix(self, fix_type: str) -> bool:
        """Apply the specified fix to the configuration."""
        try:
            if fix_type == 'increase_router_temp':
                old_start = self.config['hsae']['router_temp']['start']
                new_start = old_start + self.router_temp_increase
                self.config['hsae']['router_temp']['start'] = new_start
                logger.info(f"🌡️  Increased router temp: {old_start} → {new_start}")
                
            elif fix_type == 'increase_width':
                old_m_top = self.config['hsae']['m_top']
                old_m_low = self.config['hsae']['m_low']
                new_m_top = int(old_m_top * self.width_scale_factor)
                new_m_low = int(old_m_low * self.width_scale_factor)
                self.config['hsae']['m_top'] = new_m_top
                self.config['hsae']['m_low'] = new_m_low
                logger.info(f"📏 Increased width: m_top {old_m_top} → {new_m_top}, m_low {old_m_low} → {new_m_low}")
                
            elif fix_type == 'increase_capacity':
                old_dim = self.config['hsae']['subspace_dim']
                new_dim = old_dim + self.capacity_increase
                self.config['hsae']['subspace_dim'] = new_dim
                logger.info(f"🔧 Increased capacity: subspace_dim {old_dim} → {new_dim}")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to apply {fix_type}: {e}")
            return False
    
    def _save_restart_checkpoint(self, step: int) -> str:
        """Save checkpoint before restarting."""
        checkpoint_path = self.exp_dir / f"sanity_restart_checkpoint_step_{step}.pt"
        
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict(),
                'step': step,
                'config': self.config,
                'restart_count': self.restart_count,
                'issue_history': self.issue_history
            }, checkpoint_path)
            
            logger.info(f"💾 Saved restart checkpoint: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            return ""
    
    def _log_fix_applied(self, fix_type: str, old_config: Dict, step: int):
        """Log the fix that was applied."""
        fix_log = {
            'step': step,
            'restart_count': self.restart_count,
            'fix_type': fix_type,
            'old_config': {
                'router_temp': old_config['hsae']['router_temp'],
                'm_top': old_config['hsae']['m_top'],
                'm_low': old_config['hsae']['m_low'],
                'subspace_dim': old_config['hsae']['subspace_dim']
            },
            'new_config': {
                'router_temp': self.config['hsae']['router_temp'],
                'm_top': self.config['hsae']['m_top'],
                'm_low': self.config['hsae']['m_low'],
                'subspace_dim': self.config['hsae']['subspace_dim']
            },
            'issues_detected': self.issue_history[-1]['issues'] if self.issue_history else []
        }
        
        # Save fix log
        fix_log_path = self.exp_dir / "sanity_fix_log.json"
        
        try:
            # Load existing log or create new
            if fix_log_path.exists():
                with open(fix_log_path, 'r') as f:
                    all_fixes = json.load(f)
            else:
                all_fixes = []
            
            all_fixes.append(fix_log)
            
            with open(fix_log_path, 'w') as f:
                json.dump(all_fixes, f, indent=2)
                
            logger.info(f"📝 Logged fix to {fix_log_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log fix: {e}")