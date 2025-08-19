import torch

from polytope_hsae.models import HierarchicalSAE, HSAEConfig
from polytope_hsae.training import HSAETrainer


def _build_trainer(tmp_path, use_scheduler=False):
    model_cfg = HSAEConfig(
        input_dim=4,
        n_parents=2,
        topk_parent=1,
        subspace_dim=2,
        n_children_per_parent=2,
        topk_child=1,
    )
    model = HierarchicalSAE(model_cfg)
    train_cfg = {
        "training": {"lr": 0.01, "weight_decay": 0.0},
        "logging": {"save_dir": str(tmp_path), "log_every": 1, "checkpoint_every": 1},
    }
    if use_scheduler:
        train_cfg["training"]["warmup_steps"] = 2
        train_cfg["training"]["total_steps"] = 4
    trainer = HSAETrainer(model, train_cfg, use_wandb=False)
    return model, trainer


def test_checkpoint_save_and_load(tmp_path):
    model, trainer = _build_trainer(tmp_path)
    batch = torch.randn(2, 4)
    trainer.train_step(batch)
    trainer.save_checkpoint(trainer.step)
    ckpt = tmp_path / "checkpoints" / f"hsae_step_{trainer.step}_main.pt"
    assert ckpt.exists()
    # load into new trainer
    new_model, new_trainer = _build_trainer(tmp_path)
    new_trainer.load_checkpoint(str(ckpt))
    for p1, p2 in zip(trainer.model.parameters(), new_trainer.model.parameters()):
        assert torch.allclose(p1, p2)


def test_scheduler_warmup(tmp_path):
    model, trainer = _build_trainer(tmp_path, use_scheduler=True)
    batch = torch.randn(2, 4)
    lrs = []
    for _ in range(4):
        trainer.train_step(batch)
        lrs.append(trainer.optimizer.param_groups[0]["lr"])
    assert lrs[0] < lrs[1]
    assert lrs[2] <= lrs[1]
