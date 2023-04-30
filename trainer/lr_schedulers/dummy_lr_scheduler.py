from dataclasses import dataclass

import torch
from accelerate.utils import DummyScheduler
from hydra.utils import instantiate
from omegaconf import II

try:
    import torch.distributed.nn

    has_distributed = True
except ImportError:
    has_distributed = False


@dataclass
class DummyLRSchedulerConfig:
    _target_: str = "trainer.lr_schedulers.dummy_lr_scheduler.instantiate_dummy_lr_scheduler"
    lr: float = II("optimizer.lr")
    lr_warmup_steps: int = 500
    total_num_steps: int = II("accelerator.max_steps")


def instantiate_dummy_lr_scheduler(cfg: DummyLRSchedulerConfig, optimizer):
    try:
        num_processes = torch.distributed.get_world_size()
    except RuntimeError:
        num_processes = 1
    return DummyScheduler(
        optimizer,
        total_num_steps=cfg.total_num_steps * num_processes,
        warmup_num_steps=cfg.lr_warmup_steps,
        warmup_max_lr=cfg.lr,
    )
